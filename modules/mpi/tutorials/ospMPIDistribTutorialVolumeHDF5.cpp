// Copyright 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/* This larger example shows how to use the MPIDistributedDevice to write an
 * interactive rendering application, which shows a UI on rank 0 and uses
 * all ranks in the MPI world for data loading and rendering. Each rank
 * generates a local sub-brick of volume data, as if rendering some
 * large distributed dataset.
 */

#include <imgui.h>
#include <mpi.h>
#include <iterator>
#include <memory>
#include <random>
#include "wombat.h"
#include "ChomboHDF5Reader.h"
#include "GLFWDistribOSPRayWindow.h"
#include "ospray/ospray_cpp.h"
#include "ospray/ospray_cpp/ext/rkcommon.h"
#include "ospray/ospray_util.h"
//#include "GravitySpheresVolume.h"

using namespace ospray;
using namespace rkcommon;
using namespace rkcommon::math;

struct VolumeBrick
{
  // the volume data itself
  cpp::Volume brick;
  cpp::VolumetricModel model;
  cpp::Group group;
  cpp::Instance instance;
  // the bounds of the owned portion of data
  std::vector<box3f> bounds;
  // the full bounds of the owned portion + ghost voxels
  box3f ghostBounds;
};

static box3f worldBounds;

// Generate the rank's local volume brick
VolumeBrick makeLocalVolume(const int mpiRank, const int mpiWorldSize);

int main(int argc, char **argv)
{
  int mpiThreadCapability = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiThreadCapability);
  if (mpiThreadCapability != MPI_THREAD_MULTIPLE
      && mpiThreadCapability != MPI_THREAD_SERIALIZED) {
    fprintf(stderr,
        "OSPRay requires the MPI runtime to support thread "
        "multiple or thread serialized.\n");
    return 1;
  }

  int mpiRank = 0;
  int mpiWorldSize = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiWorldSize);

  if (argc < 3) {
    if (mpiRank == 0) {
      std::cerr << "Usage: mpiexec [MPI options] " << argv[0]
                << " <hdf5 file> <component name>" << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  bool trustInWombat = true;
  if (argc > 3 && !strcmp(argv[3], "-noWombatNoCry")) {
      std::cerr << "NO WOMBAT!" << std::endl;
      trustInWombat = false;
  }

  std::cout << "OSPRay rank " << mpiRank << "/" << mpiWorldSize << "\n";

  // load the MPI module, and select the MPI distributed device. Here we
  // do not call ospInit, as we want to explicitly pick the distributed
  // device. This can also be done by passing --osp:mpi-distributed when
  // using ospInit, however if the user doesn't pass this argument your
  // application will likely not behave as expected
  ospLoadModule("mpi");

  {
    cpp::Device mpiDevice("mpiDistributed");
    mpiDevice.commit();
    mpiDevice.setCurrent();

    // set an error callback to catch any OSPRay errors and exit the application
    ospDeviceSetErrorCallback(
        mpiDevice.handle(),
        [](void *, OSPError error, const char *errorDetails) {
          std::cerr << "OSPRay error: " << errorDetails << std::endl;
          exit(error);
        },
        nullptr);

    std::cerr << "ABOUT TO READ" << std::endl;

    // open the hdf5 file, read the data and create the ospray Volume object
    ChomboHDF5::Reader hdf5reader(argv[1], mpiRank, mpiWorldSize, argv[2]);
    std::cerr << "READ!" << std::endl;

    VolumeBrick brick;
    brick.brick = hdf5reader.getVolume();
    worldBounds = hdf5reader.getDomainBounds();
    std::cerr << "WB = " << worldBounds << std::endl;

    const std::vector<box3f> &myregions = hdf5reader.getMyRegions();

    if (trustInWombat) {
      std::vector<box3f> wombatRegions;

      //prep input to wombat, sadly just a reformatting of the existing AMR metadata
      std::vector<wombat::Level> levels;
      std::vector<vec3i> levelDims;
      box3i l0extent = hdf5reader.getDomainBounds_int();
      vec3i ldims = l0extent.upper - l0extent.lower;
      int prevRefRatio = 1;
      for (auto n : hdf5reader.getMyRefRatios())
      {
        wombat::Level l;
        l.refinement[0] = n;
        l.refinement[1] = n;
        l.refinement[2] = n;
        levels.push_back(l);
        std::cerr << "L " << levelDims.size() << " " << (ldims+1) * prevRefRatio << std::endl;
        levelDims.push_back((ldims+1) * prevRefRatio);
        prevRefRatio = prevRefRatio * n;
      }
      std::vector<wombat::Box> boxes;
      const std::vector<int> &blockOwners = hdf5reader.getRankDataOwner();
      const std::vector<int> &blockLevels = hdf5reader.getBlockLevels();
      int index = 0;
      for (auto n : hdf5reader.getBlockBounds())
      {
        wombat::Box b;
        b.owningrank = blockOwners[index];
        b.level = blockLevels[index];
        b.origin[0] = n.lower.x;
        b.origin[1] = n.lower.y;
        b.origin[2] = n.lower.z;
        b.dims[0] = n.upper.x-n.lower.x; //TODO +1?
        b.dims[1] = n.upper.y-n.lower.y;
        b.dims[2] = n.upper.z-n.lower.z;
        //std::cerr << index << " " << b.owningrank << " " << b.level << " "
        //          << b.origin[0] << "," << b.origin[1] << "," << b.origin[2] << ","
        //          << b.dims[0] << "," << b.dims[1] << "," << b.dims[2] <<
        //             std::endl;
        boxes.push_back(b);
        index++;
      }

      std::vector<wombat::Box> sboxes;
      //run wombat to derive a set of convex regions
      //std::cerr << rank << " geneology [" << std::endl;
      wombat::geneology(levels, boxes, mpiRank);
      //std::cerr << rank << " ] geneology "<< std::endl;
      //std::cerr << rank << " convexify ["<< std::endl;
      wombat::convexify(levels, boxes, sboxes);
      //std::cerr << rank << " ] convexify "<< std::endl;

      //from that run, tell compositor the regions that this rank owns in worldspace
      for (int i = 0; i < sboxes.size(); ++i)
      {
        wombat::Box b = sboxes[i];
        if (b.owningrank == mpiRank)
        {
          vec3i bs = b.origin;
          vec3i be = b.dims;
          vec3f bx0 = (vec3f(bs)/vec3f(levelDims[b.level])) * worldBounds.upper;
          vec3f bx1 = bx0 + (vec3f(be)/vec3f(levelDims[b.level])) * worldBounds.upper;
          wombatRegions.push_back(box3f(vec3f(bx0.x,bx0.y,bx0.z),vec3f(bx1.x,bx1.y,bx1.z)));
          //std::cerr << i << " " << b.owningrank << " " << b.level << " "
          //          << bs << ".." << be << "->"
          //          << bx0 << "," << bx1 << std::endl;
        }
      }
      brick.bounds = wombatRegions;

    } else {
      brick.bounds = myregions;
    }
    //brick.brick.setParam("method", OSP_AMR_FINEST);
    brick.brick.commit();
    brick.model = cpp::VolumetricModel(brick.brick);
    cpp::TransferFunction tfn("piecewiseLinear");
    std::vector<vec3f> colors = {vec3f(0.f, 0.f, 0.f), vec3f(0.f, 0.f, 1.f), vec3f(0.f, 0.f, 0.f), vec3f(0.f, 0.f, 0.f)};
    std::vector<float> opacities = {0.0f, 0.5f, 0.0f, 0.0f};
    tfn.setParam("color", cpp::CopiedData(colors));
    tfn.setParam("opacity", cpp::CopiedData(opacities));
    vec2f valueRange = vec2f(0.2, 0.96435);
    tfn.setParam("valueRange", valueRange);
    tfn.commit();
    brick.model.setParam("transferFunction", tfn);
    brick.model.commit();

    brick.group = cpp::Group();
    brick.group.setParam("volume", cpp::CopiedData(brick.model));
    brick.group.commit();

    brick.instance = cpp::Instance(brick.group);
    brick.instance.commit();

    // create the "world" model which will contain all of our geometries
    cpp::World world;
    world.setParam("instance", cpp::CopiedData(brick.instance));

    world.setParam("region", cpp::CopiedData(brick.bounds));
    world.commit();

    // create OSPRay renderer
    cpp::Renderer renderer("mpiRaycast");
    //renderer.setParam("volumeSamplingRate", 20.0f);

    // create and setup an ambient light
    cpp::Light ambientLight("ambient");
    ambientLight.commit();
    renderer.setParam("light", cpp::CopiedData(ambientLight));

    // create a GLFW OSPRay window: this object will create and manage the
    // OSPRay frame buffer and camera directly
    auto glfwOSPRayWindow =
        std::unique_ptr<GLFWDistribOSPRayWindow>(new GLFWDistribOSPRayWindow(
            vec2i{1024, 768}, worldBounds, world, renderer));

    int spp = 1;
    int currentSpp = 1;
    if (mpiRank == 0) {
      glfwOSPRayWindow->registerImGuiCallback(
          [&]() { ImGui::SliderInt("pixelSamples", &spp, 1, 64); });
    }

    glfwOSPRayWindow->registerDisplayCallback(
        [&](GLFWDistribOSPRayWindow *win) {
          // Send the UI changes out to the other ranks so we can synchronize
          // how many samples per-pixel we're taking
          MPI_Bcast(&spp, 1, MPI_INT, 0, MPI_COMM_WORLD);
          if (spp != currentSpp) {
            currentSpp = spp;
            renderer.setParam("pixelSamples", spp);
            win->addObjectToCommit(renderer.handle());
          }
        });

    // start the GLFW main loop, which will continuously render
    glfwOSPRayWindow->mainLoop();
  }
  // cleanly shut OSPRay down
  ospShutdown();

  MPI_Finalize();

  return 0;
}

bool computeDivisor(int x, int &divisor)
{
  int upperBound = std::sqrt(x);
  for (int i = 2; i <= upperBound; ++i) {
    if (x % i == 0) {
      divisor = i;
      return true;
    }
  }
  return false;
}

// Compute an X x Y x Z grid to have 'num' grid cells,
// only gives a nice grid for numbers with even factors since
// we don't search for factors of the number, we just try dividing by two
vec3i computeGrid(int num)
{
  vec3i grid(1);
  int axis = 0;
  int divisor = 0;
  while (computeDivisor(num, divisor)) {
    grid[axis] *= divisor;
    num /= divisor;
    axis = (axis + 1) % 3;
  }
  if (num != 1) {
    grid[axis] *= num;
  }
  return grid;
}

/*
VolumeBrick makeLocalVolume(const int mpiRank, const int mpiWorldSize)
{
#if 0
    const vec3i grid = computeGrid(mpiWorldSize);
    const vec3i brickId(mpiRank % grid.x,
        (mpiRank / grid.x) % grid.y,
        mpiRank / (grid.x * grid.y));
    // The bricks are 64^3 + 1 layer of ghost voxels on each axis
    const vec3i brickVolumeDims = vec3i(32);
    const vec3i brickGhostDims = vec3i(brickVolumeDims + 2);

    // The grid is over the [0, grid * brickVolumeDims] box
    worldBounds = box3f(vec3f(0.f), vec3f(grid * brickVolumeDims));
    const vec3f brickLower = brickId * brickVolumeDims;
    const vec3f brickUpper = brickId * brickVolumeDims + brickVolumeDims;

    VolumeBrick brick;
    brick.bounds = box3f(brickLower, brickUpper);
    std::cerr << brickLower << " to " << brickUpper << std::endl;
    // we just put ghost voxels on all sides here, but a real application
    // would change which faces of each brick have ghost voxels dependent
    // on the actual data
    brick.ghostBounds = box3f(brickLower - vec3f(1.f), brickUpper + vec3f(1.f));

    brick.brick = cpp::Volume("structuredRegular");

    brick.brick.setParam("dimensions", brickGhostDims);

    // we use the grid origin to place this brick in the right position inside
    // the global volume
    brick.brick.setParam("gridOrigin", brick.ghostBounds.lower);

    // generate the volume data to just be filled with this rank's id
    const size_t nVoxels = brickGhostDims.x * brickGhostDims.y *
brickGhostDims.z; std::vector<uint8_t> volumeData(nVoxels,
static_cast<uint8_t>(mpiRank)); brick.brick.setParam("data",
        cpp::CopiedData(static_cast<const uint8_t *>(volumeData.data()),
            vec3ul(brickVolumeDims)));

    brick.brick.commit();

    brick.model = cpp::VolumetricModel(brick.brick);
    cpp::TransferFunction tfn("piecewiseLinear");
    std::vector<vec3f> colors = {vec3f(0.f, 0.f, 1.f), vec3f(1.f, 0.f, 0.f)};
    std::vector<float> opacities = {0.05f, 1.f};

    tfn.setParam("color", cpp::CopiedData(colors));
    tfn.setParam("opacity", cpp::CopiedData(opacities));
    // color the bricks by their rank, we pad the range out a bit to keep
    // any brick from being completely transparent
    vec2f valueRange = vec2f(0, mpiWorldSize);
    tfn.setParam("valueRange", valueRange);
    tfn.commit();
    brick.model.setParam("transferFunction", tfn);
    brick.model.setParam("samplingRate", 0.01f);
    brick.model.commit();

    brick.group = cpp::Group();
    brick.group.setParam("volume", cpp::CopiedData(brick.model));
    brick.group.commit();

#else
  VolumeBrick brick;

  GravitySpheres gs(true, mpiRank, mpiWorldSize);
  std::vector<box3f> myregions;
  gs.getRegions(myregions);
  brick.brick = gs.getVolume();

  float x0 = (2.0*mpiRank)/mpiWorldSize - 1.0;
  float x1 = (2.0*(mpiRank+1))/mpiWorldSize - 1.0;
  //brick.bounds.push_back(box3f(vec3f(x0, -1.f, -1.f), vec3f(x1, 1.f, 1.f)));
  brick.bounds = myregions;

  worldBounds = box3f(vec3f(-1.f), vec3f(1.f));
  //std::cerr << mpiRank << " BBOUNDS = " << brick.bounds << std::endl;

  brick.model = cpp::VolumetricModel(brick.brick);
  cpp::TransferFunction tfn("piecewiseLinear");
  std::vector<vec3f> colors = {vec3f(0.f, 0.f, 1.f), vec3f(1.f, 0.f, 0.f)};
  std::vector<float> opacities = {0.0f, 0.0f, 1.0f};
  tfn.setParam("color", cpp::CopiedData(colors));
  tfn.setParam("opacity", cpp::CopiedData(opacities));
  vec2f valueRange = vec2f(0, 10.0);
  tfn.setParam("valueRange", valueRange);
  tfn.commit();
  brick.model.setParam("transferFunction", tfn);
  brick.model.setParam("samplingRate", 0.01f);
  brick.model.commit();

  brick.group = cpp::Group();
  brick.group.setParam("volume", cpp::CopiedData(brick.model));
  brick.group.commit();
#endif

  brick.instance = cpp::Instance(brick.group);
  brick.instance.commit();

  return brick;
}
*/
