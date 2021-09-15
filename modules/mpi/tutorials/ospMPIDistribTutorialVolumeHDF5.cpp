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

    usleep(500000*mpiRank); //quick hack to segregate rank debugfs
    std::cerr << "-----------------------------------" << std::endl;
    std::cerr << "RANK " << mpiRank << std::endl;

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
        b.dims[0] = n.upper.x-n.lower.x;
        b.dims[1] = n.upper.y-n.lower.y;
        b.dims[2] = n.upper.z-n.lower.z;
        std::cerr << "IN:  " << index << " " << b.owningrank << " " << b.level << " "
                  << "(" << b.origin[0] << "," << b.origin[1] << "," << b.origin[2] << ")..("
                  << b.dims[0] << "," << b.dims[1] << "," << b.dims[2] << ")" << std::endl;
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
      index = 0;
      for (; index < sboxes.size(); ++index)
      {
        wombat::Box b = sboxes[index];
        if (b.owningrank == mpiRank)
        {
          vec3i bo = b.origin;
          vec3i bd = b.dims;
          vec3f bx0 = ((vec3f(bo)-vec3f(0.5))/(vec3f(levelDims[b.level]))) * worldBounds.upper;
          vec3f bx1 = ((vec3f(bo+bd)+vec3f(0.5))/(vec3f(levelDims[b.level]))) * worldBounds.upper;
          wombatRegions.push_back(box3f(vec3f(bx0.x,bx0.y,bx0.z),vec3f(bx1.x,bx1.y,bx1.z)));
          std::cerr << "OUT: " << index << " " << b.owningrank << " " << b.level << " "
                    << bo << ".." << bd << "->"
                    << bx0 << "," << bx1 << std::endl;
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
    std::vector<vec3f> colors = {vec3f(0.f), vec3f(0.f, 0.f, 1.f), vec3f(0.f), vec3f(1.f, 0.f, 0.f), vec3f(0.f)};
    //std::vector<float> opacities = {0.0f, 0.0f, 0.2f, 0.4f, 0.6f, 0.8f};
    std::vector<float> opacities = {0.0f, 0.0f, 0.5f, 0.0f, 0.00f, 0.0f};
    tfn.setParam("color", cpp::CopiedData(colors));
    tfn.setParam("opacity", cpp::CopiedData(opacities));
    vec2f valueRange = vec2f(0.2, 0.96435); //binary
    //vec2f valueRange = vec2f(0.0, 4000); //amelia
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
