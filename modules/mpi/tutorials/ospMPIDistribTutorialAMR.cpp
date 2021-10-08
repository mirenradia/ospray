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
#include "GLFWDistribOSPRayWindow.h"
#include "ospray/ospray_cpp.h"
#include "ospray/ospray_cpp/ext/rkcommon.h"
#include "ospray/ospray_util.h"
#include "GravitySpheresVolume.h"

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
VolumeBrick makeGravitySpheresVolume(const int mpiRank, const int mpiWorldSize);
VolumeBrick makeSimpleVolume1(const int mpiRank, const int mpiWorldSize);

bool trustInWombat = true;

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

  std::cout << "OSPRay rank " << mpiRank << "/" << mpiWorldSize << "\n";

  int scene = 0;
  for (int i = 0; i < argc; ++i)
  {
    if (!strcmp(argv[i], "-noWombatNoCry")) {
      std::cout << "No Wombat!" << std::endl;
      trustInWombat = false;
    }
    if (!strcmp(argv[i], "-GS")) {
      scene = 0;
    }
    if (!strcmp(argv[i], "-Simple1")) {
      scene = 1;
    }
  }

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

    // all ranks specify the same rendering parameters, with the exception of
    // the data to be rendered, which is distributed among the ranks
    VolumeBrick brick;
    switch (scene) {
    case 0:
    default:
        brick = makeGravitySpheresVolume(mpiRank, mpiWorldSize);
        break;
    case 1:
        brick = makeSimpleVolume1(mpiRank, mpiWorldSize);
    }

    // create the "world" model which will contain all of our geometries
    cpp::World world;
    world.setParam("instance", cpp::CopiedData(brick.instance));

    world.setParam("region", cpp::CopiedData(brick.bounds));
    world.commit();

    // create OSPRay renderer
    cpp::Renderer renderer("mpiRaycast");

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

VolumeBrick makeGravitySpheresVolume(const int mpiRank, const int mpiWorldSize)
{
  if (!mpiRank) std::cerr << "MAKING GRAVITY SPHERES VOLUME #1" << std::endl;
  VolumeBrick brick;

  GravitySpheres gs(true, mpiRank, mpiWorldSize);
  std::vector<box3f> myregions;
  gs.getRegions(myregions);
  brick.brick = gs.getVolume();

  if (mpiWorldSize == 1)
      brick.bounds.push_back(box3f(vec3f(-1.f, -1.f, -1.f), vec3f(1.f, 1.f, 1.f)));
  else if (!trustInWombat) {
      float x0 = (2.0*mpiRank)/mpiWorldSize - 1.0;
      float x1 = (2.0*(mpiRank+1))/mpiWorldSize - 1.0;
      brick.bounds.push_back(box3f(vec3f(x0, -1.f, -1.f), vec3f(x1, 1.f, 1.f)));
  }  else
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

  brick.instance = cpp::Instance(brick.group);
  brick.instance.commit();

  return brick;
}

VolumeBrick makeSimpleVolume1(const int mpiRank, const int mpiWorldSize)
{
  if (!mpiRank) std::cerr << "MAKING SIMPLE VOLUME #1" << std::endl;

  cpp::Volume volume("amr");

  std::vector<std::vector<float>> perBlockData;
  std::vector<box3i> perBlockBounds;
  std::vector<box3f> localPerBlockBounds;
  std::vector<int> perBlockLevels;
  std::vector<int> perBlockOwners;
  std::vector<float> perLevelCellWidths;

  int blockDims = 16;
  int numlevels = 1;
  for (int level = 0; level < numlevels; level++) {
    for (int r = 0; r < mpiWorldSize; r++) {
      std::cerr << r << std::endl;
      float v = r;
      std::vector<float> data(blockDims*blockDims*blockDims, v);
      perBlockData.push_back(data);
      std::cerr << "data " << data.size() << std::endl;

      box3i box;
      box.lower = vec3i(r*blockDims,0,0);
      box.upper = vec3i((r+1)*blockDims-1,blockDims-1,blockDims-1);
      std::cerr << "box " << box << std::endl;
      perBlockBounds.push_back(box);

      perBlockLevels.push_back(level);

      perBlockOwners.push_back(r);
    }
    float w = 1.f / powf(2, level);
    std::cerr << "L " << level << " w " << w << std::endl;
    perLevelCellWidths.push_back(w);
  }

  std::vector<cpp::CopiedData> allBlockData;
  for (const std::vector<float> &bd : perBlockData)
    allBlockData.emplace_back(bd.data(), OSP_FLOAT, bd.size());

  worldBounds = box3f(vec3f(0.f), vec3f(mpiWorldSize*blockDims, blockDims, blockDims));
  volume.setParam("gridOrigin", vec3f(0.f));
  volume.setParam("gridSpacing", vec3f(1.f));
  volume.setParam("block.data", cpp::CopiedData(allBlockData));
  volume.setParam("block.bounds", cpp::CopiedData(perBlockBounds));
  volume.setParam("block.level", cpp::CopiedData(perBlockLevels));
  volume.setParam("cellWidth", cpp::CopiedData(perLevelCellWidths));
  volume.commit();

  cpp::VolumetricModel model = cpp::VolumetricModel(volume);
  cpp::TransferFunction tfn("piecewiseLinear");
  std::vector<vec3f> colors = {vec3f(0.f, 0.f, 1.f), vec3f(1.f, 0.f, 0.f)};
  std::vector<float> opacities = {0.25f, 1.0f};
  tfn.setParam("color", cpp::CopiedData(colors));
  tfn.setParam("opacity", cpp::CopiedData(opacities));
  vec2f valueRange = vec2f(0, mpiWorldSize);
  tfn.setParam("valueRange", valueRange);
  tfn.commit();
  model.setParam("transferFunction", tfn);
  model.setParam("samplingRate", 0.01f);
  model.commit();

  cpp::Group group = cpp::Group();
  group.setParam("volume", cpp::CopiedData(model));
  group.commit();

  cpp::Instance instance = cpp::Instance(group);
  instance.commit();

  VolumeBrick brick;
  brick.brick = volume;
  brick.model = model;
  brick.group = group;
  brick.instance = instance;
  localPerBlockBounds.push_back(worldBounds); //todo - correct for wombat
  brick.bounds = localPerBlockBounds;
  brick.ghostBounds = worldBounds;

  return brick;
}
