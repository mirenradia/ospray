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

    // all ranks specify the same rendering parameters, with the exception of
    // the data to be rendered, which is distributed among the ranks
    VolumeBrick brick = makeLocalVolume(mpiRank, mpiWorldSize);

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

VolumeBrick makeLocalVolume(const int mpiRank, const int mpiWorldSize)
{
  VolumeBrick brick;

  GravitySpheres gs(true, mpiRank, mpiWorldSize);
  std::vector<box3f> myregions;
  gs.getRegions(myregions);
  brick.brick = gs.getVolume();

  bool trustinwombat = true;
  if (mpiWorldSize == 1)
      brick.bounds.push_back(box3f(vec3f(-1.f, -1.f, -1.f), vec3f(1.f, 1.f, 1.f)));
  else if (!trustinwombat) {
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