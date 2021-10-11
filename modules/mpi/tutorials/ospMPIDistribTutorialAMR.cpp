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
#include "wombat.h"

int rankToDebugf = -1;

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
VolumeBrick makeFakeMPIVolume(const int fakeWorldSize);

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
  int fakeWorldSize = 1;
  for (int i = 0; i < argc; ++i)
  {
    if (!strcmp(argv[i], "-GS")) {
      scene = 0;
    }
    if (!strcmp(argv[i], "-Simple1")) {
      scene = 1;
    }
    if (!strcmp(argv[i], "-Fake")) {
      scene = 2;
      if (i+1 < argc)
      {
        fakeWorldSize = std::stoi(std::string(argv[i+1]));
      }
    }

  }

  const char *noWombatEnv = getenv("NOWOMBAT");
  if (noWombatEnv) trustInWombat = false;
  const char *rankToDebugfEnv = getenv("RANKTODEBUGF");
  if (rankToDebugfEnv) rankToDebugf = std::stoi(std::string(rankToDebugfEnv));

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
        break;
    case 2:
        brick = makeFakeMPIVolume(fakeWorldSize);
        break;
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
  if (mpiRank==rankToDebugf) std::cerr << "MAKING GRAVITY SPHERES VOLUME #1" << std::endl;
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
  if (mpiRank==rankToDebugf) std::cerr << "MAKING SIMPLE VOLUME #1" << std::endl;

  cpp::Volume volume("amr");

  std::vector<std::vector<float>> perBlockData;
  std::vector<box3i> perBlockExtents;
  std::vector<box3f> perBlockBounds;
  std::vector<int> perBlockLevels;
  std::vector<int> perBlockOwners;
  std::vector<float> perLevelCellWidths;

  worldBounds = box3f(vec3f(0.f), vec3f(0.f));

  int baseRes = 2;
  const char *baseResEnv = getenv("BASERES");
  if (baseResEnv) baseRes = std::stoi(std::string(baseResEnv));

  int numLevels = 2;
  const char *numLevelsEnv = getenv("NUMLEVELS");
  if (numLevelsEnv) numLevels = std::stoi(std::string(numLevelsEnv));

  wombat::remoteValueMode rvm = wombat::DATA;
  float remoteValue = 0.0;
  wombat::getRemoteValue(rvm, remoteValue);

  int blockDims = baseRes*mpiWorldSize;
  for (int level = 0; level < numLevels; level++) {
    int numInnerBlocks = powf(mpiWorldSize, level+1);
    float w = 1.f / powf(mpiWorldSize, level);
    if (mpiRank==rankToDebugf) std::cerr << "L " << level << " w " << w << std::endl;
    perLevelCellWidths.push_back(w);

    for (int r = 0; r < numInnerBlocks; r++) {
      if (mpiRank==rankToDebugf) std::cerr << "blocknum " << r << std::endl;
      int owner = r%mpiWorldSize;

      if (mpiRank==rankToDebugf) std::cerr << "owner " << owner << std::endl;
      float v = owner+1;
      if (owner != mpiRank) {
        switch (rvm) {
        case wombat::DATA:
            v = owner+1;
            break;
        case wombat::RANK:
            v = mpiRank + 1;
            break;
        case wombat::ANAN:
            v = NAN;
            break;
        case wombat::ANUMBER:
            v = remoteValue;
            break;
        default:
            break;
        }
      }
      if (mpiRank==rankToDebugf) std::cerr << "value " << v << std::endl;
      std::vector<float> data(blockDims*blockDims*blockDims, v);
      if (mpiRank==rankToDebugf) std::cerr << "data " << data.size() << " " << v << "'s" << std::endl;

      box3i box;
      box.lower = vec3i(r*blockDims,0,0);
      box.upper = vec3i((r+1)*blockDims-1,blockDims-1,blockDims-1);
      if (mpiRank==rankToDebugf) std::cerr << "box " << box << std::endl;

      box3f boxf;
      boxf.lower = box.lower*w;
      boxf.upper = (box.upper+vec3i(1))*w;
      if (mpiRank==rankToDebugf) std::cerr << "boxf " << boxf << std::endl;
      worldBounds.extend(boxf);

      //if (owner != mpiRank)
      //  continue;

      perBlockLevels.push_back(level);
      perBlockOwners.push_back(owner);
      perBlockData.push_back(data);
      perBlockExtents.push_back(box);
      perBlockBounds.push_back(boxf);
    }
  }

  std::vector<cpp::CopiedData> allBlockData;
  for (const std::vector<float> &bd : perBlockData)
    allBlockData.emplace_back(bd.data(), OSP_FLOAT, bd.size());

  volume.setParam("gridOrigin", vec3f(0.f));
  volume.setParam("gridSpacing", vec3f(1.f));
  volume.setParam("block.data", cpp::CopiedData(allBlockData));
  volume.setParam("block.bounds", cpp::CopiedData(perBlockExtents));
  volume.setParam("block.level", cpp::CopiedData(perBlockLevels));
  volume.setParam("cellWidth", cpp::CopiedData(perLevelCellWidths));
  volume.setParam("background", 0.0);
  volume.setParam("method", OSP_AMR_CURRENT);
  volume.commit();

  if (trustInWombat) {
    std::vector<box3f> wRegions;

    std::vector<wombat::Level> wLevels;
    std::vector<vec3i> wLevelDims;
    vec3i l0dims(blockDims*mpiWorldSize,blockDims,blockDims); //top level dims, globally
    int prevRefRatio = 1;
    for (auto n : perLevelCellWidths)
    {
      wombat::Level l;
      l.refinement[0] = mpiWorldSize;
      l.refinement[1] = mpiWorldSize;
      l.refinement[2] = mpiWorldSize;
      wLevels.push_back(l);
      wLevelDims.push_back(l0dims * prevRefRatio);
      prevRefRatio = prevRefRatio * mpiWorldSize;
    }
    std::vector<wombat::Box> wBoxes;
    const std::vector<int> &wBlockOwners = perBlockOwners;
    const std::vector<int> &wBlockLevels = perBlockLevels;
    int index = 0;
    for (auto n : perBlockExtents)
    {
      wombat::Box b;
      b.owningrank = wBlockOwners[index];
      b.level = wBlockLevels[index];
      b.origin[0] = n.lower.x;
      b.origin[1] = n.lower.y;
      b.origin[2] = n.lower.z;
      vec3f bo(b.origin[0],b.origin[1],b.origin[2]);
      b.dims[0] = n.upper.x-n.lower.x+1;
      b.dims[1] = n.upper.y-n.lower.y+1;
      b.dims[2] = n.upper.z-n.lower.z+1;
      vec3f bd(b.dims[0],b.dims[1],b.dims[2]);
      vec3f bx0 = (vec3f(bo)/(vec3f(wLevelDims[b.level]))) * worldBounds.upper;
      vec3f bx1 = (vec3f(bo+bd)/(vec3f(wLevelDims[b.level]))) * worldBounds.upper;
      if (mpiRank==rankToDebugf) std::cerr << "IN:  " << index << " " << b.owningrank << " " << b.level << " "
                << bo << "+" << bd << "|"
                << bx0 << ".." << bx1 << std::endl;
      wBoxes.push_back(b);
      index++;
    }

    std::vector<wombat::Box> wsBoxes;
    //run wombat to derive a set of convex regions
    wombat::geneology(wLevels, wBoxes, mpiRank);
    wombat::convexify(wLevels, wBoxes, wsBoxes);

    //from that run, tell compositor the regions that this rank owns in worldspace
    index = 0;
    for (; index < wsBoxes.size(); ++index)
    {
      wombat::Box b = wsBoxes[index];
      vec3i bo = b.origin;
      vec3i bd = b.dims;
      vec3f bx0 = ((vec3f(bo)+vec3f(0.f))/(vec3f(wLevelDims[b.level]))) * worldBounds.upper;
      vec3f bx1 = ((vec3f(bo+bd)-vec3f(0.f))/(vec3f(wLevelDims[b.level]))) * worldBounds.upper;
      if (mpiRank==rankToDebugf) std::cerr << "OUT: " << index << " " << (b.owningrank==mpiRank?"*":"") << b.owningrank << " " << b.level << " "
                                           << bo << "+" << bd << "|"
                                           << bx0 << ".." << bx1 << std::endl;
      if (b.owningrank == mpiRank)
      {
        wRegions.push_back(box3f(vec3f(bx0.x,bx0.y,bx0.z),vec3f(bx1.x,bx1.y,bx1.z)));
      }
    }
    perBlockBounds = wRegions;
  }

  cpp::VolumetricModel model = cpp::VolumetricModel(volume);
  cpp::TransferFunction tfn("piecewiseLinear");
  std::vector<vec3f> colors;
  colors.push_back(vec3f(0.f, 1.f, 0.f));
  std::vector<float> opacities;
  opacities.push_back(1.0);
  int maxcolor = mpiWorldSize+1;
  for (int i = 0; i < maxcolor; i++) {
    colors.push_back(vec3f((float)i/maxcolor,0.0,1.0-(float)i/maxcolor));
    opacities.push_back(0.80);
  }
  tfn.setParam("color", cpp::CopiedData(colors));
  tfn.setParam("opacity", cpp::CopiedData(opacities));
  vec2f valueRange = vec2f(0, maxcolor);
  tfn.setParam("valueRange", valueRange);
  tfn.commit();
  model.setParam("transferFunction", tfn);
  model.setParam("samplingRate", 20.5f);
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
  brick.bounds = perBlockBounds;
  brick.ghostBounds = worldBounds; //unused so far

  return brick;
}

VolumeBrick makeFakeMPIVolume(const int mpiWorldSize)
{
  if (rankToDebugf>-1) std::cerr << "MAKING SIMPLE VOLUME #1" << std::endl;

  cpp::Volume volume("amr");

  std::vector<std::vector<float>> perBlockData;
  std::vector<box3i> perBlockExtents;
  std::vector<box3f> perBlockBounds;
  std::vector<int> perBlockLevels;
  std::vector<int> perBlockOwners;
  std::vector<float> perLevelCellWidths;

  worldBounds = box3f(vec3f(0.f), vec3f(0.f));

  int baseRes = 2;
  const char *baseResEnv = getenv("BASERES");
  if (baseResEnv) baseRes = std::stoi(std::string(baseResEnv));

  int numLevels = 2;
  const char *numLevelsEnv = getenv("NUMLEVELS");
  if (numLevelsEnv) numLevels = std::stoi(std::string(numLevelsEnv));

  int blockDims = baseRes*mpiWorldSize;
  for (int level = 0; level < numLevels; level++) {
    int numInnerBlocks = powf(mpiWorldSize, level+1);
    float w = 1.f / powf(mpiWorldSize, level);
    if (rankToDebugf>-1) std::cerr << "L " << level << " w " << w << std::endl;
    perLevelCellWidths.push_back(w);

    for (int r = 0; r < numInnerBlocks; r++) {
      if (rankToDebugf>-1) std::cerr << "blocknum " << r << std::endl;
      int owner = r%mpiWorldSize;

      if (rankToDebugf>-1) std::cerr << "owner " << owner << std::endl;
      float v = owner+1;
      if (rankToDebugf>-1) std::cerr << "value " << v << std::endl;
      std::vector<float> data(blockDims*blockDims*blockDims, v);
      if (rankToDebugf>-1) std::cerr << "data " << data.size() << " " << v << "'s" << std::endl;

      box3i box;
      box.lower = vec3i(r*blockDims,0,0);
      box.upper = vec3i((r+1)*blockDims-1,blockDims-1,blockDims-1);
      if (rankToDebugf>-1) std::cerr << "box " << box << std::endl;

      box3f boxf;
      boxf.lower = box.lower*w;
      boxf.upper = (box.upper+vec3i(1))*w;
      if (rankToDebugf>-1) std::cerr << "boxf " << boxf << std::endl;
      worldBounds.extend(boxf);

      perBlockLevels.push_back(level);
      perBlockOwners.push_back(owner);
      perBlockData.push_back(data);
      perBlockExtents.push_back(box);
      perBlockBounds.push_back(boxf);
    }
  }

  std::vector<cpp::CopiedData> allBlockData;
  for (const std::vector<float> &bd : perBlockData)
    allBlockData.emplace_back(bd.data(), OSP_FLOAT, bd.size());

  volume.setParam("gridOrigin", vec3f(0.f));
  volume.setParam("gridSpacing", vec3f(1.f));
  volume.setParam("block.data", cpp::CopiedData(allBlockData));
  volume.setParam("block.bounds", cpp::CopiedData(perBlockExtents));
  volume.setParam("block.level", cpp::CopiedData(perBlockLevels));
  volume.setParam("cellWidth", cpp::CopiedData(perLevelCellWidths));
  volume.commit();

  cpp::VolumetricModel model = cpp::VolumetricModel(volume);
  cpp::TransferFunction tfn("piecewiseLinear");

  std::vector<vec3f> colors;
  colors.push_back(vec3f(0.f, 1.f, 0.f));
  std::vector<float> opacities;
  opacities.push_back(1.0);
  int maxcolor = mpiWorldSize+1;
  for (int i = 0; i < maxcolor; i++) {
    colors.push_back(vec3f((float)i/maxcolor,0.0,1.0-(float)i/maxcolor));
    opacities.push_back(0.95);
  }
  tfn.setParam("color", cpp::CopiedData(colors));
  tfn.setParam("opacity", cpp::CopiedData(opacities));
  vec2f valueRange = vec2f(0, maxcolor);
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
  brick.bounds = perBlockBounds;
  brick.ghostBounds = worldBounds; //unused so far

  return brick;
}
