// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rkcommon/tasking/parallel_for.h"
#include "rkcommon/utility/random.h"
// stl
#include <random>
#include <vector>
// raw_to_amr
#include "rawToAMR.h"

#include "GravitySpheresVolume.h"

using namespace ospray;
using namespace rkcommon;
using namespace rkcommon::math;

// Inlined definitions ////////////////////////////////////////////////////

GravitySpheres::GravitySpheres(bool asAMR)
    : createAsAMR(asAMR)
{
  auto voxels = generateVoxels();
  auto voxelRange = vec2f(0.f, 10.f); //10 is from default numPoints of 10

  volume = createAsAMR ? createAMRVolume(voxels) : createStructuredVolume(voxels);
}

std::vector<float> GravitySpheres::generateVoxels() const
{
  struct Point
  {
    vec3f center;
    float weight;
  };

  // create random number distributions for point center and weight
  std::mt19937 gen(0);

  utility::uniform_real_distribution<float> centerDistribution(-1.f, 1.f);
  utility::uniform_real_distribution<float> weightDistribution(0.1f, 0.3f);

  // populate the points
  std::vector<Point> points(numPoints);

  for (auto &p : points) {
    p.center.x = centerDistribution(gen);
    p.center.y = centerDistribution(gen);
    p.center.z = centerDistribution(gen);

    p.weight = weightDistribution(gen);
  }

  // get world coordinate in [-1.f, 1.f] from logical coordinates in [0,
  // volumeDimension)
  auto logicalToWorldCoordinates = [&](int i, int j, int k) {
    return vec3f(-1.f + float(i) / float(volumeDimensions.x - 1) * 2.f,
        -1.f + float(j) / float(volumeDimensions.y - 1) * 2.f,
        -1.f + float(k) / float(volumeDimensions.z - 1) * 2.f);
  };

  // generate voxels
  std::vector<float> voxels(volumeDimensions.long_product());

  tasking::parallel_for(volumeDimensions.z, [&](int k) {
    for (int j = 0; j < volumeDimensions.y; j++) {
      for (int i = 0; i < volumeDimensions.x; i++) {
        // index in array
        size_t index = size_t(k) * volumeDimensions.z * volumeDimensions.y
            + size_t(j) * volumeDimensions.x + size_t(i);

        // compute volume value
        float value = 0.f;

        for (auto &p : points) {
          vec3f pointCoordinate = logicalToWorldCoordinates(i, j, k);
          const float distance = length(pointCoordinate - p.center);

          // contribution proportional to weighted inverse-square distance
          // (i.e. gravity)
          value += p.weight / (distance * distance);
        }

        voxels[index] = value;
      }
    }
  });

  return voxels;
}

cpp::Volume GravitySpheres::createStructuredVolume(
    const VoxelArray &voxels) const
{
  cpp::Volume volume("structuredRegular");

  volume.setParam("gridOrigin", vec3f(-1.f));
  volume.setParam("gridSpacing", vec3f(2.f / reduce_max(volumeDimensions)));
  volume.setParam("data", cpp::CopiedData(voxels.data(), volumeDimensions));
  volume.commit();
  return volume;
}

cpp::Volume GravitySpheres::createAMRVolume(const VoxelArray &voxels) const
{
  const int numLevels = 2;
  const int blockSize = 16;
  const int refinementLevel = 4;
  const float threshold = 1.0;

  std::vector<box3i> blockBounds;
  std::vector<int> refinementLevels;
  std::vector<float> cellWidths;
  std::vector<std::vector<float>> blockDataVectors;
  std::vector<cpp::CopiedData> blockData;

  // convert the structured volume to AMR
  makeAMR(voxels,
      volumeDimensions,
      numLevels,
      blockSize,
      refinementLevel,
      threshold,
      blockBounds,
      refinementLevels,
      cellWidths,
      blockDataVectors);

  for (const std::vector<float> &bd : blockDataVectors)
    blockData.emplace_back(bd.data(), OSP_FLOAT, bd.size());

  // create an AMR volume and assign attributes
  cpp::Volume volume("amr");

  int toplevelVolDim =
      reduce_max(volumeDimensions) / std::pow(refinementLevel, numLevels - 1);
  volume.setParam("gridOrigin", vec3f(-1.f));
  volume.setParam("gridSpacing", vec3f(2.f / toplevelVolDim));
  volume.setParam("block.data", cpp::CopiedData(blockData));
  volume.setParam("block.bounds", cpp::CopiedData(blockBounds));
  volume.setParam("block.level", cpp::CopiedData(refinementLevels));
  volume.setParam("cellWidth", cpp::CopiedData(cellWidths));

  volume.commit();

  return volume;
}
