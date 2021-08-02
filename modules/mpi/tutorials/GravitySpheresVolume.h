#pragma once

#include "ospray/ospray_cpp.h"
#include "ospray/ospray_cpp/ext/rkcommon.h"

#include <vector>

using namespace rkcommon;
using namespace rkcommon::math;

using VoxelArray = std::vector<float>;

struct GravitySpheres
{
  GravitySpheres(bool asAMR = false, int rank = 0, int numranks = 1);
  ~GravitySpheres() = default;

  ospray::cpp::Volume getVolume() { return volume; };
  void getRegions(std::vector<box3f> &regions);

 private:
  VoxelArray generateVoxels() const;

  ospray::cpp::Volume createStructuredVolume(const VoxelArray &voxels) const;
  ospray::cpp::Volume createAMRVolume(const VoxelArray &voxels);

  // Data //

  int rank;
  int numranks;
  vec3i volumeDimensions{128};
  int numPoints{10};
  bool createAsAMR{false};
  ospray::cpp::Volume volume;
  std::vector<box3f> myregions;
};
