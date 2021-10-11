// Copyright 2021 Intel Co.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>

namespace wombat {

typedef struct Level {
  double origin[3]; //NU? //worldspace lower left x,y,z coordinate
  double spacing[3]; //NU? //wordspace x,y,z size of one cell, this is required to be a multiple of its child level
  int refinement[3]; //how much finer children's spacing is, ex 2 if my spacing is 1.0,1.0,1.0 and childrens' spacing is .5,.5,.5
  int dims[3]; //NU? //global number of cells at this level
} Level;

typedef struct Box {
  int owningrank; //NU //the mpi rank of the owner of this box
  int level; //the level that this box belongs to
  int parent = -1; //index of parent at previous level within inBoxes array.
  int origin[3]; //index into level of lower left corner
  int dims[3]; //index space i,j,k number of cells
} Box;

//Takes the AMR mesh consisting of levels and inBoxes and populates
//convexBoxes, which is an equivaltent mesh consisting of only convex boxes
//with course cells removed from each level. Returns true on success and
//false on failure. Overwrites convexBoxes, does not modify levels or inBoxes.
bool convexify(const std::vector<Level> levels, const std::vector<Box> inBoxes, std::vector<Box> &convexBoxes);

//Discover parent child relationships between boxes in adjacent levels.
//The inBoxes array is sorted amongst levels, ids are taken as index within
//each level. Boxes in each refined level are flagged as children of boxes
//within parent level if they fit within it.
void geneology(const std::vector<Level> levels, std::vector<Box> &inBoxes, int rank);

//temporary, just a useful place for debugging in both examples
enum remoteValueMode { DATA, RANK, ANAN, ANUMBER };
void getRemoteValue(remoteValueMode &, float &);
}

