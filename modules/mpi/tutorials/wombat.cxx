#include "wombat.h"

#include <iostream>

namespace wombat {

std::vector<Level> levels;

typedef std::pair<int, Box> BoxWithID;

bool subdivide(Box box, std::vector<Box> children, int axis, std::vector<Box> &convexBoxes) {

  if (children.size() == 0) {
    //no children therefore convex
    convexBoxes.push_back(box); 
    return true;
  }

  Box lhalf = box;
  lhalf.origin[axis] = box.origin[axis];
  lhalf.dims[axis] = box.dims[axis] / 2;
  Box rhalf = box;
  rhalf.origin[axis] = box.origin[axis]+(box.dims[axis] / 2);
  rhalf.dims[axis] = box.dims[axis] - (box.dims[axis] / 2); //integer division, so rounding up

  int refinement = levels[box.level].refinement[axis];
  int leftedge = box.origin[axis]*refinement;
  int rightedge = leftedge + box.dims[axis]*refinement;
  int middle = (leftedge + rightedge) / 2;
  
  std::vector<Box> leftChildren;
  std::vector<Box> rightChildren;
  //decide if children are children of left half, right half or both
  for (int c = 0; c < static_cast<int>(children.size()); ++c) {
     Box child = children[c];
     int cleftedge = child.origin[axis];
     int crightedge = cleftedge + child.dims[axis];
     if (crightedge <= middle) {
        leftChildren.push_back(child);
     } else if (cleftedge >= middle) {
        rightChildren.push_back(child);
     } else {
       leftChildren.push_back(child);
       rightChildren.push_back(child);
     }
  }
  //std::cerr << leftChildren.size() << " left " << std::endl;
  //std::cerr << rightChildren.size() << " right " << std::endl;
  subdivide(lhalf, leftChildren, (axis+1)%3, convexBoxes);
  subdivide(rhalf, rightChildren, (axis+1)%3, convexBoxes);
  return true;
}

bool convexify(const std::vector<Level> _levels, const std::vector<Box> inBoxes, std::vector<Box> &convexBoxes) {
  levels = _levels;

  for (int l = 0; l < static_cast<int>(levels.size()); ++l) {
    //std::cerr << "Working on Level " << l << std::endl;
    std::vector<BoxWithID> boxesAtLevel;
    for (int b = 0; b < static_cast<int>(inBoxes.size()); ++b) {
      if (inBoxes[b].level == l) {
        boxesAtLevel.push_back(BoxWithID(b, inBoxes[b]));
      }
    }
    //std::cerr << boxesAtLevel.size() << " boxes at level " << l << std::endl;
    for (int b = 0; b < static_cast<int>(boxesAtLevel.size()); ++b) {
       //std::cerr << "box " << b << std::endl;
       std::vector<Box> children;
       for (int c = 0; c < static_cast<int>(inBoxes.size()); ++c) {
         if (inBoxes[c].parent == boxesAtLevel[b].first) {
           children.push_back(inBoxes[c]);
         }
       }
      //std::cerr << "box " << b << " has " << children.size() << " children" << std::endl;
      subdivide(boxesAtLevel[b].second, children, 0, convexBoxes);
    }
  }
  return true;
}

}
