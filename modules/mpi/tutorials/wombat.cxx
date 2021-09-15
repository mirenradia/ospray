#include "wombat.h"

#include <iostream>

#include "rkcommon/math/vec.h"

using namespace rkcommon::math;

namespace wombat {

std::vector<Level> levels;

typedef std::pair<int, Box> BoxWithID;

bool subdivide(Box box, std::vector<Box> children, int axis, std::vector<Box> &convexBoxes) {

  if (children.size() == 0) {
    //no children therefore convex
    //std::cerr << "no children" << std::endl;
    convexBoxes.push_back(box); 
    return true;
  }

  //std::cerr << "ORG " << box.origin[0] << "," << box.origin[1] << "," << box.origin[2] << ", DIMS " << box.dims[0] << "," << box.dims[1] << "," << box.dims[2] << std::endl;
  if (box.dims[0] == 0 || box.dims[1] == 0 || box.dims[2] == 0) {
    //std::cerr << "huh?" << std::endl; //May be a bug.
    return false;
  }
  if (box.dims[0] <= 1 && box.dims[1] <= 1 && box.dims[2] <= 1) {
    //convexBoxes.push_back(box);
    //std::cerr << "hey?" << std::endl; //optimization, ignore tiny boxes that are fully covered. Todo - not just tiny.
    return true;
  }

  Box lhalf = box;
  lhalf.origin[axis] = box.origin[axis];
  lhalf.dims[axis] = box.dims[axis] / 2;
  Box rhalf = box;
  rhalf.origin[axis] = box.origin[axis]+(box.dims[axis] / 2);
  rhalf.dims[axis] = box.dims[axis] - (box.dims[axis] / 2); //integer division, so rounding up

  int refinement = levels[box.level].refinement[axis]; //TODO fix this, assuming chombo 2 here
  int leftedge = box.origin[axis]*refinement;
  int rightedge = leftedge + box.dims[axis]*refinement;
  int middle = (leftedge + rightedge) / 2;
  //std::cerr << "middle " << middle << std::endl;

  std::vector<Box> leftChildren;
  std::vector<Box> rightChildren;
  //decide if children are children of left half, right half or both
  for (int c = 0; c < static_cast<int>(children.size()); ++c) {
     Box child = children[c];
     int cleftedge = child.origin[axis];
     int crightedge = cleftedge + child.dims[axis];
     if (crightedge <= middle) {
       //std::cerr << "CL " << cleftedge << " CR " << crightedge << "-> L" << std::endl;
       leftChildren.push_back(child);
     } else if (cleftedge >= middle) {
       //std::cerr << "CL " << cleftedge << " CR " << crightedge << "-> R" << std::endl;
       rightChildren.push_back(child);
     } else {
       //std::cerr << "CL " << cleftedge << " CR " << crightedge << "-> B" << std::endl;
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
    std::cerr << "Working on Level " << l << std::endl;
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
      //std::cerr << "boxcar " << b << " has " << children.size() << " children" << std::endl;
      subdivide(boxesAtLevel[b].second, children, 0, convexBoxes);
    }
  }
  return true;
}

void geneology(const std::vector<Level> _levels, std::vector<Box> &inBoxes, int rank) {
  //std::cerr << rank << " START" << std::endl;
  std::vector<std::vector<BoxWithID>> sorted;
  for (int l = 0; l < static_cast<int>(_levels.size()); ++l) {
    std::vector<BoxWithID> boxesAtLevel;
    for (int b = 0; b < static_cast<int>(inBoxes.size()); ++b) {
      if (inBoxes[b].level == l) {
        boxesAtLevel.push_back(BoxWithID(b,inBoxes[b]));
      }
    }
    //std::cerr << rank << " level " << l << " " << boxesAtLevel.size() << std::endl;
    sorted.push_back(boxesAtLevel);
  }
  //std::cerr << rank << " sorted " << sorted.size() << std::endl;
  inBoxes.clear();
  for (int l = 0; l < static_cast<int>(_levels.size())-1; ++l) {
    vec3i vref{1,1,1};
    //std::cerr << "level " << l << std::endl;
    std::vector<BoxWithID> &currentLevel = sorted[l];
    std::vector<BoxWithID> &nextLevel = sorted[l+1];
    vref.x *= _levels[l].refinement[0];
    vref.y *= _levels[l].refinement[1];
    vref.z *= _levels[l].refinement[2];
    //std::cerr << "VR IS " << vref << std::endl;
    for (int B = 0; B < static_cast<int>(currentLevel.size()); ++B) {
      Box p = currentLevel[B].second;
      vec3i px0 = vec3i{p.origin[0],p.origin[1],p.origin[2]}*vref;
      vec3i px1 = px0 + vec3i{p.dims[0]+1,p.dims[1]+1,p.dims[2]+1}*vref;
      //std::cerr << B << "," << currentLevel[B].first << " " << p.parent << " " << px0 << "-" << px1 << std::endl;

      for (int b = 0; b < static_cast<int>(nextLevel.size()); ++b) {
        Box c = nextLevel[b].second;
        vec3i bx0 = vec3i{c.origin[0],c.origin[1],c.origin[2]};
        vec3i bx1 = bx0+vec3i{c.dims[0]+1,c.dims[1]+1,c.dims[2]+1};
        if ( bx0.x >= px0.x && bx1.x <= px1.x &&
             bx0.y >= px0.y && bx1.y <= px1.y &&
             bx0.z >= px0.z && bx1.z <= px1.z) {
          //std::cerr << " " << b << "," << nextLevel[b].first << " " << currentLevel[B].first << " " <<  bx0 << "-" << bx1 << std::endl;
          nextLevel[b].second.parent = currentLevel[B].first;
        }
      }
      inBoxes.push_back(currentLevel[B].second);
    }
  }
  std::vector<BoxWithID> &leafLevel = sorted[_levels.size()-1];
  //std::cerr << rank << " push leaves" << leafLevel.size() << std::endl;
  for (int b = 0; b < static_cast<int>(leafLevel.size()); ++b) {
    //std::cerr << rank << " " << b << std::endl;
    inBoxes.push_back(leafLevel[b].second);
  }

  //std::cerr << " ownership" << std::endl;
  for (int B = 0; B < static_cast<int>(inBoxes.size()); ++B) {
     Box p = inBoxes[B];
     //std::cerr << B << " " << p.level << /*" " << p.owningrank <<*/ " " << p.parent << " " << p.origin[0] << "," << p.origin[1] << "," << p.origin[2] << " " << p.dims[0] << "," << p.dims[1] << "," << p.dims[2] << std::endl;
  }
  //std::cerr << rank << " DONE" << std::endl;

}

}
