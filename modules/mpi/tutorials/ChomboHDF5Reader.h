#ifndef CHOMBO_HDF5_READER_H
#define CHOMBO_HDF5_READER_H

#include <hdf5.h>
#include <iostream>
#include <string>
#include "ospray/ospray_cpp/ext/rkcommon.h"

namespace ChomboHDF5 {

// This class encapsulates the handle to a HDF5 file. It is based on the Chombo
// HDF5Handle class
class Handle
{
 public:
  // default constructor
  Handle();

  // open file
  int open(const std::string &a_filename);

  // destructor
  ~Handle();

  // check whether the file is open
  bool isOpen() const;

  // closes the file
  void close();

  // sets the current group to a specified absolute path
  void setGroup(const std::string &a_group);

  // sets the current group to a specified level
  void setGroupToLevel(int a_level);

  // returns the current group
  const std::string &getGroup() const;

  // remove the copy constructor and assignment operators
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;

  static hid_t box3i_id;
  static hid_t vec3i_id;
  static hid_t vec3d_id;

 private:
  hid_t m_fileID;
  hid_t m_currentGroupID;
  bool m_isOpen;
  std::string m_filename;
  std::string m_group;

  static bool s_initialized;
  static void initialize();
};

} // namespace ChomboHDF5
#endif
