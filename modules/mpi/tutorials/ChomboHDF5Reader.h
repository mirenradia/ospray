#ifndef CHOMBO_HDF5_READER_H
#define CHOMBO_HDF5_READER_H

#include <hdf5.h>
#include <map>
#include <string>
#include "ospray/ospray_cpp/ext/rkcommon.h"

namespace ChomboHDF5 {

// Encapsulates the handle to a HDF5 file. It is based on the Chombo HDF5Handle
// class
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
  int setGroup(const std::string &a_group);

  // sets the current group to a specified level
  void setGroupToLevel(int a_level);

  // returns the current group
  const std::string &getGroup() const;

  const hid_t &fileID() const;
  const hid_t &groupID() const;

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

  // called by open to verify this is a Chombo HDF5 file
  int verifyChomboFile();
};

// Reads and stores information in the HDF5 Header
class HeaderData
{
 public:
  // header readers
  int readFromFile(Handle &a_handle);
  int readFromLocation(hid_t a_loc_id);

  void clear();

  // maps to store read attributes
  std::map<std::string, double> m_double;
  std::map<std::string, int> m_int;
  std::map<std::string, std::string> m_string;
  std::map<std::string, rkcommon::math::vec3i> m_vec3i;
  std::map<std::string, rkcommon::math::box3i> m_box3i;
  std::map<std::string, rkcommon::math::vec3d> m_vec3d;
};

// This function must be externed to "C" as it is passed to a HDF5 function
extern "C" {
herr_t ChomboHDF5HeaderDataAttributeScan(hid_t a_loc_id,
    const char *a_name,
    const H5A_info_t *a_info,
    void *a_opdata);
}

} // namespace ChomboHDF5
#endif
