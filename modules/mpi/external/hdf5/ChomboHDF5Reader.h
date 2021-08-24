#ifndef CHOMBO_HDF5_READER_H
#define CHOMBO_HDF5_READER_H

#include <hdf5.h>
#include <map>
#include <string>
#include <vector>
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

  const std::string &getFilename() const;

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

// Reads and stores information in HDF5 headers
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

// Main class to read data from a HDF5 file into a format that can be
// used to create an OSPRay AMR volume
class Reader
{
 public:
  // constructor (opens the file)
  Reader(const std::string &a_filename, int a_mpiRank);

  // reads metadata such as the components and numLevels from the main header
  // under "/",
  void readMainHeader();

  // reads metadata for every level
  void readLevelHeaders();

  // reads blockBounds for every level
  int readBlocks();

  // read all block data
  int readBlockData(const std::string &a_compName);

 private:
  Handle m_handle; // the file handle
  const int m_mpiRank;
  int m_numComps; // the number of components in the file (we will only read
                  // one)
  std::map<std::string, int> m_compMap; // the names of the components
  int m_numLevels;
  bool m_mainHeaderRead = false;
  bool m_blocksRead = false;
  std::vector<int>
      m_refRatios; // the ratio of a level's resolution to the next coarser one
  std::vector<float> m_cellWidths; // the width of a single cell on each level
  std::vector<rkcommon::math::box3i>
      m_blockBounds; // all the blocks on every level
  std::vector<int> m_blockLevels; // which level each block belongs to
  std::vector<int> m_numBlocksPerLevel; // number of blocks on each level
  int m_totalNumBlocks;
  std::vector<int> m_rankDataOwner; // the MPI rank on which the data for each
                                    // block will be on
  std::vector<std::vector<float>>
      m_blockDataVector; // the flattened data in each box.

  void setRankDataOwner();

  // reads in the data for a single block on a given level
  int readSingleBlockData(int a_levelBlockIdx,
      int a_level,
      int a_comp,
      hid_t a_level_dataset_id,
      hid_t a_level_dataspace_id);
};

} // namespace ChomboHDF5
#endif
