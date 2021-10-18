#ifndef CHOMBO_HDF5_READER_H
#define CHOMBO_HDF5_READER_H

#include <hdf5.h>
#include <map>
#include <string>
#include <vector>
#include "ospray/ospray_cpp.h"
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
  // constructor
  // opens the file, reads it, creates the volume and then closes the file
  Reader(const std::string &a_filename,
      int a_mpiRank,
      int a_mpiWorldSize,
      const std::string &a_compName);

  // calculate the min and max values in the data
  rkcommon::math::vec2f calculateValueRange();

  // can pass by value since the Volume object itself does not contain very much
  ospray::cpp::Volume getVolume();

  // get bounding box of whole domain
  rkcommon::math::box3i getDomainBounds_int();
  rkcommon::math::box3f getDomainBounds();

  // returns the boundaries of the regions for which the data is on this rank
  // note that this does NOT contain ghosts
  const std::vector<rkcommon::math::box3f> &getMyRegions();

  // returns refinements per level
  const std::vector<int> &getMyRefRatios();

  // returns the bounds of the blocks without ghosts
  const std::vector<rkcommon::math::box3i> &getBlockBounds();

  // returns the bounds of the blocks with ghosts
  const std::vector<rkcommon::math::box3i> &getGhostedBlockBounds();

  // returns the number of ghost cells on each level
  const std::vector<rkcommon::math::vec3i> &getNumGhosts();

  // returns refinements per level
  const std::vector<int> &getRankDataOwner();

  // returns level number for each block
  const std::vector<int> &getBlockLevels();

 private:
  Handle m_handle; // the file handle
  const int m_mpiRank;
  const int m_mpiWorldSize;
  int m_numComps; // the number of components in the file (we will only read
                  // one)
  std::map<std::string, int> m_compMap; // the names of the components
  int m_numLevels;
  rkcommon::math::box3i m_domainBounds_int; // the box that is the whole domain
  rkcommon::math::box3f m_domainBounds; // the box that is the whole domain
  bool m_mainHeaderRead = false;
  bool m_levelHeadersRead = false;
  bool m_blocksRead = false;
  std::vector<int>
      m_refRatios; // the ratio of a level's resolution to the next coarser one
  std::vector<float> m_cellWidths; // the width of a single cell on each level
  std::vector<rkcommon::math::box3i>
      m_blockBounds; // all the blocks on every level (without ghosts)
  std::vector<int> m_blockLevels; // which level each block belongs to
  std::vector<int> m_numBlocksPerLevel; // number of blocks on each level
  int m_totalNumBlocks;
  std::vector<rkcommon::math::vec3i>
      m_numGhosts; // the number of ghosts in each direction on each level
  std::vector<rkcommon::math::box3i>
      m_ghostedBlockBounds; // all blocks on every level (with ghosts)
  std::vector<int> m_rankDataOwner; // the MPI rank on which the data for each
                                    // block will be on
  std::vector<rkcommon::math::box3f>
      m_myRegions; // the regions in the domain owned by this rank
  std::vector<std::vector<float>>
      m_blockDataVector; // the flattened data in each box.
  bool m_blockDataRead = false;
  ospray::cpp::Volume m_volume;

  // reads metadata such as the components and numLevels from the main header
  // under "/",
  void readMainHeader();

  // reads metadata for every level
  void readLevelHeaders();

  // reads blockBounds for every level
  int readBlocks();

  // converts a level index and level to a global block idx
  int levelToGlobalBlockIdx(int a_levelBlockIdx, int a_level);

  void setRankDataOwner();

  // read all block data and also ghost information (so sets
  // m_ghostedBlockBounds)
  int readBlockData(const std::string &a_compName);

  // reads in the data for a single block on a given level
  int readSingleBlockData(int a_levelBlockIdx,
      int a_level,
      int a_comp,
      hid_t a_level_dataset_id,
      hid_t a_level_dataspace_id);

  // sets the block data to a specified value for blocks owned by other ranks
  void setSingleBlockData(int a_levelBlockIdx, int a_level, float a_value);

  // create OSPRay AMR volume
  void createVolume();
};

} // namespace ChomboHDF5
#endif
