#include "ChomboHDF5Reader.h"
#include <string.h>
#include <stdexcept>

namespace ChomboHDF5 {

bool Handle::s_initialized = false;
hid_t Handle::box3i_id = 0;
hid_t Handle::vec3i_id = 0;
hid_t Handle::vec3d_id = 0;

using namespace rkcommon::math;

void Handle::initialize()
{
  // create HDF5 types
  box3i_id = H5Tcreate(H5T_COMPOUND, sizeof(box3i));
  H5Tinsert(box3i_id, "lo_i", HOFFSET(box3i, lower.x), H5T_NATIVE_INT);
  H5Tinsert(box3i_id, "lo_j", HOFFSET(box3i, lower.y), H5T_NATIVE_INT);
  H5Tinsert(box3i_id, "lo_k", HOFFSET(box3i, lower.z), H5T_NATIVE_INT);
  H5Tinsert(box3i_id, "hi_i", HOFFSET(box3i, upper.x), H5T_NATIVE_INT);
  H5Tinsert(box3i_id, "hi_j", HOFFSET(box3i, upper.y), H5T_NATIVE_INT);
  H5Tinsert(box3i_id, "hi_k", HOFFSET(box3i, upper.z), H5T_NATIVE_INT);

  vec3i_id = H5Tcreate(H5T_COMPOUND, sizeof(vec3i));
  H5Tinsert(vec3i_id, "intvecti", HOFFSET(vec3i, x), H5T_NATIVE_INT);
  H5Tinsert(vec3i_id, "intvectj", HOFFSET(vec3i, y), H5T_NATIVE_INT);
  H5Tinsert(vec3i_id, "intvectk", HOFFSET(vec3i, z), H5T_NATIVE_INT);

  // note that even though we will want floats rather than doubles, they
  // will be doubles in the HDF5 file
  vec3d_id = H5Tcreate(H5T_COMPOUND, sizeof(vec3d));
  H5Tinsert(vec3d_id, "x", HOFFSET(vec3d, x), H5T_NATIVE_DOUBLE);
  H5Tinsert(vec3d_id, "y", HOFFSET(vec3d, y), H5T_NATIVE_DOUBLE);
  H5Tinsert(vec3d_id, "z", HOFFSET(vec3d, z), H5T_NATIVE_DOUBLE);

  s_initialized = true;
}

Handle::Handle() : m_isOpen{false}
{
  if (!s_initialized)
    initialize();
}

Handle::~Handle()
{
  if (m_isOpen) {
    close();
  }
}

bool Handle::isOpen() const
{
  return m_isOpen;
}

int Handle::open(const std::string &a_filename)
{
  if (m_isOpen) {
    throw std::runtime_error("Calling open() on an already open file.");
  }

  m_filename = a_filename;
  m_group = "/";

  hid_t file_access = 0;

  // Make HDF5 aware of MPI communicator
  file_access = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(file_access, MPI_COMM_WORLD, MPI_INFO_NULL);

  m_fileID = H5Fopen(a_filename.c_str(), H5F_ACC_RDONLY, file_access);
  // negative fileID is an error
  if (m_fileID < 0)
    return m_fileID;

  H5Pclose(file_access);

  m_currentGroupID = H5Gopen2(m_fileID, m_group.c_str(), H5P_DEFAULT);

  if (m_fileID >= 0 && m_currentGroupID >= 0)
    m_isOpen = true;

  int ret = verifyChomboFile();
  return ret;
}

int Handle::verifyChomboFile()
{
  int ret = 0;
  const std::string chomboGlobalGroupName = "Chombo_global";
  hid_t group = H5Gopen2(m_fileID, chomboGlobalGroupName.c_str(), H5P_DEFAULT);
  if (group < 0) {
    throw std::runtime_error("File " + m_filename + " does not contain a "
        + chomboGlobalGroupName + " group, so cannot be a Chombo HDF5 file.");
  }

  HeaderData header_info;
  header_info.readFromLocation(group);
  if (header_info.m_int.find("SpaceDim") == header_info.m_int.end()) {
    throw std::runtime_error("File "
        + m_filename +
        " does not contain SpaceDim"
        " in the Chombo_global group, so cannot be a Chombo HDF5 file.");
  } else if (header_info.m_int["SpaceDim"] != 3) {
    throw std::runtime_error("This code only works with SpaceDim = 3 but file"
        + m_filename + " has SpaceDim = "
        + std::to_string(header_info.m_int["SpaceDim"]) + ".");
  }

  // if this HDF5 file uses single precision, this won't be read by HeaderData
  // so need to verify separately
  hid_t attr =
      H5Aopen_by_name(group, ".", "testReal", H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0)
    return 1;
  hid_t attr_type = H5Aget_type(attr);
  size_t fileprecision = H5Tget_precision(attr_type);
  size_t codeprecision = sizeof(double);

  if (fileprecision > codeprecision) {
    ret = 2;
  } else if (codeprecision > fileprecision) {
    std::cerr << "File" << m_filename
              << "floating point precision = " << fileprecision << " bits"
              << " but we assume " << codeprecision << " bits." << std::endl;
    ret = 2;
  }
  H5Aclose(attr);
  H5Tclose(attr_type);
  H5Gclose(group);

  return ret;
}

void Handle::close()
{
  if (!m_isOpen) {
    throw std::runtime_error("Calling close() on an unopened file handle");
  }
  if (m_currentGroupID >= 0)
    H5Gclose(m_currentGroupID);
  if (m_fileID >= 0)
    H5Fclose(m_fileID);
  m_isOpen = false;
}

int Handle::setGroup(const std::string &a_group)
{
  if (!m_isOpen) {
    throw std::runtime_error("Cannot set group unless file is open");
  }
  if (m_group == a_group)
    return 0;

  int ret = 0;

  ret = H5Gclose(m_currentGroupID);
  if (ret < 0) {
    std::cerr << "ChomboHDF5: error closing old group" << std::endl;
    return ret;
  }

  m_currentGroupID = H5Gopen2(m_fileID, a_group.c_str(), H5P_DEFAULT);

  // put things back to how they were if opening the group failed
  if (m_currentGroupID < 0) {
    m_currentGroupID = H5Gopen2(m_fileID, m_group.c_str(), H5P_DEFAULT);
    ret = -1;
  } else {
    m_group = a_group;
  }

  return ret;
}

void Handle::setGroupToLevel(int a_level)
{
  char level_str[100];
  sprintf(level_str, "level_%i", a_level);
  setGroup(level_str);
}

const std::string &Handle::getGroup() const
{
  return m_group;
}

const hid_t &Handle::fileID() const
{
  return m_fileID;
}

const hid_t &Handle::groupID() const
{
  return m_currentGroupID;
}

int HeaderData::readFromFile(Handle &a_handle)
{
  return readFromLocation(a_handle.groupID());
}

int HeaderData::readFromLocation(hid_t a_loc_id)
{
  hsize_t *n = nullptr;
  return H5Aiterate2(a_loc_id,
      H5_INDEX_CRT_ORDER,
      H5_ITER_NATIVE,
      n,
      ChomboHDF5HeaderDataAttributeScan,
      this);
}

void HeaderData::clear()
{
  m_double.clear();
  m_int.clear();
  m_string.clear();
  m_vec3i.clear();
  m_box3i.clear();
  m_vec3d.clear();
}

extern "C" {
herr_t ChomboHDF5HeaderDataAttributeScan(hid_t a_loc_id,
    const char *a_name,
    const H5A_info_t *a_info,
    void *a_opdata)
{
  herr_t ret = 0;
  HeaderData &data = *(static_cast<HeaderData *>(a_opdata));

  hid_t attr = H5Aopen_by_name(a_loc_id, ".", a_name, H5P_DEFAULT, H5P_DEFAULT);
  hid_t attr_type = H5Aget_type(attr);
  hid_t attr_class = H5Tget_class(attr_type);
  char *buf = NULL;
  size_t size = 0;

  switch (attr_class) {
  case H5T_INTEGER:
    int Ivalue;
    ret = H5Aread(attr, H5T_NATIVE_INT, &Ivalue);
    if (ret < 0)
      break;
    data.m_int[a_name] = Ivalue;
    break;
  case H5T_FLOAT:
    double Dvalue;
    ret = H5Aread(attr, H5T_NATIVE_DOUBLE, &Dvalue);
    if (ret < 0)
      break;
    data.m_double[a_name] = Dvalue;
    break;
  case H5T_STRING:
    size = H5Tget_size(attr_type);
    buf = new char[size + 1];
    ret = H5Aread(attr, attr_type, buf);
    if (ret < 0)
      break;
    buf[size] =
        0; // for some reason HDF5 is not null terminating strings correctly
    data.m_string[a_name] = std::string(buf);
    break;
  case H5T_COMPOUND:
    if (strcmp(H5Tget_member_name(attr_type, 0), "lo_i") == 0) {
      box3i value;
      ret = H5Aread(attr, Handle::box3i_id, &value);
      if (ret < 0)
        break;
      data.m_box3i[a_name] = value;
      break;
    } else if (strcmp(H5Tget_member_name(attr_type, 0), "intvecti") == 0) {
      vec3i value;
      ret = H5Aread(attr, Handle::vec3i_id, &value);
      if (ret < 0)
        break;
      data.m_vec3i[a_name] = value;
      break;
    } else if (strcmp(H5Tget_member_name(attr_type, 0), "x") == 0) {
      vec3d value;
      ret = H5Aread(attr, Handle::vec3d_id, &value);
      if (ret < 0)
        break;
      data.m_vec3d[a_name] = value;
      break;
    }
  default:
    std::cerr
        << "ChomboHDF5HeaderDataAttributeScan encountered unrecognized attributes"
        << std::endl;
  }
  delete[] buf;
  H5Tclose(attr_type);
  H5Aclose(attr);
  return ret;
}
}
} // namespace ChomboHDF5
