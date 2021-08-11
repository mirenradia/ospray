#include "ChomboHDF5Reader.h"

namespace ChomboHDF5 {

bool Handle::s_initialized = false;
hid_t Handle::box3i_id = 0;
hid_t Handle::vec3i_id = 0;
hid_t Handle::vec3d_id = 0;

void Handle::initialize()
{
  using namespace rkcommon::math;

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
  assert(!m_isOpen);

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

  return 0;
}

void Handle::close()
{
  assert(m_isOpen);
  if (m_currentGroupID >= 0)
    H5Gclose(m_currentGroupID);
  if (m_fileID >= 0)
    H5Fclose(m_fileID);
  m_isOpen = false;
}
} // namespace ChomboHDF5
