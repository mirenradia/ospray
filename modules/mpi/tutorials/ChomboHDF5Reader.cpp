#include "ChomboHDF5Reader.h"

void print_hdf5_lib_version()
{
  unsigned int maj_version, min_version, rel_number;
  H5get_libversion(&maj_version, &min_version, &rel_number);

  std::cout << "HDF5 Lib v" << maj_version << "." << min_version << "."
            << rel_number << std::endl;
}
