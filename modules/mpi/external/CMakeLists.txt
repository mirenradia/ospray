## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(pico_bench)
if (OSPRAY_MPI_USE_HDF5)
  add_subdirectory(hdf5)
endif()
