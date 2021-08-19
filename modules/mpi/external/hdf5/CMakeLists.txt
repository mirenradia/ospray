# Find HDF5
set(HDF5_PREFER_PARALLEL TRUE)
set(HDF5_FIND_DEBUG TRUE)
find_package(HDF5 REQUIRED)
if (NOT HDF5_IS_PARALLEL)
  message(FATAL_ERROR "Found HDF5 library has not been built with MPI")
endif()

# Create hdf5::hdf5 imported target if not done by CMake FindHDF5
# module (v < 3.19)
if (NOT TARGET hdf5::hdf5)
  add_library(hdf5::hdf5 UNKNOWN IMPORTED)
  string(REPLACE "-D" "" _hdf5_definitions "${HDF5_C_DEFINITIONS}")
  if (DEFINED "HDF5_hdf5_LIBRARY")
    set(_hdf5_location "${HDF5_$hdf5_LIBRARY}")
  elseif (DEFINED "HDF5_C_LIBRARY")
    set(_hdf5_location "${HDF5_C_LIBRARY}")
  elseif (DEFINED "HDF5_C_LIBRARY_hdf5")
    set(_hdf5_location "${HDF5_C_LIBRARY_hdf5}")
  else ()
    # Error if we still don't have the location.
    message(SEND_ERROR
      "HDF5 was found, but a different variable was set which contains "
      "the location of the `hdf5::hdf5` library.")
  endif ()

  set_target_properties(hdf5::hdf5 PROPERTIES
    IMPORTED_LOCATION "${_hdf5_location}"
    INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS};${MPI_C_INCLUDE_PATH}"
    INTERFACE_COMPILE_DEFINITIONS "${_hdf5_definitions}"
  )
  unset(_hdf5_location)
  unset(_hdf5_definitions)
endif()

# Chombo HDF5 reader library
add_library(chombo_hdf5_reader STATIC ChomboHDF5Reader.cpp)
target_link_libraries(chombo_hdf5_reader PUBLIC
  ospray
  hdf5::hdf5
)
target_include_directories(chombo_hdf5_reader INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}
)
