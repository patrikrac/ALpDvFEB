cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

project(mfem_general)

#find_library(mfem_LIBRARY mfem REQUIRED)
#find_path(mfem_INCLUDE_DIR NAMES mfem.hpp REQUIRED)
message(STATUS "Looking for mfem ...")
find_package(mfem REQUIRED NAMES MFEM)
message(STATUS "Found mfem config in: ${mfem_DIR} (version ${MFEM_VERSION})")

message(STATUS "Mfem compiler is: ${MFEM_CXX_COMPILER}")
if (NOT CMAKE_CXX_COMPILER AND MFEM_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${MFEM_CXX_COMPILER}")
endif()

enable_language(C)
enable_language(CXX)

target_include_directories(mfem_general PRIVATE ${MFEM_INCLUDE_DIRS})

add_executable(mfem_general mfem_general.cpp)

target_link_libraries(mfem_general PRIVATE ${MFEM_LIBRARIES})

