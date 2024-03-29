cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

SET(TARGET "mfem_main")

project(${TARGET})

option ( USE_TIMING "ENABLE TIME MEASUREMENTS" ON )
option ( USE_OUTPUT "ENABLE DATA OUTPUT" ON )

FILE(GLOB_RECURSE TARGET_SRC  "source/*.cpp")
FILE(GLOB_RECURSE TARGET_INC  "source/*.hpp")

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

add_executable(${TARGET} ${TARGET_INC} ${TARGET_SRC})

target_include_directories(${TARGET} PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE ${MFEM_LIBRARIES})

if ( USE_TIMING )
  target_compile_definitions( ${TARGET} PRIVATE "-DUSE_TIMING" )
endif()

if ( USE_OUTPUT )
  target_compile_definitions( ${TARGET} PRIVATE "-DUSE_OUTPUT" )
endif()