##
#  CMake script for the step-3 tutorial program:
##

# Set the name of the project and target:
SET(TARGET "deal_main")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
FILE(GLOB_RECURSE TARGET_SRC  "source/*.cpp")
FILE(GLOB_RECURSE TARGET_INC  "source/*.hpp")
SET(TARGET_SRC ${TARGET_SRC} ${TARGET_INC} )
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
#SET(TARGET_SRC ${TARGET}.cpp)

option ( USE_TIMING "ENABLE TIME MEASUREMENTS" ON )
option ( USE_OUTPUT "ENABLE DATA OUTPUT" ON )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(deal.II 10.0.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

if ( USE_TIMING )
  target_compile_definitions( ${TARGET} PRIVATE "-DUSE_TIMING" )
endif()

if ( USE_OUTPUT )
  target_compile_definitions( ${TARGET} PRIVATE "-DUSE_OUTPUT" )
endif()