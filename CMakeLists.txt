#CC Build instructions
# module load cmake mariadb tbb cuda
cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Enable debug build
set(CMAKE_BUILD_TYPE Debug)


# message( "Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}" )

# option(WORKFLOW_COMPILER "Flags used when compiling using github workflow" OFF)
# if(WORKFLOW_COMPILER)
#     set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread -ldl -lm -I/usr/include/mariadb")
# else()
#     set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")
# endif(WORKFLOW_COMPILER)

find_package(PkgConfig REQUIRED)
# Find pthreads
find_package(Threads REQUIRED)
pkg_check_modules(mongocxx REQUIRED libmongocxx)

link_directories(${mongocxx_LIBRARY_DIRS})
include_directories(${mongocxx_INCLUDE_DIRS})
# For testing decorrelators when using generate and test
add_executable(GNTDecorrelation
        src/utils.cpp
        experiment_decorrelator_generate_n_test.cpp)


add_subdirectory(src)
target_link_libraries(GNTDecorrelation PUBLIC NN Exp Logger ${mongocxx_LIBRARIES} Atari Threads::Threads)

project(FlexibleNN LANGUAGES CXX)
