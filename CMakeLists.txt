#CC Build instructions
# module load cmake mariadb tbb cuda
cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message("Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}")

option(WORKFLOW_COMPILER "Flags used when compiling using github workflow" OFF)
if (WORKFLOW_COMPILER)
    set(CMAKE_CXX_FLAGS "-O3")
else ()
    set(CMAKE_CXX_FLAGS "-O3")
endif (WORKFLOW_COMPILER)


# For testing decorrelators when using generate and test
add_executable(GNTDecorrelation
        src/utils.cpp
        experiment_decorrelator_generate_n_test.cpp)

add_subdirectory(src)
target_link_libraries(GNTDecorrelation PUBLIC NN Exp Logger)


project(FlexibleNN LANGUAGES CXX)
