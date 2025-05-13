#CC Build instructions
# module load cmake mariadb tbb cuda
cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Enable debug build
set(CMAKE_BUILD_TYPE Debug)

add_compile_options(-include cstdint)

find_package(PkgConfig REQUIRED)
# Find pthreads
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

pkg_check_modules(mongocxx REQUIRED libmongocxx)

link_directories(${mongocxx_LIBRARY_DIRS})
include_directories(${mongocxx_INCLUDE_DIRS})
# For testing decorrelators when using generate and test
add_executable(GNTDecorrelation
        src/utils.cpp
        experiment_decorrelator_generate_n_test.cpp)

add_executable(AtariExp
        src/utils.cpp
        experiment_atari.cpp)

add_executable(Synthetic
        src/utils.cpp
        experiment_synthetic_nonstationary.cpp)

add_subdirectory(src)
target_link_libraries(GNTDecorrelation PUBLIC NN Exp Logger ${mongocxx_LIBRARIES} Atari Threads::Threads Evaluators TBB::tbb)
target_link_libraries(AtariExp PUBLIC NN Exp Logger ${mongocxx_LIBRARIES} Atari Threads::Threads Evaluators TBB::tbb)
target_link_libraries(Synthetic PUBLIC NN Exp Logger ${mongocxx_LIBRARIES} Atari Threads::Threads Evaluators TBB::tbb)

project(FlexibleNN LANGUAGES CXX)
