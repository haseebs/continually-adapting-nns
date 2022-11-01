#CC Build instructions
# module load cmake mariadb tbb cuda
cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message( "Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}" )

option(WORKFLOW_COMPILER "Flags used when compiling using github workflow" OFF)
if(WORKFLOW_COMPILER)
    set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread -ldl -lm -I/usr/include/mariadb")
else()
    set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")
endif(WORKFLOW_COMPILER)

find_package(Torch REQUIRED)

add_subdirectory(pybind11)
include_directories(pybind11/include/pybind11)


# For pruning on the binary MNIST classification task
add_executable(BinaryMNISTPruning src/utils.cpp include/utils.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
        include/nn/message.h src/nn/message.cpp include/nn/dynamic_elem.h src/nn/dynamic_elem.cpp
        include/nn/networks/synced_network.h src/nn/networks/synced_network.cpp
        include/nn/synced_neuron.h include/nn/synced_synapse.h src/nn/synced_neuron.cpp src/nn/synced_synapse.cpp
        include/nn/networks/pretrained_dense_network.h src/nn/networks/pretrained_dense_network.cpp
        include/json.hpp experiment_prune_binary_mnist.cpp)

# For testing backpropagation on a small linear model
#add_executable(TestLinearModel src/utils.cpp include/utils.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
#        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
#        include/nn/message.h src/nn/message.cpp include/nn/dynamic_elem.h src/nn/dynamic_elem.cpp
#        include/nn/networks/synced_network.h src/nn/networks/synced_network.cpp
#        include/nn/synced_neuron.h include/nn/synced_synapse.h src/nn/synced_neuron.cpp src/nn/synced_synapse.cpp
#        include/nn/networks/small_network.h src/nn/networks/small_network.cpp
#        include/json.hpp linear_model_test.cpp)

# For testing the export of pytorch trained models into c++
#add_executable(TestModelExport src/utils.cpp include/utils.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
#        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
#        include/nn/message.h src/nn/message.cpp include/nn/dynamic_elem.h src/nn/dynamic_elem.cpp
#        include/nn/networks/synced_network.h src/nn/networks/synced_network.cpp
#        include/nn/synced_neuron.h include/nn/synced_synapse.h src/nn/synced_neuron.cpp src/nn/synced_synapse.cpp
#        include/nn/networks/pretrained_dense_network.h src/nn/networks/pretrained_dense_network.cpp
#        include/json.hpp tests/cpp_mnist_prediction.cpp)
#

# For counting the total number of runs based on the config file
add_executable(Count include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        count_configs.cpp)

# Pybind interface for using the experiment and logging code from c++ in python
pybind11_add_module(FlexibleNN src/utils.cpp include/utils.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
        pybind_interface/interface.cpp)


# For linking locally
target_link_libraries(FlexibleNN PUBLIC tbb)
target_link_libraries(FlexibleNN PUBLIC mariadb)
target_link_libraries(FlexibleNN PUBLIC torch)

target_link_libraries(BinaryMNISTPruning PUBLIC tbb)
target_link_libraries(BinaryMNISTPruning PUBLIC mariadb)
target_link_libraries(BinaryMNISTPruning PUBLIC torch)

#target_link_libraries(TestLinearModel PUBLIC tbb)
#target_link_libraries(TestLinearModel PUBLIC mariadb)
#target_link_libraries(TestLinearModel PUBLIC torch)
#
#target_link_libraries(TestModelExport PUBLIC tbb)
#target_link_libraries(TestModelExport PUBLIC mariadb)
#target_link_libraries(TestModelExport PUBLIC torch)

target_link_libraries(Count PUBLIC tbb)
target_link_libraries(Count PUBLIC mariadb)

project(FlexibleNN LANGUAGES CXX)
