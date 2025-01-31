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


# For testing decorrelators when using generate and test
add_executable(GNTDecorrelation include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp
        src/utils.cpp include/utils.h src/nn/utils.cpp include/nn/utils.h
        include/nn/networks/single_layer_network.h src/nn/networks/single_layer_network.cpp
        include/nn/synced_neuron.h include/nn/synced_synapse.h src/nn/synced_neuron.cpp src/nn/synced_synapse.cpp
        include/nn/dynamic_elem.h include/nn/message.h src/nn/dynamic_elem.cpp src/nn/message.cpp
        experiment_decorrelator_generate_n_test.cpp)

# For counting the total number of runs based on the config file
add_executable(Count include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        count_configs.cpp)



# For linking locally
target_link_libraries(GNTDecorrelation PUBLIC tbb)
target_link_libraries(GNTDecorrelation PUBLIC mariadb)

target_link_libraries(Count PUBLIC tbb)
target_link_libraries(Count PUBLIC mariadb)

project(FlexibleNN LANGUAGES CXX)
