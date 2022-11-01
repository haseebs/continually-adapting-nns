# Continually Adapting Networks
![Build](https://github.com/haseebs/online-network-pruning/actions/workflows/cmake.yml/badge.svg?event=push) ![cpplint](https://github.com/haseebs/online-network-pruning/actions/workflows/cpplint.yml/badge.svg?event=push)

The vision for this project is to build algorithms for the automatic
adaptation of neural networks based solely on the observed stream of
experience in order to obtain a compact representation.
The focus here is on two main components: Generation/discovery and Testing/pruning of the features.
Currently, there are two active research directions in this project:

1) Feature Decorrelator
When rapidly generating many features, we end up with many highly correlated features.
The challenge here is to detect and remove these redundant features online and in a
scalable way. Additionally, maintaining the model performance after removing these
features is not trivial since these features often tend to have high utilities
(high contributions to the output).

2) Feature Tester
Feature testing here involves quickly and accurately evaluating the features to
determine which ones we should keep and which we can replace. Most existing
pruning strategies are unsuitable here since they are too expensive
or unsuitable to run online.

## Requirements
In order to run this project, you'll need the following things installed:
* GCC >= 9.3.0
* CMake
* [MariaDB](https://mariadb.com/kb/en/getting-installing-and-upgrading-mariadb/) (and a C++ connector for MariaDB
  found [here](https://mariadb.com/kb/en/mariadb-connector-c/))
* Python >= 3.6

## Setup Instructions
* Install python packages: `pip install -r requirements.txt`
* Setup Pybind11 and Libtorch: `bash setup.sh`
* Compile: `bash compile.sh`

## Directory Structure
* `include/`, `src/`: Implementation of network primitives and pruning strategies in C++
* `model_pretrainers/`: Pytorch code for pretraining and saving MNIST models
* `cfg/`: Config files
* `data/`: Dataset files are saved here
* `results/`: Notebooks for plotting and visualization of results
* `tests/`: Misc tests
* `trained_models/`: Pretrained models are saved here

## Running the code
The first step is to obtain some pretrained MNIST models using: `python model_pretrainers/train_binary_mnist.py --config cfg/<config_file>`

Next up, evaluate the pruners on these pretrained models using: `./BinaryMNISTPruning --config cfg/<config_file>`

See `cfg/` for example of some config files.
