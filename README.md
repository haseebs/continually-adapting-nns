# Online Network Pruning
![Build](https://github.com/haseebs/online-network-pruning/actions/workflows/cmake.yml/badge.svg?event=push) ![cpplint](https://github.com/haseebs/online-network-pruning/actions/workflows/cpplint.yml/badge.svg?event=push)

This repository contains the code for implementing and evaluating neural network pruners in the online continual
learning setting.

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
