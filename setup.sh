#!/bin/bash

echo "> Setting up Libtorch..."
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
rm -f libtorch-shared-with-deps-latest.zip

echo "> Setting up Pybind..."
wget https://github.com/pybind/pybind11/archive/refs/heads/master.zip
unzip master.zip
rm -f master.zip
mv pybind11-master pybind11
