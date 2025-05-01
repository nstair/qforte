#!/bin/bash

which conda > /dev/null 2>&1 || { echo "conda not found..exiting"; exit 1;}

if [ $# -gt 0 ]; then
    QFORTE_CONDA_ENV=$1
else
    QFORTE_CONDA_ENV="qforte-default-env"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

#CREATE ENVIRONMENT
conda create -n "$QFORTE_CONDA_ENV" python=3.8

#INSTALL REQUIRED PACKAGES
conda install -n "$QFORTE_CONDA_ENV" -c conda-forge psi4 cmake openblas libopenblas

#TODO: ADD AN OPTION TO INSTALL CUDA
#conda install -n $QFORTE_CONDA_ENV -c nvidia cuda

#ACTIVATE CONDA ENV
conda activate $QFORTE_CONDA_ENV

#SET THE CMAKE PREFIX TO THE CONDA PREFIX
sed -i "s|set(CMAKE_PREFIX_PATH \".*\")|set(CMAKE_PREFIX_PATH \"$CONDA_PREFIX\")|" CMakeLists.txt

#SET LIBOPENBLAS PATH DEPENDING ON OS
OS="$(uname)"
if [ "$OS" = "Linux" ]; then
	echo "You are on Linux: using libopenblas.so"
	sed -i 's|set(OPENBLAS_EXE ".*")|set(OPENBLAS_EXE ${CMAKE_PREFIX_PATH}/lib/libopenblas.so)|' CMakeLists.txt
elif [ "$OS" = "Darwin" ]; then
	echo "You are on MacOS: using libopenblas.dylib"
	sed -i 's|set(OPENBLAS_EXE ".*")|set(OPENBLAS_EXE ${CMAKE_PREFIX_PATH}/lib/libopenblas.dylib)|' CMakeLists.txt
	exit
else
	echo "Unknown OS -> exiting..."
	exit
fi

#BUILD
python setup.py develop && echo "qforte successfully installed" || { echo "qforte failed to install"; exit 1; }
