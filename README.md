# BlockFormer: An Efficient Private Batch Transformer Inference with Homomorphic Encryption via Block Packing

This repository contains the implementation of the paper **BlockFormer: An Efficient Private Batch Transformer Inference with Homomorphic Encryption via Block Packing**.
The paper was accepted at the IEEE Access, available [here]().

## Prerequests
1. Git & Docker

## Project Structure
- examples: Core Imprelementation of BlockFormer


## Installation
```
# Clone the project
git clone https://github.com/whcjimmy/BlockFormer
cd BlockFormer

# Pull the HEAAN docker
docker pull cryptolabinc/heaan:0.2.0-x86_64-hexl-ubuntu22.04

# Run the docker container
docker run -v ./:/app -it cryptolabinc/heaan:0.2.0-x86_64-hexl-ubuntu22.04 bash


# --- The following commands should be executed inside the container ---

# Install requested libraries
apt update 
apt install -y cmake 

# Build the BlockFormer Code
cd /app/examples
mkdir build && cd build 
cmake .. && make -j8
```

## Usage
1. To evaluate PCBMM, run ```./bin/pcmm [row of matrix A] [col of matrix A] [col of matrix B] [alpha] [ct level] [use gpu?]```
2. To evaluate SSBMM and SRBMM, run ```./bin/ccmm [input length] [alpha] [use gpu?]```
3. To evaluate non-linear functions, run ```./bin/nonlinear [name of non-linear funciton] [input length] [alpha] [use gpu?]```


To enable GPU support, three steps are required:
1. Prepare the HEAAN library with GPU support.
2. Set the last parameter of the execution to 1; otherwise, set it to 0.
3. Comment out the lines referring to "CPU MODE" and uncomment the lines referring to "GPU MODE" in the source code and cmakelists.

## Citation
```
@article{BlockFormer,
  title={{BlockFormer: An Efficient Private Batch Transformer Inference with Homomorphic Encryption via Block Packing}},
  author={Wang, Huan-Chih and Wu, Ja-Ling},
  booktitle={IEEE Access},
  year={2026}
}
```
