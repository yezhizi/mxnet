# Build MXnet from source with GPU support
## Of My Environment
- OS: Ubuntu 20.04(WSL2)
- GPU: NVIDIA GeForce GTX 3060
- CUDA: 11.2
- cuDNN: 8.1.1
- Python: 3.9
- gcc version: 9.3.0
- MXnet to build: 1.9.1
> Note: Choose your cuda and cnDNN version  according to [here](https://developer.nvidia.com/cuda-gpus) and [here](https://docs.nvidia.com/deeplearning/cudnn/reference/support-matrix.html) and [this link](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)


## Install Prerequisites
Please follow the official guide to install CUDA and cuDNN first. 
Then install the following packages:
```bash
sudo apt-get update
sudo apt-get install -y build-essential git ninja-build ccache libopenblas-dev libopencv-dev cmake
```
If you want to install `kvstore` for distributed training, you need to install `zmq` and `protobuf`:
```bash 
sudo apt install libprotobuf-dev protobuf-compiler libzmq3-dev
```

## Get the Source Code
You can get the source code from the official repository:
```bash
git clone --recursive https://github.com/apache/mxnet
```
Since MXnet is a large project, you can try to use `--depth 1` to clone a shallow copy of the repository and choose the specific branch you want to build. Then update the submodule:
```bash
git clone  --depth=1 --branch=v1.9.1 https://github.com/apache/mxnet

git submodule update --init --recursive --depth=1
```

## Configure build options
Cmake is the recommended way to build MXnet. You can copy the `config/linux_gpu.cmake` to the root directory and modify it according to your needs. 
```bash
cp ./config/linux_gpu.cmake config.cmake
```
Then modify the `config.cmake` file according to your needs. For example, I modified the following lines:
```cmake
set(USE_CUDA ON CACHE BOOL "Build with CUDA support")
set(USE_CUDNN ON CACHE BOOL "Build with cudnn support, if found")

set(USE_OPENCV ON CACHE BOOL "Build with OpenCV support")

# distributed computing
set(USE_DIST_KVSTORE ON CACHE BOOL "Build with DIST_KVSTORE support")

```

## Build
```bash
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release  ..
cmake --build . --parallel 8
```
> Note: the `--parallel` option is used to speed up the build process. Compile MXnet is a memory-consuming process, so you need to make sure you have enough memory to compile it or you may lower the number of parallel jobs.

> It may terminate when building, you can check and if there is no specific error you can simply run the command again to continue(it may be caused by the memory limit).

## Install Python Package
After building, you can install the python package:
```bash
cd python
python -m pip install  --user -e .
```
Then you can test the installation:
```bash
python -c "import mxnet as mx; arr = mx.nd.array([1,2,3],ctx=mx.gpu()); print(mx.__version__);print(arr)"
```
If you see the version of MXnet and the array printed, then you have successfully installed MXnet with GPU support.

## Some Issues You May Encounter
### 1. Cannot find libmxnet.so
When you import mxnet in python, it may raise an error that it cannot find `libmxnet.so`. You can copy the `build/libmxnet.so`(which you built in the previous step) to the directory it tells you to. 

### 2. 100% mxnet_unit_test failed or ld: cannot find -lcuda 
Add the following line to your .bashrc or .zshrc (depending on your shell) should solve the problem([link](https://github.com/NVlabs/tiny-cuda-nn/issues/183)):
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"

You .bashrc or .zshrc should look like this:
```bash
# cuda
export CUDA_HOME="/usr/local/cuda"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/extras/CUPTI/lib64"
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:$LIBRARY_PATH"
export PATH="$PATH:$CUDA_HOME/bin"
```


### 3. AttributeError: module 'numpy' has no attribute 'bool'
This is caused by the numpy version. You can downgrade the numpy version to 1.23.1:
```bash
pip uninstall numpy
pip install numpy==1.23.1
```
Then you can test the installation again.
