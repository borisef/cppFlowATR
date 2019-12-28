# CppFlowATR
###### a fork by borisef

## Installation

- Clone the project:
```sh
   git clone https://github.com/borisef/cppFlowATR.git
```

- Download Complied libtensorflow[version].so 1.13 - 15 from [TensorFlow](https://www.tensorflow.org/install/lang_c), and its source files (models).


### Requirements

* **Dependencies** 
  * CUDA 10.0
  * CUDnn 7.6.4
  * OpenCV 3.4.x



* Files for testing:
    * frozen_inference_graph_humans.pb
    * 00000018.tif
    * 00006160.raw

### How To Run It

1) Make sure the paths in `CMakeLists.txt` are correct

2) Build files using cmake:
```sh
mkdir build
cd build
cmake ..
make ```