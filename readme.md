# CppFlowATR
###### a fork by borisef

## About 

Implements inference for ATR followed by color classifier 
Both ATR model and color classifier are tensorflow models (frozen .PB files) 

Operates in multi-thread fashion (1 thread for CPU, 1 thread mainly for GPU) in such a way some frames are skipped if GPU is busy. 
Supports several additional functionalities:
* log file, config file
* several input formats
* works in cropped region
* post-processing user-defined filters (e.g. NMS, per class threshold score, size filter etc.)
* supports resize mode 

Folders structure: 
   * config - contains config files (json files)
   * graphs - location of models (pb files)
   * tests - cpp files for tests 
   * include - h files (source code)
   * src - cpp files (source code)
   * media - images, videos, raw files etc. 



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

3) Test with build/stressTestSmall or build/stressTest or build/videoTest
