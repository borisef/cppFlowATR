1) clone my project 
   https://github.com/borisef/cppFlowATR.git

2) Download compiled  libtensrorflow***so 1.13 - 1.15
from https://www.tensorflow.org/install/lang_c

And its source files (model)

3) Make sure you have 
* CUDA 10.0
* CUDnn 7.6.4
* OpenCV 3.4.x

4) Make sure you have for testing :
    frozen_inference_graph_humans.pb
    00000018.tif
    00006160.raw

5) Make sure the paths is CMakeLists.txt are correct

6) mkdir build
   cd build
   cmake ..
   make