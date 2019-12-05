cmake_minimum_required(VERSION 3.10)
project(cppFlowATR)

set(CMAKE_CXX_STANDARD 17)
add_executable(main_like_coco src/main_like_coco.cpp src/Model.cpp src/Tensor.cpp)
find_package( OpenCV REQUIRED )
target_include_directories(main_like_coco PRIVATE include)
target_link_libraries (main_like_coco -ltensorflow ${OpenCV_LIBS})

add_executable(main_ATR src/main_ATR.cpp src/Model.cpp src/Tensor.cpp src/InterfaceATR.cpp)
find_package( OpenCV REQUIRED )
target_include_directories(main_ATR PRIVATE include)
FIND_LIBRARY(MY_LIB NAMES libstam.so PATHS /home/borisef/projects/cppflowATR/libs/shared/)
target_link_libraries (main_ATR ${MY_LIB} ${OpenCV_LIBS})

add_library(mbATR SHARED src/InterfaceATR.cpp)
find_package( OpenCV REQUIRED )
target_include_directories(mbATR PRIVATE include)
target_link_libraries (mbATR -ltensorflow ${OpenCV_LIBS})