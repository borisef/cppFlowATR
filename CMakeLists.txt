cmake_minimum_required(VERSION 3.10)
project(cppFlowATR)

add_compile_options(-pthread)

include_directories(PRIVATE include)


find_package( OpenCV REQUIRED )
set(OpenCV_STATIC ON)

link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${OpenCV_INCLUDE_DIRS})

#FIND_LIBRARY(MY_LIB NAMES libstam.so PATHS /home/borisef/projects/cppflowATR/libs/shared/)
set(CMAKE_CXX_STANDARD 17)

add_executable(main_like_coco tests/main_like_coco.cpp src/Model.cpp src/Tensor.cpp)
target_link_libraries (main_like_coco -ltensorflow ${OpenCV_LIBS})

add_executable(main_ATR tests/main_ATR.cpp src/Model.cpp 
                         src/Tensor.cpp src/InterfaceATR.cpp 
                         src/Object_Detection_API.cpp src/imgUtils.cpp)

target_link_libraries (main_ATR -ltensorflow ${OpenCV_LIBS})

#SO 
add_library(mbATR SHARED src/InterfaceATR.cpp  src/Object_Detection_API.cpp)
target_link_libraries (mbATR -ltensorflow ${OpenCV_LIBS})



#TEST
add_executable(testUtils tests/testUtils.cpp src/imgUtils.cpp)
target_link_libraries (testUtils  ${OpenCV_LIBS})



add_executable(testATRInference  tests/testATRinference.cpp src/Model.cpp src/Tensor.cpp 
                                    src/imgUtils.cpp src/Object_Detection_API.cpp src/InterfaceATR.cpp)
target_link_libraries (testATRInference -ltensorflow ${OpenCV_LIBS})