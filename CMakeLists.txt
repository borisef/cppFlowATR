cmake_minimum_required(VERSION 3.10)
project(cppFlowATR)
set(CMAKE_CXX_STANDARD 17)

add_definitions(-DTEST_MODE)
add_compile_options(-pthread)

include_directories(PRIVATE include)

AUX_SOURCE_DIRECTORY(src SRC)

find_package( OpenCV REQUIRED )
set(OpenCV_STATIC ON)

#include_directories(PRIVATE include e:/Installs/opencv/sources/include e:/Installs/libtensorflow-gpu-windows-x86_64-1.15.0/include c:/Users/efrat/.conda/envs/keras-gpu/Library/include )
#set(OpenCV_DIR e:/Installs/opencv/build)
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} e:/Installs/opencv/build)
#find_package( OpenCV REQUIRED)
#find_package(OpenCV REQUIRED PATHS e:/Installs/opencv/build/x64/vc14/bin NO_DEFAULT_PATH)

FIND_LIBRARY(MY_LIB NAMES libtensorflow.so PATHS /home/magshim/cppflowATR/libs/shared/115 )
set(CMAKE_CXX_STANDARD 17)

# add_executable(main_like_coco tests/main_like_coco.cpp src/Model.cpp src/Tensor.cpp)
# target_link_libraries (main_like_coco -ltensorflow ${OpenCV_LIBS})

# add_executable(main_ATR tests/main_ATR.cpp src/Model.cpp 
#                          src/Tensor.cpp src/InterfaceATR.cpp 
#                          src/Object_Detection_API.cpp src/imgUtils.cpp)

# target_link_libraries (main_ATR -ltensorflow ${OpenCV_LIBS})

#add_executable(main_ATR src/main_ATR.cpp src/Model.cpp src/Tensor.cpp src/InterfaceATR.cpp src/Object_Detection_API.cpp src/imgUtils.cpp)
#find_package( OpenCV REQUIRED )
#target_include_directories(main_ATR PRIVATE include)
#target_link_libraries (main_ATR ${MY_LIB} ${OpenCV_LIBS})
#target_link_libraries (main_ATR -ltensorflow ${OpenCV_LIBS})

# add_library(mbATR SHARED src/InterfaceATR.cpp)
# #find_package( OpenCV REQUIRED )
# #target_include_directories(mbATR PRIVATE include)
# target_link_libraries (mbATR -ltensorflow ${OpenCV_LIBS})

#TEST
include_directories(PRIVATE include e:/Installs/opencv/build/include e:/Installs/opencv/sources/include e:/Installs/libtensorflow-gpu-windows-x86_64-1.15.0/include )
link_directories(e:/Installs/opencv/build/x64/vc14/lib e:/Installs/libtensorflow-gpu-windows-x86_64-1.15.0/lib)
#add_executable(testUtils tests/testUtils.cpp src/imgUtils.cpp)
#target_link_libraries (testUtils  opencv_world343d.lib)

# set(SOURCE main.cpp)
# add_executable(${PROJECT_NAME} ${SOURCE})

add_executable(testATRInference  tests/testATRinference.cpp src/Model.cpp src/Tensor.cpp 
                                    src/imgUtils.cpp src/Object_Detection_API.cpp src/InterfaceATR.cpp
                                    src/Object_Detection_Handler.cpp)





# test threads
add_executable(testThread  tests/testThread.cpp)

add_executable(videoTest  tests/videoTest.cpp ${SRC})





#target_link_libraries (testATRInference -ltensorflow ${OpenCV_LIBS})
target_link_libraries (videoTest ${MY_LIB} ${OpenCV_LIBS})

#SO 
add_library(mbATR SHARED    src/InterfaceATR.cpp  src/Object_Detection_API.cpp
                            src/Model.cpp src/Tensor.cpp    src/imgUtils.cpp
                            src/Object_Detection_Handler.cpp)
target_link_libraries (mbATR -ltensorflow ${OpenCV_LIBS})

