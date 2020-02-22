cmake_minimum_required(VERSION 3.10)
project(cppFlowATR)
set(CMAKE_CXX_STANDARD 17)

AUX_SOURCE_DIRECTORY(src SRC)

add_definitions(-DTEST_MODE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTEST_MODE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTEST_MODE")





#include_directories(PRIVATE include e:/Installs/opencv/sources/include e:/Installs/libtensorflow-gpu-windows-x86_64-1.15.0/include c:/Users/efrat/.conda/envs/keras-gpu/Library/include )
#set(OpenCV_DIR e:/Installs/opencv/build)
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} e:/Installs/opencv/build)
#find_package( OpenCV REQUIRED)
#find_package(OpenCV REQUIRED PATHS e:/Installs/opencv/build/x64/vc14/bin NO_DEFAULT_PATH)

 # For some reason CMake uses its defaults before the above paths.)

#FIND_LIBRARY(MY_LIBS NAMES tensorflow.lib opencv_world343.lib opencv_world343d.lib PATHS e:/Installs/libtensorflow-gpu-windows-x86_64-1.15.0/lib
#e:/Installs/opencv/build/x64/vc14/lib)


# add_executable(main_like_coco src/main_like_coco.cpp src/Model.cpp src/Tensor.cpp)
# #find_package( OpenCV REQUIRED )
# target_include_directories(main_like_coco PRIVATE include)
# target_link_libraries (main_like_coco  ${MY_LIB} ${OpenCV_LIBS})

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

# add_executable(testATRInference  tests/testATRinference.cpp src/Model.cpp src/Tensor.cpp 
#                                      src/imgUtils.cpp src/Object_Detection_API.cpp src/InterfaceATR.cpp src/Object_Detection_Handler.cpp)

# #target_link_libraries (testATRInference ${MY_LIB})
# target_link_libraries (testATRInference opencv_world343d.lib tensorflow.dll )


add_executable(stressTest tests/testStress.cpp ${SRC})
target_link_libraries (stressTest opencv_world343d.lib tensorflow.dll )


# # test threads
# add_executable(testThread  tests/testThread.cpp)

# add_executable(testTileImages tests/testTileImages.cpp ${SRC})
# target_link_libraries (testTileImages  opencv_world343d.lib tensorflow.dll)


add_executable(testLikeMnist tests/testLikeMnist.cpp ${SRC})
target_link_libraries (testLikeMnist  opencv_world343d.lib  tensorflow.dll)


add_executable(testColorNet tests/testColorNet.cpp ${SRC})
target_link_libraries (testColorNet opencv_world343d.lib  tensorflow.dll)

add_executable(jsonTest tests/jsonTest.cpp ${SRC})
target_link_libraries (jsonTest opencv_world343d.lib  tensorflow.dll)

add_executable(testIniFromJson tests/testIniFromJson.cpp ${SRC})
target_link_libraries (testIniFromJson opencv_world343d.lib  tensorflow.dll)

add_executable(testlog tests/logtest.cpp ${SRC})
target_link_libraries (testlog opencv_world343d.lib  tensorflow.dll)