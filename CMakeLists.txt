cmake_minimum_required(VERSION 3.10)
project(cppFlowATR)

#add_compile_options(-pthread -lpthread)

add_definitions(-DTEST_MODE) # windows

set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -DTEST_MODE) #linux not working 
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DTEST_MODE)#linux not working 
add_compile_options(-DTEST_MODE)#linux not working 

include_directories(PRIVATE include)

add_definitions(-DTEST_MODE)

find_package( OpenCV REQUIRED )
set(OpenCV_STATIC ON)
AUX_SOURCE_DIRECTORY(src SRC)


link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${OpenCV_INCLUDE_DIRS})

FIND_LIBRARY(MY_TF_LIB NAMES libtensorflow.so PATHS /home/borisef/projects/cppflowATR/libs/shared/115 )
set(CMAKE_CXX_STANDARD 17)

# add_executable(main_like_coco tests/main_like_coco.cpp src/Model.cpp src/Tensor.cpp)
# target_link_libraries (main_like_coco -ltensorflow ${OpenCV_LIBS})

# add_executable(main_ATR tests/main_ATR.cpp src/Model.cpp 
#                          src/Tensor.cpp src/InterfaceATR.cpp 
#                          src/Object_Detection_API.cpp src/imgUtils.cpp)

# target_link_libraries (main_ATR -ltensorflow ${OpenCV_LIBS})



#TESTS
#(1)test Utils
#add_executable(testUtils tests/testUtils.cpp src/imgUtils.cpp)
#target_link_libraries (testUtils  ${OpenCV_LIBS})

#(2) Test ATR with API Inference
add_executable(testATRInference  tests/testATRinference.cpp src/Model.cpp src/Tensor.cpp 
                                     src/imgUtils.cpp src/Object_Detection_API.cpp src/InterfaceATR.cpp
                                     src/Object_Detection_Handler.cpp)

# #target_link_libraries (testATRInference -ltensorflow ${OpenCV_LIBS})
target_link_libraries (testATRInference ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)


#(3) Test testATRInference but with SO dependencies
# add_executable(testTamir tests/testATRinference.cpp)
# FIND_LIBRARY(MY_LIB1 NAMES libmbATR.so PATHS /home/borisef/projects/cppflowATR/build )
# target_link_libraries (testTamir  ${MY_TF_LIB} ${OpenCV_LIBS} ${MY_LIB1})


# (4) test threads
#add_executable(testThread  tests/testThread.cpp)
#target_link_libraries(testThread -lpthread)

#(5) test video 
add_executable(videoTest  tests/videoTest.cpp ${SRC})
target_link_libraries (videoTest ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)





#target_link_libraries (testATRInference -ltensorflow ${OpenCV_LIBS})
target_link_libraries (videoTest ${MY_LIB} ${OpenCV_LIBS})





#SO 
# add_library(mbATR SHARED    src/InterfaceATR.cpp  src/Object_Detection_API.cpp
#                             src/Model.cpp src/Tensor.cpp    src/imgUtils.cpp
#                             src/Object_Detection_Handler.cpp)
# target_link_libraries (mbATR -ltensorflow ${OpenCV_LIBS})

