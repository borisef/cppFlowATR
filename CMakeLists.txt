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
add_executable(testATRInference  tests/testATRinference.cpp ${SRC})
target_link_libraries (testATRInference ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)



# #SO 
add_library(mbATR SHARED   ${SRC})
target_link_libraries (mbATR ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)
# # file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/build/libmbATR.so
# #      DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/shared/ATR/libmbATR.so)




# #(3) Test testATRInference but with SO dependencies
# add_executable(testTamir tests/testATRinference.cpp)
# FIND_LIBRARY(MY_LIB1 NAMES libmbATR.so PATHS /home/borisef/projects/cppFlowATR/build
#                                               /home/borisef/projects/cppflowATR/libs/shared/ATR)
# target_link_libraries (testTamir  ${MY_TF_LIB} ${OpenCV_LIBS} ${MY_LIB1}  -lpthread)


# # (4) test threads
# # (4) test threads
# add_executable(testThread  tests/testThread.cpp)
# target_link_libraries(testThread -lpthread)

# #(5) test video 
add_executable(testvideo  tests/testvideo.cpp ${SRC})
target_link_libraries (testvideo ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)


# add_executable(test_itay  tests/test_itay.cpp ${SRC})
# target_link_libraries (test_itay ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)



# target_link_libraries (videoTest ${MY_LIB} ${OpenCV_LIBS})


#(6) test tiles
add_executable(testTileImages  tests/testTileImages.cpp ${SRC})
target_link_libraries (testTileImages ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)


#stress test 

add_executable(stressTest tests/testStress.cpp ${SRC})
target_link_libraries (stressTest ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)

add_executable(convertYUVtoRGB_test  tests/convertYUVtoRGB_test.cpp ${SRC})
target_link_libraries (convertYUVtoRGB_test ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)

#color 
add_executable(testColorNet  tests/testColorNet.cpp ${SRC})
target_link_libraries (testColorNet ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)

#yuv2rgb
add_executable(yuv2rgb  tests/convertYUVtoRGB_test.cpp ${SRC})
target_link_libraries (yuv2rgb ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)

#ini
add_executable(jsonTest  tests/jsonTest.cpp ${SRC})
target_link_libraries (jsonTest ${MY_TF_LIB} ${OpenCV_LIBS} -lpthread)
