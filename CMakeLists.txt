cmake_minimum_required(VERSION 3.10)
project(cppFlowATR)
set(CMAKE_CXX_STANDARD 17)

message($ENV{USER})

#DEFINE PATHS
# IF($ENV{USER} STREQUAL "magshim")

# elseif()



set(TEST_MODE TRUE)

set(NO_TRT TRUE)
set(CMAKE_MAJOR_4 FALSE)

if(TEST_MODE)
    add_definitions(-DTEST_MODE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTEST_MODE")#jic
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTEST_MODE")#jic
endif()

if( WIN32 OR MSVC OR MSYS OR MINGW OR (${CMAKE_SYSTEM_NAME} STREQUAL "Windows"))
    add_definitions(-DOS_WINDOWS)
    set(OS_WINDOWS TRUE)
    set(OS_LINUX FALSE)
    message("-------Windows-------")
elseif((UNIX OR LINUX OR (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")) AND NOT APPLE)
    add_definitions(-DOS_LINUX)
    set(OS_WINDOWS FALSE)
    set(OS_LINUX TRUE)
    message("-------Linux-------")
endif()


IF((${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm") OR (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM")  OR (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64") )
    add_definitions (-DJETSON)
    set(JETSON TRUE)
    message("-------Jetson-------")
ELSE()
    message("-------NOT Jetson-------")
    set(JETSON FALSE)
ENDIF()


macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()
#print_all_variables()



if(OS_WINDOWS)
    include_directories(PRIVATE include e:/Installs/opencv/build/include e:/Installs/opencv/sources/include e:/Installs/libtensorflow-gpu-windows-x86_64-1.15.0/include )
    link_directories(e:/Installs/opencv/build/x64/vc14/lib e:/Installs/libtensorflow-gpu-windows-x86_64-1.15.0/lib)
    set(MY_TF_LIB "tensorflow.dll")
    set(OpenCV_LIBS "opencv_world343d.lib")
    set(MY_TF_LIB2 "")
else(OS_WINDOWS) #Linux
    if(NOT JETSON)
        set (NO_TRT TRUE)
    endif()
    include_directories(PRIVATE include)
    set(OpenCV_DIR /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/cmake/opencv4)
    find_package(OpenCV REQUIRED)

    message("OpenCV_VERSION_MAJOR=" ${OpenCV_VERSION_MAJOR})
    IF((${OpenCV_VERSION_MAJOR} MATCHES "4"))
        set(CMAKE_MAJOR_4 TRUE)
        message ("Changes CMAKE_MAJOR_4 to" ${CMAKE_MAJOR_4})
    ENDIF()

    set(OpenCV_STATIC ON)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
    include_directories(${OpenCV_INCLUDE_DIRS})

    message(${OpenCV_INCLUDE_DIRS})

    message("locating cuda and TensorRT ...")
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

    # Sets variable to a value if variable is unset.
    macro(set_ifndef var val)
        if (NOT ${var})
            set(${var} ${val})
        endif()
        message(STATUS "Configurable variable ${var} set to ${${var}}")
    endmacro()

    # TensorRT's nvinfer and nvinfer_plugin lib
    ###########################################
    if(NO_TRT)
        set(NVINFER_LIB "_NVINFER_LIB-NOTFOUND")
        set(NVINFER_PLUGIN_LIB "_NVINFER_PLUGIN_LIB-NOTFOUND")
    else()
        set_ifndef(TRT_LIB /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
        set_ifndef(TRT_INCLUDE /usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)

        find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
        find_library(_NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
        set_ifndef(NVINFER_LIB ${_NVINFER_LIB})
        set_ifndef(NVINFER_PLUGIN_LIB ${_NVINFER_PLUGIN_LIB})
    endif()
    if((${NVINFER_LIB} STREQUAL "_NVINFER_LIB-NOTFOUND") OR (${NVINFER_PLUGIN_LIB} STREQUAL "_NVINFER_PLUGIN_LIB-NOTFOUND"))
        message(STATUS "Tensor-RT NOT FOUND: This version does not support .engiene models")
        add_definitions(-DNO_TRT)
        set(MY_CUDA_LIBS "")
    else()
        message(STATUS "Tensor-RT FOUND and will be supported")
        include_directories(${TRT_INCLUDE})
        set(MY_CUDA_LIBS ${CUDA_CUDART_LIBRARY} ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB})
    endif()

    if(JETSON) #Linux Jetson ONLY
        add_compile_options(-pthread -lm)
        include_directories(PRIVATE include tensorflow/include)
        FIND_LIBRARY(MY_TF_LIB NAMES libtensorflow_framework.so PATHS tensorflow/lib)
        FIND_LIBRARY(MY_TF_LIB2 NAMES libtensorflow.so PATHS tensorflow/lib)
    else(JETSON) # Linux PC ONLY
        include_directories(PRIVATE include tensorflow/include)
        FIND_LIBRARY(MY_TF_LIB NAMES libtensorflow.so PATHS tensorflow/lib)
        set(MY_TF_LIB2 "")
    endif(JETSON)
endif(OS_WINDOWS)

if(CMAKE_MAJOR_4)
    add_definitions (-DOPENCV_MAJOR_4)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DOPENCV_MAJOR_4")#jic
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_MAJOR_4")#jic
endif(CMAKE_MAJOR_4)


AUX_SOURCE_DIRECTORY(src SRC)



#add_executable(testUtils tests/testUtils.cpp src/imgUtils.cpp)
#target_link_libraries (testUtils  opencv_world343d.lib)

# set(SOURCE main.cpp)
# add_executable(${PROJECT_NAME} ${SOURCE})

# add_executable(testATRInference  tests/testATRinference.cpp src/Model.cpp src/Tensor.cpp 
#                                      src/imgUtils.cpp src/Object_Detection_API.cpp src/InterfaceATR.cpp src/Object_Detection_Handler.cpp)

# #target_link_libraries (testATRInference ${MY_LIB})
# target_link_libraries (testATRInference opencv_world343d.lib tensorflow.dll )


add_executable(stressTest tests/testStress.cpp ${SRC})
target_link_libraries (stressTest  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread )


add_executable(testTileImages tests/testTileImages.cpp ${SRC})
target_link_libraries (testTileImages  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

add_executable(testLikeMnist tests/testLikeMnist.cpp ${SRC})
target_link_libraries (testLikeMnist  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

add_executable(testColorNet tests/testColorNet.cpp ${SRC})
target_link_libraries (testColorNet ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

add_executable(jsonTest tests/jsonTest.cpp ${SRC})
target_link_libraries (jsonTest ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

add_executable(testIniFromJson tests/testIniFromJson.cpp ${SRC})
target_link_libraries (testIniFromJson ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

add_executable(logtest tests/logtest.cpp ${SRC})
target_link_libraries (logtest  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

add_executable(videoTest tests/testvideo.cpp ${SRC})
target_link_libraries (videoTest  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

add_executable(testCmake tests/testCmake.cpp ${SRC})
target_link_libraries (testCmake  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

#SO
add_library(mbATR SHARED   ${SRC})
target_link_libraries (mbATR ${MY_TF_LIB} ${MY_TF_LIB2} ${OpenCV_LIBS} -lpthread)

# #(3) Test stressTest but with SO dependencies
add_executable(testTamir tests/testStress.cpp  ${SRC})
FIND_LIBRARY(MY_LIB1 NAMES libmbATR.so PATHS build/
                                             build/libs/shared/ATR)
target_link_libraries (testTamir  ${MY_TF_LIB} ${MY_TF_LIB2} ${OpenCV_LIBS} ${MY_CUDA_LIBS} -lpthread)

#TEMP
add_executable(stressTestSmall tests/testStressSmall.cpp ${SRC})
target_link_libraries (stressTestSmall ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

add_executable(stressTestSmall_NV12 tests/testStressSmall_NV12.cpp ${SRC})
target_link_libraries (stressTestSmall_NV12 ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)


#testMultiModels
add_executable(testMultiModels tests/testMultiModels.cpp ${SRC})
target_link_libraries (testMultiModels ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)


#NV12toTIFF
add_executable(NV12toTIFF tests/NV12toTIFF.cpp ${SRC})
target_link_libraries (NV12toTIFF ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)


#testOnFolderRecurse
add_executable(testOnFolderRecurse tests/testOnFolderRecurse.cpp ${SRC})
target_link_libraries (testOnFolderRecurse ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)

# add_executable(sizeMatching tests/sizeMatching_test.cpp ${SRC})
# target_link_libraries (sizeMatching ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

#testSizeFilter
add_executable(testSizeFilter tests/testSizeFilter.cpp ${SRC})
target_link_libraries (testSizeFilter ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} ${MY_CUDA_LIBS} -lpthread)
