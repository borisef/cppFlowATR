cmake_minimum_required(VERSION 3.10)
project(cppFlowATR)
set(CMAKE_CXX_STANDARD 17)

message($ENV{USER})

#DEFINE PATHS
# IF($ENV{USER} STREQUAL "magshim")

# elseif()



set(TEST_MODE TRUE)
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
    include_directories(PRIVATE include)
    find_package( OpenCV REQUIRED)
    
    message("OpenCV_VERSION_MAJOR=" ${OpenCV_VERSION_MAJOR})
    IF((${OpenCV_VERSION_MAJOR} MATCHES "4"))
        set(CMAKE_MAJOR_4 TRUE)
        message ("Changes CMAKE_MAJOR_4 to" ${CMAKE_MAJOR_4})
    ENDIF()

    set(OpenCV_STATIC ON)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
    include_directories(${OpenCV_INCLUDE_DIRS})

    message(${OpenCV_INCLUDE_DIRS})

    if(JETSON) #Linux Jetson ONLY
        add_compile_options(-pthread -lm)
        include_directories(PRIVATE include /media/borisef/nvmeSSD/tf_1_13_source)
	FIND_LIBRARY(MY_TF_LIB NAMES libtensorflow_framework.so PATHS /media/borisef/nvmeSSD/tensorflow_Jetson )
        FIND_LIBRARY(MY_TF_LIB2 NAMES libtensorflow.so PATHS /media/borisef/nvmeSSD/tensorflow_Jetson )
    else(JETSON) # Linux PC ONLY
        FIND_LIBRARY(MY_TF_LIB NAMES libtensorflow.so PATHS /home/magshim/projects/cppflowATR/libs/shared/113 )
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
target_link_libraries (stressTest  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread )


add_executable(testTileImages tests/testTileImages.cpp ${SRC})
target_link_libraries (testTileImages  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

add_executable(testLikeMnist tests/testLikeMnist.cpp ${SRC})
target_link_libraries (testLikeMnist  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

add_executable(testColorNet tests/testColorNet.cpp ${SRC})
target_link_libraries (testColorNet ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

add_executable(jsonTest tests/jsonTest.cpp ${SRC})
target_link_libraries (jsonTest ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

add_executable(testIniFromJson tests/testIniFromJson.cpp ${SRC})
target_link_libraries (testIniFromJson ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

add_executable(logtest tests/logtest.cpp ${SRC})
target_link_libraries (logtest  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

add_executable(videoTest tests/testvideo.cpp ${SRC})
target_link_libraries (videoTest  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

add_executable(testCmake tests/testCmake.cpp ${SRC})
target_link_libraries (testCmake  ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

#SO 
add_library(mbATR SHARED   ${SRC})
target_link_libraries (mbATR ${MY_TF_LIB} ${MY_TF_LIB2} ${OpenCV_LIBS} -lpthread)

# #(3) Test stressTest but with SO dependencies
add_executable(testTamir tests/testStress.cpp  ${SRC})
FIND_LIBRARY(MY_LIB1 NAMES libmbATR.so PATHS build/
                                             build/libs/shared/ATR)               
target_link_libraries (testTamir  ${MY_TF_LIB} ${MY_TF_LIB2} ${OpenCV_LIBS} -lpthread)

#TEMP
add_executable(stressTestSmall tests/testStressSmall.cpp ${SRC})
target_link_libraries (stressTestSmall ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)

add_executable(stressTestSmall_NV12 tests/testStressSmall_NV12.cpp ${SRC})
target_link_libraries (stressTestSmall_NV12 ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)


#testMultiModels
add_executable(testMultiModels tests/testMultiModels.cpp ${SRC})
target_link_libraries (testMultiModels ${OpenCV_LIBS} ${MY_TF_LIB}  ${MY_TF_LIB2} -lpthread)