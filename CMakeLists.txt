cmake_minimum_required(VERSION 3.10)
project(cppFlowATR)
set(CMAKE_CXX_STANDARD 17)



set(TEST_MODE TRUE)
if(TEST_MODE)
    add_definitions(-DTEST_MODE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTEST_MODE")#jic
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTEST_MODE")#jic
endif()

if( WIN32 OR MSVC OR MSYS OR MINGW OR (${CMAKE_SYSTEM_NAME} STREQUAL "Windows"))
    add_definitions(-DOS_WINDOWS)
    set(OS_WINDOWS TRUE)
    set(OS_LINUX FALSE)
elseif((UNIX OR LINUX OR (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")) AND NOT APPLE)
    add_definitions(-DOS_LINUX)
    set(OS_WINDOWS FALSE)
    set(OS_LINUX TRUE)
endif()

IF((${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm") OR (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM") )
    add_definitions (-DJETSON)
    set(JETSON TRUE)
ELSE()
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
else(OS_WINDOWS) #Linux
    find_package( OpenCV REQUIRED )
    set(OpenCV_STATIC ON)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
    include_directories(${OpenCV_INCLUDE_DIRS})

    if(JETSON) #Linux Jetson

    else(JETSON) # Linux PC
        FIND_LIBRARY(MY_TF_LIB NAMES libtensorflow.so PATHS /home/borisef/projects/cppflowATR/libs/shared/115 )


    endif(JETSON)
endif(OS_WINDOWS)

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
target_link_libraries (stressTest opencv_world343d.lib tensorflow.dll -lpthread)


# # test threads
# add_executable(testThread  tests/testThread.cpp)

 add_executable(testTileImages tests/testTileImages.cpp ${SRC})
 target_link_libraries (testTileImages  opencv_world343d.lib tensorflow.dll -lpthread)


add_executable(testLikeMnist tests/testLikeMnist.cpp ${SRC})
target_link_libraries (testLikeMnist  ${OpenCV_LIBS} ${MY_TF_LIB} -lpthread)

add_executable(testColorNet tests/testColorNet.cpp ${SRC})
target_link_libraries (testColorNet ${OpenCV_LIBS} ${MY_TF_LIB} -lpthread)

add_executable(jsonTest tests/jsonTest.cpp ${SRC})
target_link_libraries (jsonTest ${OpenCV_LIBS} ${MY_TF_LIB} -lpthread)

add_executable(testIniFromJson tests/testIniFromJson.cpp ${SRC})
target_link_libraries (testIniFromJson ${OpenCV_LIBS} ${MY_TF_LIB} -lpthread)

add_executable(testlog tests/logtest.cpp ${SRC})
target_link_libraries (testlog  ${OpenCV_LIBS} ${MY_TF_LIB} -lpthread)

add_executable(videoTest tests/testvideo.cpp ${SRC})
target_link_libraries (videoTest  ${OpenCV_LIBS} ${MY_TF_LIB} -lpthread)

add_executable(testCmake tests/testCmake.cpp ${SRC})
target_link_libraries (testCmake   ${OpenCV_LIBS} ${MY_TF_LIB} -lpthread)