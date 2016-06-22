cmake_minimum_required(VERSION 3.5)
project(minimal_sophus_ceres)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(minimal_sophus_ceres ${SOURCE_FILES})
target_link_libraries(minimal_sophus_ceres ${CERES_LIBRARIES})