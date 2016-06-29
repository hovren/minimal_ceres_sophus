cmake_minimum_required(VERSION 3.4)
project(minimal_sophus_ceres)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp circularconstraint.h)
add_executable(minimal_sophus_ceres ${SOURCE_FILES})
target_link_libraries(minimal_sophus_ceres ${CERES_LIBRARIES} gflags) # ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES})

add_executable(probe_gradient probe_gradient.cpp)
target_link_libraries(probe_gradient ${CERES_LIBRARIES} gflags)