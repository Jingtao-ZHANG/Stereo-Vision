cmake_minimum_required(VERSION 3.15)
project(ORB_feature_extraction)

set(CMAKE_CXX_STANDARD 14)

add_executable(ORB_feature_extraction main.cpp)

# find opencv package
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ORB_feature_extraction ${OpenCV_LIBS})