cmake_minimum_required(VERSION 3.15)
project(Pose_Estimation_2d2d)

set(CMAKE_CXX_STANDARD 14)

add_executable(Pose_Estimation_2d2d main.cpp)

# find opencv package
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Pose_Estimation_2d2d ${OpenCV_LIBS})