cmake_minimum_required(VERSION 3.10)
project(DropletAnalyzer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
set(OpenCV_DIR "C:/OpenCV/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(DropletAnalyzer main.cpp)

# Link OpenCV
target_link_libraries(DropletAnalyzer ${OpenCV_LIBS})