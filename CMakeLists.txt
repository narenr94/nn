#main CmakeLists
cmake_minimum_required(VERSION 3.10) 
project(NNProject LANGUAGES CXX)
# Set policy CMP0079 to NEW 
#cmake_policy(SET CMP0079 NEW)
# Add include directory 
include_directories(${CMAKE_SOURCE_DIR}/include)
# Add subdirectories 
add_subdirectory(src)