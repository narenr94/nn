#main CmakeLists
cmake_minimum_required(VERSION 3.10) 
project(NNProject LANGUAGES CXX)

#to build use
#mkdir build
#cd build
#cmake ..
#cmake --build .

# Set policy CMP0079 to NEW 
#cmake_policy(SET CMP0079 NEW)
# Add include directory 
include_directories(${CMAKE_SOURCE_DIR}/include)
# Add subdirectories 
add_subdirectory(utils)
add_subdirectory(src)
add_subdirectory(tests)
