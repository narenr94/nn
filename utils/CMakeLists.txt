set(UTILS_FILES nn_utils.cpp)
add_library(nnutils SHARED ${UTILS_FILES})
target_include_directories(nnutils PUBLIC ${CMAKE_SOURCE_DIR}/utils)