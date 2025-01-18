# examples/CMakeLists.txt 
# Explicitly list source files with the correct relative path 
set(TESTS mnist_basic_test.cpp 
    mnist_copy_test.cpp 
    mnist_batch_test.cpp 
    mnist_actFunc_test.cpp
    mnist_save_load_test.cpp
    mnist_regLossFunc_test.cpp
    mnist_bce_test.cpp) 

set(EXECUTABLE_TARGETS)

# Create executables for each example 
foreach(TEST ${TESTS}) 
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    add_executable(${TEST_NAME} ${TEST})
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/utils) 
    target_link_libraries(${TEST_NAME} ${CMAKE_BINARY_DIR}/src/libnncore.so ${CMAKE_BINARY_DIR}/utils/libnnutils.so) 
    list(APPEND EXECUTABLE_TARGETS ${TEST_NAME})
endforeach()

#Add a custom target that depends on all executables 
add_custom_target( RunAfterBuild ALL 
    DEPENDS ${EXECUTABLE_TARGETS} 
    COMMENT "Running download.sh script after all executables are built" )

# Add a custom command to run the script after the custom target is built 
add_custom_command( TARGET RunAfterBuild POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E echo "Copying download.sh to build/tests" 
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/tests/download_mnist.sh ${CMAKE_BINARY_DIR}/tests/
    COMMAND ${CMAKE_COMMAND} -E echo "Running download.sh" 
    COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_SOURCE_DIR}/tests/download_mnist.sh 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests 
    COMMENT "Running download.sh script" )
