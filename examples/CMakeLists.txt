# examples/CMakeLists.txt 
# Explicitly list source files with the correct relative path 
set(EXAMPLES ${CMAKE_SOURCE_DIR}/examples/mnist_example.cpp 
    ${CMAKE_SOURCE_DIR}/examples/mnist_copy_example.cpp 
    ${CMAKE_SOURCE_DIR}/examples/mnist_example_batch.cpp 
    ${CMAKE_SOURCE_DIR}/examples/mnist_actFunc_example.cpp ) 

set(EXECUTABLE_TARGETS)

# Create executables for each example 
foreach(EXAMPLE ${EXAMPLES}) 
    get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE})
    target_include_directories(${EXAMPLE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include) 
    target_link_libraries(${EXAMPLE_NAME} ${CMAKE_BINARY_DIR}/src/libnncore.so) 
    list(APPEND EXECUTABLE_TARGETS ${EXAMPLE_NAME})
endforeach()

# Add a custom target that depends on all executables 
add_custom_target( RunAfterBuild ALL 
    DEPENDS ${EXECUTABLE_TARGETS} 
    COMMENT "Running download.sh script after all executables are built" )

# Add a custom command to run the script after the custom target is built 
add_custom_command( TARGET RunAfterBuild POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E echo "Running download.sh" 
    COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_SOURCE_DIR}/examples/download_mnist.sh 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples 
    COMMENT "Running download.sh script" )
