# Source files in src 
set(SRC_FILES nn_core.cpp nn_math.cpp nn_defines.cpp)
set(LYR_FILES Layers/baseLayer.cpp
    Layers/denseLayer.cpp
    Layers/convLayer.cpp
    Layers/inputLayer.cpp)
set(ACT_FUNC_FILES ActivationFunctions/leakyReluActFunc.cpp 
    ActivationFunctions/reluActFunc.cpp 
    ActivationFunctions/sigmoidActFunc.cpp 
    ActivationFunctions/tanhActFunc.cpp
    ActivationFunctions/softmaxActFunc.cpp)
set(OPT_FUNC_FILES Optimizers/baseOptimizer.cpp
    Optimizers/adam.cpp 
    Optimizers/rmsprop.cpp 
    Optimizers/stochasticGradientDescent.cpp ) 
set(LOSS_FUNC_FILES LossFunctions/huberLoss.cpp
    LossFunctions/meanAbsoluteError.cpp
    LossFunctions/meanSquaredError.cpp
    LossFunctions/competitiveCrossEntropyLoss.cpp
    LossFunctions/binaryCrossEntropyLoss.cpp)
set(ACC_FUNC_FILES Accelerators/cpuAccelerator.cpp)

option(OPENCL_ACC "Enable this feature" OFF)

# Check if the option is enabled
# cmake -DOPENCL_ACC=ON ..
if(OPENCL_ACC)
    add_definitions(-DOPENCL_ACC)
    find_package(OpenCL REQUIRED)
    set(ACC_FUNC_FILES ${ACC_FUNC_FILES} 
    Accelerators/openclAccelerator.cpp)
endif()
 
# Create the main library from src 
add_library(nncore SHARED ${SRC_FILES} ${ACT_FUNC_FILES} ${LYR_FILES} ${OPT_FUNC_FILES} ${LOSS_FUNC_FILES} ${ACC_FUNC_FILES})
target_include_directories(nncore PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(OPENCL_ACC)
    target_link_libraries(nncore OpenCL::OpenCL)
endif()