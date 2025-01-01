# Source files in src 
set(SRC_FILES nn_core.cpp nn_l2l_weight_matrix.cpp 
    nn_layer.cpp nn_math.cpp nn_node.cpp nn_utils.cpp)
set(ACT_FUNC_FILES ActivationFunctions/activationFunction.cpp 
    ActivationFunctions/leakyReluActFunc.cpp 
    ActivationFunctions/reluActFunc.cpp 
    ActivationFunctions/sigmoidActFunc.cpp 
    ActivationFunctions/tanhActFunc.cpp )
set(OPT_FUNC_FILES Optimizers/optimizer.cpp 
    Optimizers/adam.cpp 
    Optimizers/rmsprop.cpp 
    Optimizers/stochasticGradientDescent.cpp ) 
 
# Create the main library from src 
add_library(nncore SHARED ${SRC_FILES} ${ACT_FUNC_FILES} ${OPT_FUNC_FILES})
target_include_directories(nncore PUBLIC ${CMAKE_SOURCE_DIR}/include)