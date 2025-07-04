set(CMAKE_BUILD_TYPE Debug)

include(GoogleTest)

set(EXEC_NAME nn_core_tests)

set(DEPS_INC "../../build/libs/include/")

set(NN_INC "../../../../include/")

set(NN_SRC "../../../../src/")

set(UTESTS_ROOT "../../")

include_directories(${NN_INC})
include_directories(${DEPS_INC})

set(TEST_SOURCES    nn_core_tests_main.cpp
	nn_core_tests.cpp)

set(NN_SOURCES ${NN_SRC}/Accelerators/cpuAccelerator.cpp
${NN_SRC}/Layers/denseLayer.cpp
${NN_SRC}/Layers/convLayer.cpp
${NN_SRC}/Layers/inputLayer.cpp
${NN_SRC}/Layers/baseLayer.cpp
${NN_SRC}/nn_math.cpp
${NN_SRC}/nn_core.cpp)

set(ACT_SOURCES ${NN_SRC}/ActivationFunctions/sigmoidActFunc.cpp
${NN_SRC}/ActivationFunctions/leakyReluActFunc.cpp
${NN_SRC}/ActivationFunctions/reluActFunc.cpp
${NN_SRC}/ActivationFunctions/softmaxActFunc.cpp
${NN_SRC}/ActivationFunctions/tanhActFunc.cpp
)

set(LOSS_SOURCES ${NN_SRC}/LossFunctions/binaryCrossEntropyLoss.cpp
${NN_SRC}/LossFunctions/competitiveCrossEntropyLoss.cpp
${NN_SRC}/LossFunctions/huberLoss.cpp
${NN_SRC}/LossFunctions/meanAbsoluteError.cpp
${NN_SRC}/LossFunctions/meanSquaredError.cpp
)

set(OPT_SOURCES ${NN_SRC}/Optimizers/adam.cpp
${NN_SRC}/Optimizers/rmsprop.cpp
${NN_SRC}/Optimizers/stochasticGradientDescent.cpp)

add_executable(${EXEC_NAME}	${FAKE_SOURCES} ${LOSS_SOURCES} ${ACT_SOURCES} ${OPT_SOURCES} ${TEST_SOURCES} ${NN_SOURCES})

target_link_libraries(${EXEC_NAME} ${GMOCK_LINK_LIBRARIES} ${GMOCK_LINK_LIBRARIES})

nn_utest_run_add(${EXEC_NAME})


