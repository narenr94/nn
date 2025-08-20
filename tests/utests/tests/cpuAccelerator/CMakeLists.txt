set(CMAKE_BUILD_TYPE Debug)

include(GoogleTest)

set(EXEC_NAME cpuAccelerator_tests)

set(DEPS_INC "../../build/libs/include/")

set(NN_INC "../../../../include/")

set(NN_SRC "../../../../src/")

set(UTESTS_ROOT "../../")

include_directories(${NN_INC})
include_directories(${DEPS_INC})

set(TEST_SOURCES cpuAccelerator_tests_main.cpp
cpuAccelerator_forwardpass_tests.cpp
cpuAccelerator_outLayer_backwardpass_tests.cpp
cpuAccelerator_backwardpass_tests.cpp)

set(NN_SOURCES ${NN_SRC}/Accelerators/cpuAccelerator.cpp
${NN_SRC}/Layers/baseLayer.cpp
${NN_SRC}/Layers/denseLayer.cpp
${NN_SRC}/Layers/convLayer.cpp
${NN_SRC}/Layers/inputLayer.cpp
${NN_SRC}/Layers/poolingLayer.cpp
${NN_SRC}/nn_math.cpp
${NN_SRC}/nn_defines.cpp)

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

#todo : selectively apply for tests that requires below
add_definitions(-DIGNORE_LAYER_INPUT_VALIDATION_FOR_TESTING)

add_executable(${EXEC_NAME} ${TEST_SOURCES} ${LOSS_SOURCES} ${ACT_SOURCES} ${NN_SOURCES})

target_link_libraries(${EXEC_NAME} ${GMOCK_LINK_LIBRARIES} ${GMOCK_LINK_LIBRARIES})

nn_utest_run_add(${EXEC_NAME})


