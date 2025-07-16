set(CMAKE_BUILD_TYPE Debug)

include(GoogleTest)

set(EXEC_NAME ConvLayer_tests)

set(DEPS_INC "../../build/libs/include/")

set(NN_INC "../../../../include/")

set(NN_SRC "../../../../src/")

set(UTESTS_ROOT "../../")

include_directories(${NN_INC})
include_directories(${DEPS_INC})

set(TEST_SOURCES    convLayer_tests_main.cpp
	convLayer_tests.cpp)

set(NN_SOURCES ${NN_SRC}/Layers/convLayer.cpp
	${NN_SRC}/Layers/baseLayer.cpp
	${NN_SRC}/Layers/inputLayer.cpp
	${NN_SRC}/nn_defines.cpp)

set(FAKE_SOURCES ${UTESTS_ROOT}/fakes/fake_sigmoidActFunc.cpp
${UTESTS_ROOT}/fakes/fake_leakyReluActFunc.cpp
${UTESTS_ROOT}/fakes/fake_reluActFunc.cpp
${UTESTS_ROOT}/fakes/fake_softmaxActFunc.cpp
${UTESTS_ROOT}/fakes/fake_tanhActFunc.cpp
${UTESTS_ROOT}/fakes/fake_cpuAccelerator.cpp
${UTESTS_ROOT}/fakes/fake_nn_math.cpp)

add_executable(${EXEC_NAME}	${FAKE_SOURCES} ${TEST_SOURCES}	${NN_SOURCES})

target_link_libraries(${EXEC_NAME} ${GMOCK_LINK_LIBRARIES} ${GMOCK_LINK_LIBRARIES})

nn_utest_run_add(${EXEC_NAME})


