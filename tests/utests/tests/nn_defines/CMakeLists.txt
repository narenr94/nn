set(CMAKE_BUILD_TYPE Debug)

include(GoogleTest)

set(EXEC_NAME nn_defines_tests)

set(DEPS_INC "../../build/libs/include/")

set(NN_INC "../../../../include/")

set(NN_SRC "../../../../src/")

set(UTESTS_ROOT "../../")

include_directories(${NN_INC})
include_directories(${DEPS_INC})

set(TEST_SOURCES    nn_defines_tests_main.cpp
	nn_defines_tests.cpp)

set(NN_SOURCES ${NN_SRC}/nn_defines.cpp)

add_executable(${EXEC_NAME}	${FAKE_SOURCES} ${TEST_SOURCES}	${NN_SOURCES})

target_link_libraries(${EXEC_NAME} ${GMOCK_LINK_LIBRARIES} ${GMOCK_LINK_LIBRARIES})

nn_utest_run_add(${EXEC_NAME})


