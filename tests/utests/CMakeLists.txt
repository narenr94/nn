cmake_minimum_required(VERSION 3.10.3)

project(UnitTests)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GTEST REQUIRED gtest)
pkg_check_modules(GMOCK REQUIRED gmock)

link_directories(${GTEST_LIBRARY_DIRS} ${GMOCK_LIBRARY_DIRS})

add_subdirectory(tests)