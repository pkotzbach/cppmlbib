cmake_minimum_required(VERSION 3.22)

project(cppmlbib
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GTEST
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# CTEST
include(CTest)
enable_testing()

# CUDA
find_package(CUDAToolkit REQUIRED)

# Directories
set(SRC_DIR         ${CMAKE_SOURCE_DIR}/Source)
set(CUDA_SRC_DIR    ${SRC_DIR}/cuda)
set(TEST_DIR        ${CMAKE_SOURCE_DIR}/Tests)
set(EX_DIR          ${CMAKE_SOURCE_DIR}/Examples)
set(HEADER_DIR      ${CMAKE_SOURCE_DIR}/Headers)
set(CUDA_HEADER_DIR ${HEADER_DIR}/cuda)

# Sources
file(GLOB COMMON_SRCS ${SRC_DIR}/*.cpp)
file(GLOB CUDA_SRCS ${CUDA_SRC_DIR}/*.cu)
file(GLOB TEST_SRCS ${TEST_DIR}/*.cpp)

# Common object library
add_library(common_objs OBJECT
    ${COMMON_SRCS}
    ${CUDA_SRCS}
)

target_include_directories(common_objs PUBLIC
    ${HEADER_DIR}
    ${CUDA_HEADER_DIR}
)

target_link_libraries(common_objs PUBLIC
    CUDA::cudart
)

target_compile_options(common_objs PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-g -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-g>
)

set_target_properties(common_objs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

#### Executables
# Tests
add_executable(tests
    $<TARGET_OBJECTS:common_objs>
    ${TEST_SRCS}
)

target_include_directories(tests PRIVATE
    ${HEADER_DIR}
    ${CUDA_HEADER_DIR}
)

target_link_libraries(tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
)

set_target_properties(tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

include(GoogleTest)

gtest_discover_tests(tests
    DISCOVERY_MODE PRE_TEST
)

# Iris
add_executable(iris
    $<TARGET_OBJECTS:common_objs>
    ${EX_DIR}/iris.cpp
)

target_include_directories(iris PRIVATE
    ${HEADER_DIR}
    ${CUDA_HEADER_DIR}
)

set_target_properties(iris PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
