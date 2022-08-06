cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(FastReformer LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2 --extended-lambda")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(DCMAKE_BUILD_TYPE "Release")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/pybind11/include ${CUDA_TOOLKIT_ROOT_DIR}/include)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(CUDA)
find_package(CUDAToolkit)

add_subdirectory(pybind11)
add_subdirectory(reformer/inference)
add_subdirectory(tests/cpp)