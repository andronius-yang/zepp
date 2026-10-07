cmake_minimum_required(VERSION 3.17)
project(ZEPP_GENERATOR LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
find_package(CUDAToolkit REQUIRED)
set(ROOT_DIR ${PROJECT_SOURCE_DIR}/../../)
include_directories(
  ${CUDAToolkit_INCLUDE_DIRS}
  ${ROOT_DIR}/include
  ${ROOT_DIR}/3rdparty/cutlass/include
  ${ROOT_DIR}/3rdparty/cutlass/tools/util/include)
add_executable(gen_dispatch gen_dispatch.cc)
add_executable(gen_combine gen_combine.cc)
add_executable(gen_gemm gen_gemm.cc)
