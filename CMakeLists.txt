cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(ZEPP LANGUAGES C CXX CUDA)

# Target: NVIDIA A100 (sm80, 108 SMs), bf16, NVSHMEM over libfabric/CXI.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_CUDA_ARCHITECTURES 80)
set(SM_CORES 108)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
  message(FATAL_ERROR "CUDA >= 12.0 is required (found ${CUDAToolkit_VERSION})")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\"")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG -Xcompiler -Wno-psabi -Xcompiler=-fno-strict-aliasing")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUTLASS_DEBUG_TRACE_LEVEL=0 --expt-extended-lambda --expt-relaxed-constexpr --std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- PyTorch (the ops are torch tensors in, torch tensors out) ----
find_package(Python 3 REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)
set(ENV{TORCH_CUDA_ARCH_LIST} "8.0")
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import os, torch; print(os.path.dirname(torch.__file__), end='')"
  RESULT_VARIABLE _PY_OK OUTPUT_VARIABLE TORCH_DIR)
if(NOT _PY_OK MATCHES 0)
  message(FATAL_ERROR "python cannot import torch (source env/setup.sh first)")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
if(TORCH_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")
endif()
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; _ = torch.float8_e4m3fn"
  RESULT_VARIABLE TORCH_FP8_STATUS OUTPUT_QUIET ERROR_QUIET)
if(TORCH_FP8_STATUS EQUAL 0)
  add_compile_options("-DTORCH_SUPPORT_FP8")
endif()

# ---- NVSHMEM (symmetric heap + one-sided puts) ----
if(NOT NVSHMEM_HOME)
  set(NVSHMEM_HOME $ENV{NVSHMEM_HOME})
endif()
if(NOT EXISTS "${NVSHMEM_HOME}/include/nvshmem.h")
  message(FATAL_ERROR "NVSHMEM_HOME (${NVSHMEM_HOME}) does not contain include/nvshmem.h")
endif()
add_definitions(-DFLUX_SHM_USE_NVSHMEM)

set(COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${NVSHMEM_HOME}/include)
foreach(inc_dir ${COMMON_HEADER_DIRS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${inc_dir}")
endforeach()
include_directories(${COMMON_HEADER_DIRS})
link_directories(${CUDAToolkit_LIBRARY_DIR} ${NVSHMEM_HOME}/lib)

add_subdirectory(src)
