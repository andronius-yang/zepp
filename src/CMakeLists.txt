cmake_minimum_required(VERSION 3.17)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# one object library per kernel directory
function(zepp_add_cu_obj_lib TARGET FILES)
  add_library(${TARGET} OBJECT ${FILES})
  set_target_properties(${TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(${TARGET} PUBLIC CUDA::cudart ${CMAKE_THREAD_LIBS_INIT})
endfunction()

# kernel-space generators (host-only executables, built first)
include(ExternalProject)
ExternalProject_Add(ZEPP_GENERATOR
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generator
  CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND "")
ExternalProject_Get_property(ZEPP_GENERATOR SOURCE_DIR)
ExternalProject_Get_property(ZEPP_GENERATOR BINARY_DIR)
set(GENERATOR_BINARY_DIR ${BINARY_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} ${SOURCE_DIR} WORKING_DIRECTORY ${GENERATOR_BINARY_DIR} COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${CMAKE_COMMAND} --build . --parallel 4 WORKING_DIRECTORY ${GENERATOR_BINARY_DIR} COMMAND_ERROR_IS_FATAL ANY)

set(OP_TARGETS "")
add_subdirectory(dispatch)
add_subdirectory(combine)
add_subdirectory(direct)
add_subdirectory(dwire)

# the library: infrastructure + torch-facing op sources + the routing kernel
set(CORE_FILES
  core/op_registry.cu core/op_registry_proto_utils.cc core/cudaipc_barrier_all.cu
  core/utils.cc core/cuda_common.cc core/cuda_common.cu core/cuda_stub.cc core/nvml_stub.cc
  core/ths_op.cc core/flux_shm.cc core/helper_ops.cc core/wire_runtime.cc core/kernel_registry.cc
  core/verdict.cc core/step_state.cu)
set(OP_FILES
  dispatch/ths_op/dispatch_gemm.cc
  combine/ths_op/gemm_combine.cc
  direct/all2all_single_2d.cc direct/gemm_only.cc
  planner/routing.cu planner/swap_decide.cu planner/lane_device.cu)
set(_VER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/core/version.ld)
add_library(zepp_cuda SHARED ${CORE_FILES} ${OP_FILES})
set_target_properties(zepp_cuda PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINK_DEPENDS ${_VER_FILE} POSITION_INDEPENDENT_CODE ON)
target_link_libraries(zepp_cuda
  PUBLIC -lnvshmem_host CUDA::cudart CUDA::cuda_driver ${OP_TARGETS} ${TORCH_LIBRARIES}
  PRIVATE -lnvshmem_device)
target_link_options(zepp_cuda PRIVATE
  LINKER:--version-script=${_VER_FILE} LINKER:--no-as-needed)
install(TARGETS zepp_cuda LIBRARY DESTINATION lib)
