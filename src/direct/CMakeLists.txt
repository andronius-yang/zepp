file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/registers")
execute_process(
  COMMAND ${GENERATOR_BINARY_DIR}/gen_gemm "--dir=./registers" "--archs=80" "--sm_cores=${SM_CORES}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND_ERROR_IS_FATAL ANY)
file(GLOB OP_REGS ${CMAKE_CURRENT_BINARY_DIR}/registers/*.cu)
file(GLOB TUNING_CONFIGS tuning_config/*.cu)
zepp_add_cu_obj_lib(zepp_direct_gemm "${OP_REGS};${TUNING_CONFIGS}")
# the all-to-all wire uses device-side NVSHMEM (relocatable device code)
zepp_add_cu_obj_lib(zepp_direct_wire "all2all_single_2d_impl.cu")
target_compile_options(zepp_direct_wire PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
set(OP_TARGETS ${OP_TARGETS} zepp_direct_gemm zepp_direct_wire PARENT_SCOPE)
