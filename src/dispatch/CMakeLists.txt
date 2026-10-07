file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/registers")
execute_process(
  COMMAND ${GENERATOR_BINARY_DIR}/gen_dispatch "--dir=./registers" "--archs=80" "--sm_cores=${SM_CORES}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND_ERROR_IS_FATAL ANY)
file(GLOB OP_REGS ${CMAKE_CURRENT_BINARY_DIR}/registers/*.cu)
zepp_add_cu_obj_lib(zepp_dispatch "${OP_REGS};sort_util.cu;workspace_util.cu")
set(OP_TARGETS ${OP_TARGETS} zepp_dispatch PARENT_SCOPE)
