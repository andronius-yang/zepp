# device-issued wire (dwire.h): device-side NVSHMEM puts -> relocatable device code, linked like zepp_direct_wire
zepp_add_cu_obj_lib(zepp_dwire "dispatch_wire.cu")
target_compile_options(zepp_dwire PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
set(OP_TARGETS ${OP_TARGETS} zepp_dwire PARENT_SCOPE)
