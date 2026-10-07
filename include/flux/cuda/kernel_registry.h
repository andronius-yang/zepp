// Kernel registry: every CUDA kernel this library may launch, listed next to its definition (one
// function per source file) and loaded once before serving.
//
// Why: under CUDA_MODULE_LOADING=LAZY (the CUDA 12 default; EAGER breaks NCCL in the torch process) a
// kernel's code is loaded at its first launch, and while code is being loaded with another kernel
// resident, the GPU runs no newly submitted work from any thread until the resident kernel ends. The
// fused ops' GEMM tiles and receivers spin on data that other threads and ranks issue, so the first
// launch of any not-yet-loaded kernel beside them can hang the step. cudaFuncGetAttributes forces the
// load of a kernel; doing it for every entry while the device is idle removes that class.
//
// The registry must equal the kernel list of libzepp_cuda.so (tests/test_kernel_registry.py compares
// the names with `cuobjdump -symbols`). A new kernel goes into the list function of its source file.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace bytedance::flux {

struct KernelEntry {
  // the kernel's name as c++filt prints it, without return type, namespaces or parameter list,
  // e.g. "a2av_combine_pack_kernel<__nv_bfloat16, true>"
  std::string name;
  const void *func;  // host stub
};
using KernelList = std::vector<KernelEntry>;

#define ZEPP_KERNEL_ENTRY(fn) ::bytedance::flux::KernelEntry{#fn, reinterpret_cast<const void *>(&fn)}
#define ZEPP_KERNEL_ENTRY_T(name, ...) \
  ::bytedance::flux::KernelEntry{name, reinterpret_cast<const void *>(&__VA_ARGS__)}

// One list function per source file that defines kernels (appends to `out`).
void cuda_common_kernels(KernelList &out);          // core/cuda_common.cu
void cudaipc_barrier_kernels(KernelList &out);      // core/cudaipc_barrier_all.cu
void dispatch_meta_kernels(KernelList &out);        // dispatch/sort_util.cu
void dispatch_workspace_kernels(KernelList &out);   // dispatch/workspace_util.cu
void combine_kernels(KernelList &out);              // combine/combine_kernels.cu
void combine_workspace_kernels(KernelList &out);    // combine/workspace_helper.cu
void direct_wire_kernels(KernelList &out);          // direct/all2all_single_2d_impl.cu
void routing_kernels(KernelList &out);              // planner/routing.cu
void swap_decide_kernels(KernelList &out);          // planner/swap_decide.cu
void lane_device_kernels(KernelList &out);          // planner/lane_device.cu
void step_state_kernels(KernelList &out);           // core/step_state.cu
void dwire_kernels(KernelList &out);                // dwire/dispatch_wire.cu
// the kernel of every GEMM op registered in the OpRegistry (all hparams of all metas)
void gemm_op_kernels(KernelList &out);

// Every kernel of the library: the lists above plus the registered GEMM ops.
KernelList all_kernels();
// Names only (no CUDA call; usable without a GPU).
std::vector<std::string> kernel_registry_names();
// cudaFuncGetAttributes on every entry of `list` (forces the lazy load). Call with the device idle.
void preload_kernels(KernelList const &list);
// preload_kernels(all_kernels()) once per process; later calls return at once. Returns the number of
// kernels loaded by the first call.
int64_t preload_all_kernels();

}  // namespace bytedance::flux
