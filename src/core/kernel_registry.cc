// Kernel registry (include/flux/cuda/kernel_registry.h): the per-file kernel lists, the kernels of the
// registered GEMM ops, and the one-time preload.
#include "flux/cuda/kernel_registry.h"

#include <mutex>
#include <stdexcept>

#include "flux/op_registry.h"

namespace bytedance::flux {

void
gemm_op_kernels(KernelList &out) {
  for (auto const &op : OpRegistry::instance().all_ops()) {
    const void *func = op->kernel_func();
    std::string name = op->kernel_name();
    if (func == nullptr || name.empty()) {
      throw std::runtime_error("kernel registry: a registered GEMM op does not name its kernel");
    }
    out.push_back(KernelEntry{std::move(name), func});
  }
}

KernelList
all_kernels() {
  KernelList out;
  cuda_common_kernels(out);
  cudaipc_barrier_kernels(out);
  dispatch_meta_kernels(out);
  dispatch_workspace_kernels(out);
  combine_kernels(out);
  combine_workspace_kernels(out);
  direct_wire_kernels(out);
  routing_kernels(out);
  swap_decide_kernels(out);
  lane_device_kernels(out);
  step_state_kernels(out);
  dwire_kernels(out);
  gemm_op_kernels(out);
  return out;
}

std::vector<std::string>
kernel_registry_names() {
  std::vector<std::string> names;
  for (auto const &e : all_kernels()) {
    names.push_back(e.name);
  }
  return names;
}

void
preload_kernels(KernelList const &list) {
  for (auto const &e : list) {
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, e.func);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "kernel registry: cudaFuncGetAttributes(" + e.name + ") failed: " + cudaGetErrorString(err));
    }
  }
}

int64_t
preload_all_kernels() {
  static std::once_flag once;
  static int64_t loaded = 0;
  std::call_once(once, []() {
    KernelList list = all_kernels();
    preload_kernels(list);
    loaded = (int64_t)list.size();
  });
  return loaded;
}

}  // namespace bytedance::flux
