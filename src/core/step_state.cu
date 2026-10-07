// Device step state (core/step_state.h).
#include "core/step_state.h"

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

#include "flux/cuda/kernel_registry.h"

namespace bytedance::flux {

namespace {
uint64_t *g_block = nullptr;
std::once_flag g_alloc;
std::atomic<int> g_slot{-1};
constexpr size_t kBlockWords = (size_t)(StepSlot::kSlots + 1) * StepSlot::kWords;

void
check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("step state: ") + what + ": " + cudaGetErrorString(e));
  }
}

__global__ void
step_head_kernel(uint64_t *blk, int slot, int swap_advance) {
  uint64_t *c = blk + (size_t)StepSlot::kSlots * StepSlot::kWords;
  uint64_t *s = blk + (size_t)slot * StepSlot::kWords;
  const uint64_t d = c[StepSlot::kDispatchRun] + 1, r = c[StepSlot::kCombineRun] + 1,
                 e = c[StepSlot::kSwapEpoch] + (swap_advance ? 1 : 0);
  c[StepSlot::kDispatchRun] = d;
  c[StepSlot::kCombineRun] = r;
  c[StepSlot::kSwapEpoch] = e;
  s[StepSlot::kDispatchRun] = d;
  s[StepSlot::kCombineRun] = r;
  s[StepSlot::kSwapEpoch] = e;
}

__global__ void
step_sync_kernel(uint64_t *blk, uint64_t d, uint64_t r, uint64_t e) {
  uint64_t *c = blk + (size_t)StepSlot::kSlots * StepSlot::kWords;
  c[StepSlot::kDispatchRun] = d;
  c[StepSlot::kCombineRun] = r;
  c[StepSlot::kSwapEpoch] = e;
}
}  // namespace

uint64_t *
step_block() {
  std::call_once(g_alloc, [] {
    check(cudaMalloc(&g_block, kBlockWords * sizeof(uint64_t)), "cudaMalloc");
    check(cudaMemset(g_block, 0, kBlockWords * sizeof(uint64_t)), "cudaMemset");
  });
  return g_block;
}

const uint64_t *
step_slot_word(int slot, int word) {
  if (slot < 0 || slot >= StepSlot::kSlots || word < 0 || word >= StepSlot::kWords) {
    throw std::runtime_error("step state: slot / word out of range");
  }
  return step_block() + (size_t)slot * StepSlot::kWords + word;
}

void
step_head(int slot, int swap_advance, cudaStream_t stream) {
  if (slot < 0 || slot >= StepSlot::kSlots) {
    throw std::runtime_error("step state: slot out of range");
  }
  step_head_kernel<<<1, 1, 0, stream>>>(step_block(), slot, swap_advance);
  check(cudaGetLastError(), "step_head_kernel");
  g_slot.store(slot, std::memory_order_relaxed);
}

int
step_current_slot() {
  return g_slot.load(std::memory_order_relaxed);
}

void
step_release() {
  g_slot.store(-1, std::memory_order_relaxed);
}

void
step_sync_counters(uint64_t dispatch_run, uint64_t combine_run, uint64_t swap_epoch, cudaStream_t stream) {
  step_sync_kernel<<<1, 1, 0, stream>>>(step_block(), dispatch_run, combine_run, swap_epoch);
  check(cudaGetLastError(), "step_sync_kernel");
}

// kernel registry (include/flux/cuda/kernel_registry.h): every kernel of this file
void
step_state_kernels(KernelList &out) {
  out.push_back(ZEPP_KERNEL_ENTRY(step_head_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(step_sync_kernel));
}

}  // namespace bytedance::flux
