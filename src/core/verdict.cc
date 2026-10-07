// Deferred capacity verdict (core/verdict.h).
#include "core/verdict.h"

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

namespace bytedance::flux {

namespace {
int64_t *g_block = nullptr;
std::once_flag g_alloc;
std::atomic<bool> g_armed{false};

void
check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("deferred verdict: ") + what + ": " + cudaGetErrorString(e));
  }
}
}  // namespace

int64_t *
verdict_dev() {
  std::call_once(g_alloc, [] {
    check(cudaMalloc(&g_block, Verdict::kWords * sizeof(int64_t)), "cudaMalloc");
    check(cudaMemset(g_block, 0, Verdict::kWords * sizeof(int64_t)), "cudaMemset");
  });
  return g_block;
}

int64_t *
verdict_armed() {
  return g_armed.load(std::memory_order_relaxed) ? g_block : nullptr;
}

void
verdict_begin(cudaStream_t stream) {
  int64_t *b = verdict_dev();
  check(cudaMemsetAsync(b, 0, Verdict::kWords * sizeof(int64_t), stream), "cudaMemsetAsync");
  g_armed.store(true, std::memory_order_relaxed);
}

void
verdict_end() {
  g_armed.store(false, std::memory_order_relaxed);
}

void
verdict_reset(cudaStream_t stream) {
  int64_t *b = verdict_dev();
  check(cudaMemsetAsync(b, 0, Verdict::kWords * sizeof(int64_t), stream), "cudaMemsetAsync");
}

}  // namespace bytedance::flux
