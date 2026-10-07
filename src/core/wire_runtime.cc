#include "core/wire_runtime.h"

#include <cuda_runtime.h>

#include <atomic>
#include <mutex>

#include "flux/cuda/cuda_common.h"

namespace bytedance::flux {

namespace {
struct KillWord {
  uint64_t *host = nullptr;
  uint64_t *dev = nullptr;
};
KillWord &
kill_word() {
  static KillWord k = [] {
    KillWord w;
    void *p = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p, sizeof(uint64_t), cudaHostAllocMapped | cudaHostAllocPortable));
    w.host = static_cast<uint64_t *>(p);
    *reinterpret_cast<volatile uint64_t *>(w.host) = 0;
    void *d = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d, p, 0));
    w.dev = static_cast<uint64_t *>(d);
    return w;
  }();
  return k;
}
}  // namespace

void
wire_runtime_init() {
  static std::once_flag once;
  std::call_once(once, [] {
    (void)kill_word();
    stream_wait_geq_preload();
  });
}

const uint64_t *
kill_word_dev() {
  return kill_word().dev;
}

void
kill_word_set(uint64_t value) {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  *reinterpret_cast<volatile uint64_t *>(kill_word().host) = value;
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

uint64_t
kill_word_get() {
  return *reinterpret_cast<volatile uint64_t *>(kill_word().host);
}

}  // namespace bytedance::flux
