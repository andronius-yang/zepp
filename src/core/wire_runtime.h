// Process-wide helpers of the wire: the kill word the spinning helper kernels poll.
#pragma once

#include <cstdint>

namespace bytedance::flux {

// Kill word: one pinned host-mapped 64-bit flag per process. Spinning helper kernels (stream_wait_geq,
// the combine's resident pre-reduce) poll it and give up their wait once it is non-zero, so a watchdog
// can turn a hang into an error; a host store reaches them even while the GPU executes no new work.
// wire_runtime_init() allocates it and preloads the spin kernel (the op constructors call it, on an
// idle device; a first launch next to a spinning kernel would be a lazy module load).
void wire_runtime_init();
const uint64_t *kill_word_dev();
void kill_word_set(uint64_t value);
uint64_t kill_word_get();
void stream_wait_geq_preload();  // cuda_common.cu

}  // namespace bytedance::flux
