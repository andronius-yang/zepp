//===- cuda_common.cu ------------------------------------------- C++ ---===//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "core/wire_runtime.h"
#include "flux/cuda/kernel_registry.h"

namespace bytedance::flux {
void
copy_continous_aligned(
    void *dst,
    const void *src,
    size_t nbytes,
    int threadblock_count,
    int thread_count,
    cudaStream_t stream) {
  dim3 grid(threadblock_count);
  dim3 block(thread_count);
  {  // copy by uint4
    using PackT = uint4;
    constexpr int kPackSize = sizeof(PackT);
    if (intptr_t(dst) % sizeof(PackT) == 0 && intptr_t(src) % sizeof(PackT) == 0 &&
        nbytes % kPackSize == 0) {
      copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
      CUTE_CHECK_ERROR(cudaGetLastError());
      return;
    }
  }
  {  // copy by uint2
    using PackT = uint2;
    constexpr int kPackSize = sizeof(PackT);
    if (intptr_t(dst) % sizeof(PackT) == 0 && intptr_t(src) % sizeof(PackT) == 0 &&
        nbytes % kPackSize == 0) {
      copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
      CUTE_CHECK_ERROR(cudaGetLastError());
      return;
    }
  }
  {  // copy by uint
    using PackT = uint;
    constexpr int kPackSize = sizeof(PackT);
    if (intptr_t(dst) % sizeof(PackT) == 0 && intptr_t(src) % sizeof(PackT) == 0 &&
        nbytes % kPackSize == 0) {
      copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
      CUTE_CHECK_ERROR(cudaGetLastError());
      return;
    }
  }
  {  // copy by int16_t
    using PackT = int16_t;
    constexpr int kPackSize = sizeof(PackT);
    if (intptr_t(dst) % sizeof(PackT) == 0 && intptr_t(src) % sizeof(PackT) == 0 &&
        nbytes % kPackSize == 0) {
      copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
      CUTE_CHECK_ERROR(cudaGetLastError());
      return;
    }
  }
  {  // copy by int8_t
    using PackT = int8_t;
    copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
    CUTE_CHECK_ERROR(cudaGetLastError());
    return;
  }
}

}  // namespace bytedance::flux

namespace bytedance::flux {
// One-warp spin wait on a 64-bit word (>= value). A kernel, not a stream memory operation: a memop
// wait blocks its hardware channel until satisfied, so a stream that happens to share the channel
// with the producer of the value deadlocks (e.g. the swap lane inside a serving process, whose
// stream layout differs from the benchmark's). A resident one-block kernel occupies an SM slot
// briefly instead and never blocks other streams' launches. The word is read with system-scope acquire
// (its writers are other streams' memops, peers' copies and the NIC). The process kill word (pinned host
// memory, core/wire_runtime.h) is polled every 64 rounds: once set, the wait gives up, so a watchdog can
// turn a hang into an error.
__global__ void wait_geq_kernel(
    const unsigned long long *word, unsigned long long value, const unsigned long long *kill) {
  if (threadIdx.x == 0) {
    unsigned int rounds = 0;
    while (true) {
      unsigned long long v;
      asm volatile("ld.acquire.sys.global.u64 %0, [%1];\n" : "=l"(v) : "l"(word) : "memory");
      if (v >= value) {
        break;
      }
      if ((++rounds & 63u) == 0 && kill != nullptr &&
          *reinterpret_cast<volatile const unsigned long long *>(kill) != 0) {
        break;
      }
      __nanosleep(256);
    }
    __threadfence();
  }
}

void stream_wait_geq(const void *word, uint64_t value, cudaStream_t stream) {
  wait_geq_kernel<<<1, 32, 0, stream>>>(
      reinterpret_cast<const unsigned long long *>(word), value,
      reinterpret_cast<const unsigned long long *>(kill_word_dev()));
}

void stream_wait_geq_preload() {
  cudaFuncAttributes attr;
  CUDA_CHECK(cudaFuncGetAttributes(&attr, wait_geq_kernel));
}

// kernel registry (include/flux/cuda/kernel_registry.h): every kernel of this file
void cuda_common_kernels(KernelList &out) {
  out.push_back(ZEPP_KERNEL_ENTRY(wait_geq_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY_T("copy_continous_aligned_kernel<uint4>", copy_continous_aligned_kernel<uint4>));
  out.push_back(ZEPP_KERNEL_ENTRY_T("copy_continous_aligned_kernel<uint2>", copy_continous_aligned_kernel<uint2>));
  out.push_back(
      ZEPP_KERNEL_ENTRY_T("copy_continous_aligned_kernel<unsigned int>", copy_continous_aligned_kernel<uint>));
  out.push_back(ZEPP_KERNEL_ENTRY_T("copy_continous_aligned_kernel<short>", copy_continous_aligned_kernel<int16_t>));
  out.push_back(
      ZEPP_KERNEL_ENTRY_T("copy_continous_aligned_kernel<signed char>", copy_continous_aligned_kernel<int8_t>));
}
}  // namespace bytedance::flux
