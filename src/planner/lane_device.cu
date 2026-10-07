// Staged swap lane on the device (serving, device swap decision): every step of arming, pushing and
// committing reads the result block of swap_decide (swap_decide.cu) on the device, so the host issues
// one launch per phase (no per-move copies or flag writes), and the pad table of the next step is
// rebuilt on the device (a host rebuild would need a pageable upload, which synchronizes the stream).
// Driven by SwapLane (arm_device / staged_phase_before / commit_after); pad_rebuild builds the table
// LayerState._refresh_pads builds on the host.
//
//   lane_arm      this rank, if it receives: raise its gate words of the unchanged slots (both
//                 matrices), write the moved-last schedule encoding and the weight-pointer overrides
//                 of the incoming slots (to the staging entry of each pull)
//   lane_push     matrix k: copy every slot this rank gives to a node peer into the peer's staging
//                 entry, then raise the peer's gate word of that slot (after a system-scope fence)
//   lane_commit   matrix k, after the GEMM: per incoming slot wait for its gate word, copy the staging
//                 entry into the slot, clear the override
//   pad_rebuild   the pad-expert table of this rank from the new placement (cyclic over the rank's
//                 single-replica hosted experts, else its hosted experts, else its home range)
// Every kernel returns at once when the block reports no swap (rounds == 0).
#include <torch/all.h>
#include "planner/routing.h"
#include "core/step_state.h"
#include "flux/cuda/kernel_registry.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define LANEDEV_CUDA_CHECK(expr)                                \
  do {                                                          \
    cudaError_t _e = (expr);                                    \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",              \
                cudaGetErrorString(_e));                        \
  } while (0)

namespace bytedance::flux {
namespace lanedev {

constexpr int kMaxNlp = 256;
constexpr int kMaxOut = 64;      // pulls one rank serves per step: <= ranks per node x staging cap
constexpr int kCopyBlocks = 32;
constexpr int kCopyThreads = 512;

__device__ __forceinline__ const long long *moves_of(const long long *blk, int R, int cap, int r) {
  return blk + 3 + R + (long long)r * cap * 4;
}

// always != 0 (deferred verdict: the host does not know whether this rank receives a slot, so both GEMMs take
// the gated path every step): the kernel also arms a step without incoming slots (every gate word raised,
// identity schedule), writes the front-class count at sched[gpe] and, when gmap is given, the combine's
// per-expert gate map (gmap[1 + j] = j for an incoming slot j, -1 otherwise).
__global__ void lane_arm_kernel(const long long *blk, int R, int rank, int nlp, int cap, long long *gate,
                                int *sched, long long *ovr0, long long *ovr1, const long long *stag_addr,
                                long long epoch_arg, const unsigned long long *epoch_ptr, int always, int *gmap) {
  const long long epoch = epoch_ptr != nullptr ? (long long)*epoch_ptr : epoch_arg;  // device step state
  if (threadIdx.x != 0) return;
  const int n = blk[0] == 0 ? 0 : (int)blk[3 + rank];
  if (n == 0 && !always) return;
  const int gpe = nlp + 1;
  const long long *mv = moves_of(blk, R, cap, rank);
  unsigned long long changed[kMaxNlp / 64] = {0, 0, 0, 0};
  for (int i = 0; i < n; ++i) {
    const int dj = (int)mv[i * 4];
    changed[dj >> 6] |= 1ull << (dj & 63);
    ovr0[1 + dj] = stag_addr[i];
    ovr1[1 + dj] = stag_addr[cap + i];
  }
  int f = 0, d = 0;
  sched[0] = f++;                                   // the pad group stays in front
  if (gmap != nullptr) gmap[0] = -1;
  for (int j = 0; j < nlp; ++j) {
    if ((changed[j >> 6] >> (j & 63)) & 1ull) {
      sched[1 + j] = (1 << 30) | d++;               // incoming: deferred class, moved last
      if (gmap != nullptr) gmap[1 + j] = j;         // index into the gate words past the pad group
    } else {
      sched[1 + j] = f++;
      gate[1 + j] = epoch;                          // unchanged slots are raised locally
      gate[gpe + 1 + j] = epoch;
      if (gmap != nullptr) gmap[1 + j] = -1;
    }
  }
  if (always) sched[gpe] = f;                       // front-class count (dispatch GEMM, sched_n_front < 0)
}

__global__ void lane_push_kernel(const long long *blk, int R, int L, int rank, int nlp, int cap, int k,
                                 const char *slots, long long slot_bytes, const long long *peer_stag,
                                 const long long *peer_gate, long long epoch_arg, const unsigned long long *epoch_ptr,
                                 unsigned int *counter) {
  const long long epoch = epoch_ptr != nullptr ? (long long)*epoch_ptr : epoch_arg;  // device step state
  __shared__ int s_n, s_ss[kMaxOut], s_dl[kMaxOut], s_idx[kMaxOut], s_dj[kMaxOut];
  if (blk[0] == 0) return;
  if (threadIdx.x == 0) {
    int n = 0;
    const int node = rank / L;
    for (int r = node * L; r < (node + 1) * L; ++r) {
      const int c = (int)blk[3 + r];
      const long long *mv = moves_of(blk, R, cap, r);
      for (int i = 0; i < c; ++i) {
        if ((int)mv[i * 4 + 1] == rank && n < kMaxOut) {
          s_dj[n] = (int)mv[i * 4];
          s_ss[n] = (int)mv[i * 4 + 2];
          s_dl[n] = r % L;
          s_idx[n] = i;
          ++n;
        }
      }
    }
    s_n = n;
  }
  __syncthreads();
  const int n = s_n;
  if (n == 0) return;
  const long long vpm = slot_bytes / 16;
  const long long total = (long long)n * vpm;
  for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < total;
       v += (long long)gridDim.x * blockDim.x) {
    const int m = (int)(v / vpm);
    const long long o = v - (long long)m * vpm;
    const int4 *src = reinterpret_cast<const int4 *>(slots + (long long)(1 + s_ss[m]) * slot_bytes) + o;
    int4 *dst = reinterpret_cast<int4 *>(reinterpret_cast<char *>(peer_stag[s_dl[m]]) +
                                         (long long)s_idx[m] * slot_bytes) + o;
    *dst = *src;
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int prev = atomicAdd(counter, 1u);
    if (prev == gridDim.x - 1) {                    // every block's copies are fenced: raise the gates
      __threadfence_system();
      for (int m = 0; m < n; ++m) {
        volatile long long *g = reinterpret_cast<volatile long long *>(peer_gate[s_dl[m]]);
        g[k * (nlp + 1) + 1 + s_dj[m]] = epoch;
      }
      __threadfence_system();
      *counter = 0u;
    }
  }
}

__global__ void lane_commit_kernel(const long long *blk, int R, int rank, int nlp, int cap, int k,
                                   const long long *gate, char *slots, long long slot_bytes,
                                   const char *stag, long long *ovr, long long epoch_arg,
                                   const unsigned long long *epoch_ptr) {
  const long long epoch = epoch_ptr != nullptr ? (long long)*epoch_ptr : epoch_arg;  // device step state
  if (blk[0] == 0) return;
  const int n = (int)blk[3 + rank];
  if (n == 0) return;
  const long long *mv = moves_of(blk, R, cap, rank);
  const long long vpm = slot_bytes / 16;
  for (int i = 0; i < n; ++i) {
    const int dj = (int)mv[i * 4];
    if (threadIdx.x == 0) {                          // the push may land after the GEMM (no rows, no wait)
      volatile const long long *g = reinterpret_cast<volatile const long long *>(gate + k * (nlp + 1) + 1 + dj);
      while (*g < epoch) __nanosleep(256);
      __threadfence();
    }
    __syncthreads();
    const int4 *src = reinterpret_cast<const int4 *>(stag + (long long)i * slot_bytes);
    int4 *dst = reinterpret_cast<int4 *>(slots + (long long)(1 + dj) * slot_bytes);
    for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < vpm;
         v += (long long)gridDim.x * blockDim.x) {
      dst[v] = src[v];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) ovr[1 + dj] = 0;  // the GEMM that read it has finished
  }
}

// pad_prev / flag (deferred verdict, nullable): a rebuild first saves the table it overwrites into pad_prev and
// sets *flag = 1; a step without a swap sets *flag = 0. A forward that runs again after a capacity abort then
// pads with the table the aborted pass used (pad_prev when *flag is set), so its routing is the same.
__global__ void pad_rebuild_kernel(const long long *blk, const long long *p2l, const int *lcnts, int rank,
                                   int nlp, int home, int *pad, long long n_entries, int *pad_prev, int *flag) {
  __shared__ int s_list[kMaxNlp];
  __shared__ int s_len;
  if (flag != nullptr && blockIdx.x == 0 && threadIdx.x == 0) *flag = blk[0] == 0 ? 0 : 1;
  if (blk[0] == 0) return;
  if (threadIdx.x == 0) {
    int n = 0;
    for (int j = 0; j < nlp; ++j) {
      const long long e = p2l[(long long)rank * nlp + j];
      if (e >= 0 && lcnts[e] == 1) s_list[n++] = (int)e;
    }
    if (n == 0) {
      for (int j = 0; j < nlp; ++j) {
        const long long e = p2l[(long long)rank * nlp + j];
        if (e >= 0) s_list[n++] = (int)e;
      }
    }
    s_len = n;
  }
  __syncthreads();
  const int n = s_len;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n_entries;
       i += (long long)gridDim.x * blockDim.x) {
    if (pad_prev != nullptr) pad_prev[i] = pad[i];
    pad[i] = n > 0 ? s_list[i % n] : (int)((long long)rank * home + i % home);
  }
}

// ---- dual3 lane on the device (gpu_plan with swap): the 3D schedule of the overlapped lane, driven by the decision
// block. W1 is pushed by its sender under the sender's dispatch GEMM (dual3_push_w1), W2 is pulled by its receiver
// under the receiver's combine GEMM from the sender's slot on the symmetric heap (dual3_pull_w2), so the combine
// GEMM's tile gates depend only on the receiver's own progress. The phase kernels are launched before their GEMM
// and spin on its start mark themselves (a stream wait parked on a side stream can block a hardware queue the
// forward stream shares). The commits join on words: dual3_wait_pushed before the W1 commit (this rank's pushes
// raised the receivers' gates), dual3_wait_acks before the W2 commit (the receivers pulled the slots it gave away).
__device__ __forceinline__ void wait_mark(const long long *mark, long long epoch) {
  if (threadIdx.x == 0) {
    volatile const long long *mk = reinterpret_cast<volatile const long long *>(mark);
    while (*mk < epoch) __nanosleep(256);
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kCopyThreads, 2)
dual3_push_w1_kernel(const long long *blk, int R, int L, int rank, int nlp, int cap, const char *slots,
                     long long slot_bytes, const long long *peer_stag, const long long *peer_gate,
                     const long long *mark, long long epoch_arg, const unsigned long long *epoch_ptr,
                     unsigned int *counter) {
  const long long epoch = epoch_ptr != nullptr ? (long long)*epoch_ptr : epoch_arg;
  __shared__ int s_n, s_ss[kMaxOut], s_dl[kMaxOut], s_idx[kMaxOut], s_dj[kMaxOut];
  if (blk[0] == 0) return;
  if (threadIdx.x == 0) {
    int n = 0;
    const int node = rank / L;
    for (int r = node * L; r < (node + 1) * L; ++r) {
      const int c = (int)blk[3 + r];
      const long long *mv = moves_of(blk, R, cap, r);
      for (int i = 0; i < c; ++i) {
        if ((int)mv[i * 4 + 1] == rank && n < kMaxOut) {
          s_dj[n] = (int)mv[i * 4];
          s_ss[n] = (int)mv[i * 4 + 2];
          s_dl[n] = r % L;
          s_idx[n] = i;
          ++n;
        }
      }
    }
    s_n = n;
  }
  __syncthreads();
  const int n = s_n;
  if (n == 0) return;
  wait_mark(mark, epoch);
  const long long vpm = slot_bytes / 16;
  const long long total = (long long)n * vpm;
  for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < total;
       v += (long long)gridDim.x * blockDim.x) {
    const int m = (int)(v / vpm);
    const long long o = v - (long long)m * vpm;
    const int4 *src = reinterpret_cast<const int4 *>(slots + (long long)(1 + s_ss[m]) * slot_bytes) + o;
    int4 *dst = reinterpret_cast<int4 *>(reinterpret_cast<char *>(peer_stag[s_dl[m]]) +
                                         (long long)s_idx[m] * slot_bytes) + o;
    *dst = *src;
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int prev = atomicAdd(counter, 1u);
    if (prev == gridDim.x - 1) {                    // every block's copies are fenced: raise the W1 gates
      __threadfence_system();
      for (int m = 0; m < n; ++m)
        reinterpret_cast<volatile long long *>(peer_gate[s_dl[m]])[1 + s_dj[m]] = epoch;
      __threadfence_system();
      *counter = 0u;
    }
  }
}

__global__ void __launch_bounds__(kCopyThreads, 2)
dual3_pull_w2_kernel(const long long *blk, int R, int L, int rank, int nlp, int cap, const long long *peer_w2,
                     long long slot_bytes, char *stag, long long *gate, const long long *peer_ack, const long long *mark,
                     long long epoch_arg, const unsigned long long *epoch_ptr, unsigned int *counter) {
  const long long epoch = epoch_ptr != nullptr ? (long long)*epoch_ptr : epoch_arg;
  if (blk[0] == 0) return;
  const int n = (int)blk[3 + rank];
  if (n == 0) return;
  wait_mark(mark, epoch);
  const long long *mv = moves_of(blk, R, cap, rank);
  const long long vpm = slot_bytes / 16;
  const long long total = (long long)n * vpm;
  for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < total;
       v += (long long)gridDim.x * blockDim.x) {
    const int m = (int)(v / vpm);
    const long long o = v - (long long)m * vpm;
    const int sr = (int)mv[m * 4 + 1], ss = (int)mv[m * 4 + 2];
    const int4 *src = reinterpret_cast<const int4 *>(reinterpret_cast<const char *>(peer_w2[sr % L]) +
                                                     (long long)(1 + ss) * slot_bytes) + o;
    int4 *dst = reinterpret_cast<int4 *>(stag + (long long)m * slot_bytes) + o;
    *dst = *src;
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int prev = atomicAdd(counter, 1u);
    if (prev == gridDim.x - 1) {                    // raise this rank's W2 gates, acknowledge every sender
      __threadfence_system();
      for (int m = 0; m < n; ++m) {
        const int dj = (int)mv[m * 4], sr = (int)mv[m * 4 + 1];
        reinterpret_cast<volatile long long *>(gate)[(nlp + 1) + 1 + dj] = epoch;
        reinterpret_cast<volatile long long *>(peer_ack[sr % L])[(rank % L) * cap + m] = epoch;
      }
      __threadfence_system();
      *counter = 0u;
    }
  }
}

// one warp: lane t covers node peer t; waits until every word this rank expects from (or put at) that peer is current
__global__ void dual3_wait_kernel(const long long *blk, int R, int L, int rank, int cap, const long long *words,
                                  int pushed, long long epoch_arg, const unsigned long long *epoch_ptr) {
  const long long epoch = epoch_ptr != nullptr ? (long long)*epoch_ptr : epoch_arg;
  if (blk[0] == 0) return;
  const int t = threadIdx.x;
  if (t < L) {
    const int r = (rank / L) * L + t;
    const int c = (int)blk[3 + r];
    const long long *mv = moves_of(blk, R, cap, r);
    for (int i = 0; i < c; ++i) {
      if ((int)mv[i * 4 + 1] != rank) continue;
      // pushed: the W1 gate this rank raised at peer t (words = peer gate addresses); else: peer t's acknowledgment
      volatile const long long *w =
          pushed ? reinterpret_cast<volatile const long long *>(reinterpret_cast<const long long *>(words)[t]) + 1 +
                       mv[i * 4]
                 : reinterpret_cast<volatile const long long *>(words) + (long long)t * cap + i;
      while (*w < epoch) __nanosleep(256);
    }
  }
  __syncwarp();
  __threadfence_system();
}

// epoch < 0: the swap epoch of the current step's slot (device step state, core/step_state.h)
const unsigned long long *epoch_ptr(int64_t epoch) {
  if (epoch >= 0) return nullptr;
  const int slot = step_current_slot();
  TORCH_CHECK(slot >= 0, "lane kernels: epoch < 0 needs the device step state of the step");
  return reinterpret_cast<const unsigned long long *>(step_slot_word(slot, StepSlot::kSwapEpoch));
}

void check_block(const torch::Tensor &blk) {
  TORCH_CHECK(blk.is_cuda() && blk.scalar_type() == torch::kLong && blk.is_contiguous(),
              "lane device ops: result block int64 cuda");
}

}  // namespace lanedev

void lane_arm(const torch::Tensor blk, int64_t R, int64_t rank, int64_t nlp, int64_t cap, torch::Tensor gate,
              torch::Tensor sched, torch::Tensor ovr0, torch::Tensor ovr1, const torch::Tensor stag_addr,
              int64_t epoch, bool always, c10::optional<torch::Tensor> gmap) {
  lanedev::check_block(blk);
  TORCH_CHECK(nlp >= 1 && nlp <= lanedev::kMaxNlp, "lane_arm: nlp <= 256");
  TORCH_CHECK(gate.scalar_type() == torch::kLong && gate.numel() >= 2 * (nlp + 1), "lane_arm: gate int64 [2*gpe]");
  TORCH_CHECK(sched.scalar_type() == torch::kInt && sched.numel() >= nlp + 1 + (always ? 1 : 0),
              "lane_arm: sched int32 [gpe] (+1 for the front-class count when always)");
  TORCH_CHECK(stag_addr.scalar_type() == torch::kLong && stag_addr.numel() >= 2 * cap, "lane_arm: stag_addr [2, cap]");
  TORCH_CHECK(!gmap.has_value() || (gmap->is_cuda() && gmap->scalar_type() == torch::kInt && gmap->numel() >= nlp + 1),
              "lane_arm: gmap int32 [gpe]");
  lanedev::lane_arm_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)rank, (int)nlp, (int)cap,
      reinterpret_cast<long long *>(gate.data_ptr<int64_t>()), sched.data_ptr<int>(),
      reinterpret_cast<long long *>(ovr0.data_ptr<int64_t>()), reinterpret_cast<long long *>(ovr1.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(stag_addr.data_ptr<int64_t>()), (long long)epoch, lanedev::epoch_ptr(epoch),
      always ? 1 : 0,
      gmap.has_value() ? gmap->data_ptr<int>() : nullptr);
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

void lane_push(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t nlp, int64_t cap, int64_t k,
               const torch::Tensor slots, int64_t slot_bytes, const torch::Tensor peer_stag,
               const torch::Tensor peer_gate, int64_t epoch, torch::Tensor counter) {
  lanedev::check_block(blk);
  TORCH_CHECK(slot_bytes % 16 == 0 && slots.is_contiguous(), "lane_push: slot size multiple of 16 bytes");
  TORCH_CHECK(counter.scalar_type() == torch::kInt && counter.numel() >= 1, "lane_push: counter int32");
  TORCH_CHECK(L * cap <= lanedev::kMaxOut, "lane_push: ranks per node x cap <= 64");
  lanedev::lane_push_kernel<<<lanedev::kCopyBlocks, lanedev::kCopyThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)L, (int)rank, (int)nlp, (int)cap,
      (int)k, static_cast<const char *>(slots.data_ptr()), (long long)slot_bytes,
      reinterpret_cast<const long long *>(peer_stag.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(peer_gate.data_ptr<int64_t>()), (long long)epoch, lanedev::epoch_ptr(epoch),
      reinterpret_cast<unsigned int *>(counter.data_ptr<int>()));
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

void lane_commit(const torch::Tensor blk, int64_t R, int64_t rank, int64_t nlp, int64_t cap, int64_t k,
                 const torch::Tensor gate, torch::Tensor slots, int64_t slot_bytes, int64_t stag_addr,
                 torch::Tensor ovr, int64_t epoch) {
  lanedev::check_block(blk);
  TORCH_CHECK(slot_bytes % 16 == 0 && slots.is_contiguous(), "lane_commit: slot size multiple of 16 bytes");
  lanedev::lane_commit_kernel<<<lanedev::kCopyBlocks, lanedev::kCopyThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)rank, (int)nlp, (int)cap, (int)k,
      reinterpret_cast<const long long *>(gate.data_ptr<int64_t>()), static_cast<char *>(slots.data_ptr()),
      (long long)slot_bytes, reinterpret_cast<const char *>(stag_addr),
      reinterpret_cast<long long *>(ovr.data_ptr<int64_t>()), (long long)epoch, lanedev::epoch_ptr(epoch));
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

void dual3_push_w1(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t nlp, int64_t cap,
                   const torch::Tensor slots, int64_t slot_bytes, const torch::Tensor peer_stag,
                   const torch::Tensor peer_gate, const torch::Tensor mark, int64_t epoch, torch::Tensor counter) {
  lanedev::check_block(blk);
  TORCH_CHECK(slot_bytes % 16 == 0 && slots.is_contiguous() && peer_stag.numel() == L && peer_gate.numel() == L,
              "dual3_push_w1: 16-byte slots, [L] peer tables");
  lanedev::dual3_push_w1_kernel<<<lanedev::kCopyBlocks, lanedev::kCopyThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)L, (int)rank, (int)nlp, (int)cap,
      static_cast<const char *>(slots.data_ptr()), (long long)slot_bytes,
      reinterpret_cast<const long long *>(peer_stag.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(peer_gate.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(mark.data_ptr<int64_t>()), (long long)epoch, lanedev::epoch_ptr(epoch),
      reinterpret_cast<unsigned int *>(counter.data_ptr<int>()));
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

void dual3_pull_w2(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t nlp, int64_t cap,
                   const torch::Tensor peer_w2, int64_t slot_bytes, int64_t stag_addr, torch::Tensor gate,
                   const torch::Tensor peer_ack, const torch::Tensor mark, int64_t epoch, torch::Tensor counter) {
  lanedev::check_block(blk);
  TORCH_CHECK(slot_bytes % 16 == 0 && peer_w2.numel() == L && peer_ack.numel() == L, "dual3_pull_w2: [L] tables");
  lanedev::dual3_pull_w2_kernel<<<lanedev::kCopyBlocks, lanedev::kCopyThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)L, (int)rank, (int)nlp, (int)cap,
      reinterpret_cast<const long long *>(peer_w2.data_ptr<int64_t>()), (long long)slot_bytes,
      reinterpret_cast<char *>(stag_addr), reinterpret_cast<long long *>(gate.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(peer_ack.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(mark.data_ptr<int64_t>()), (long long)epoch, lanedev::epoch_ptr(epoch),
      reinterpret_cast<unsigned int *>(counter.data_ptr<int>()));
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

void dual3_wait(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t cap, const torch::Tensor words,
                bool pushed, int64_t epoch) {
  lanedev::check_block(blk);
  TORCH_CHECK(L <= 32 && words.scalar_type() == torch::kLong, "dual3_wait: one warp, int64 words");
  lanedev::dual3_wait_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()), (int)R, (int)L, (int)rank, (int)cap,
      reinterpret_cast<const long long *>(words.data_ptr<int64_t>()), pushed ? 1 : 0, (long long)epoch,
      lanedev::epoch_ptr(epoch));
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

void pad_rebuild(const torch::Tensor blk, const torch::Tensor p2l, const torch::Tensor lcnts, int64_t rank,
                 int64_t nlp, int64_t home, torch::Tensor pad, c10::optional<torch::Tensor> pad_prev,
                 c10::optional<torch::Tensor> flag) {
  lanedev::check_block(blk);
  TORCH_CHECK(nlp >= 1 && nlp <= lanedev::kMaxNlp, "pad_rebuild: nlp <= 256");
  TORCH_CHECK(p2l.scalar_type() == torch::kLong && lcnts.scalar_type() == torch::kInt, "pad_rebuild: p2l int64, lcnts int32");
  TORCH_CHECK(pad.scalar_type() == torch::kInt && pad.is_contiguous(), "pad_rebuild: pad table int32");
  TORCH_CHECK(!pad_prev.has_value() || (pad_prev->scalar_type() == torch::kInt && pad_prev->is_contiguous() &&
                                         pad_prev->numel() == pad.numel()),
              "pad_rebuild: pad_prev int32 like the pad table");
  TORCH_CHECK(!flag.has_value() || (flag->is_cuda() && flag->scalar_type() == torch::kInt && flag->numel() >= 1),
              "pad_rebuild: flag int32 [1]");
  lanedev::pad_rebuild_kernel<<<8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const long long *>(blk.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(p2l.data_ptr<int64_t>()), lcnts.data_ptr<int>(), (int)rank, (int)nlp,
      (int)home, pad.data_ptr<int>(), (long long)pad.numel(),
      pad_prev.has_value() ? pad_prev->data_ptr<int>() : nullptr, flag.has_value() ? flag->data_ptr<int>() : nullptr);
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

// Row-bounded kernels of a deferred-verdict step (the host does not know how many rows this rank computes; the
// dispatch plan block holds it on the device): zero the first `rows` rows of the GEMM 1 output, and the SwiGLU
// activation over the first `rows` rows. rows = min(*rows_dev, rows_cap). Same arithmetic as the eager torch path
// (silu rounded to bf16, then the product rounded).
namespace actb {
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 432;

__global__ void zero_rows_bounded_kernel(uint4 *buf, long long row_vecs, const long long *rows_dev,
                                         long long rows_cap) {
  long long rows = *rows_dev;
  rows = rows < 0 ? 0 : (rows > rows_cap ? rows_cap : rows);
  const long long total = rows * row_vecs;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    buf[i] = make_uint4(0u, 0u, 0u, 0u);
  }
}

__global__ void silu_mul_bounded_kernel(const __nv_bfloat16 *h, __nv_bfloat16 *out, long long ffn,
                                        long long h_stride, long long o_stride, const long long *rows_dev,
                                        long long rows_cap) {
  long long rows = *rows_dev;
  rows = rows < 0 ? 0 : (rows > rows_cap ? rows_cap : rows);
  const long long vpr = ffn / 8;
  const long long total = rows * vpr;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    const long long r = i / vpr;
    const long long c = (i - r * vpr) * 8;
    const uint4 gv = *reinterpret_cast<const uint4 *>(h + r * h_stride + c);
    const uint4 uv = *reinterpret_cast<const uint4 *>(h + r * h_stride + ffn + c);
    const __nv_bfloat16 *g = reinterpret_cast<const __nv_bfloat16 *>(&gv);
    const __nv_bfloat16 *u = reinterpret_cast<const __nv_bfloat16 *>(&uv);
    uint4 ov;
    __nv_bfloat16 *o = reinterpret_cast<__nv_bfloat16 *>(&ov);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float x = __bfloat162float(g[k]);
      const __nv_bfloat16 s = __float2bfloat16(x / (1.0f + expf(-x)));
      o[k] = __float2bfloat16(__bfloat162float(s) * __bfloat162float(u[k]));
    }
    *reinterpret_cast<uint4 *>(out + r * o_stride + c) = ov;
  }
}

// layer-graph staging (one launch instead of up to six copy-engine copies / memsets): rows [0, n) of x / ids / w
// into the bucket buffers, rows [n, S_b) zero activations, the pad ids and zero weights
__global__ void graph_stage_in_kernel(const uint4 *x, const int *ids, const float *w, const int *pad, uint4 *xb, int *idsb,
                                      float *wb, long long n, long long S_b, long long xvec, int K) {
  const long long nx = S_b * xvec, ni = S_b * K, total = nx + 2 * ni;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    if (i < nx) {
      const long long r = i / xvec;
      xb[i] = r < n ? x[i] : make_uint4(0, 0, 0, 0);
    } else if (i < nx + ni) {
      const long long j = i - nx, r = j / K;
      idsb[j] = r < n ? ids[j] : pad[j - n * K];
    } else {
      const long long j = i - nx - ni, r = j / K;
      wb[j] = r < n ? w[j] : 0.0f;
    }
  }
}

// planner tail (one launch after the routing exchange): gather [R, 2n] int32 (rank r: n physical slots, then n
// gate weights as float bits) -> probs [R * n] (the bits), vce [R * n] = floor(phys / nlp) * gpe + 1 +
// (phys mod nlp), with torch's floor division and remainder semantics
__global__ void planner_tail_kernel(const int *gather, int *probs, int *vce, long long R, long long n, int nlp,
                                    int gpe) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < R * n;
       i += (long long)gridDim.x * blockDim.x) {
    const long long r = i / n, j = i - r * n;
    const int p = gather[r * 2 * n + j];
    probs[i] = gather[r * 2 * n + n + j];
    int q = p / nlp, m = p % nlp;
    if (m != 0 && ((m < 0) != (nlp < 0))) {
      m += nlp;
      q -= 1;
    }
    vce[i] = q * gpe + 1 + m;
  }
}

int blocks_for(long long work) {
  const long long b = (work + kThreads - 1) / kThreads;
  return (int)(b < 1 ? 1 : (b > kMaxBlocks ? kMaxBlocks : b));
}

void check_rows(const torch::Tensor &rows_dev) {
  TORCH_CHECK(rows_dev.is_cuda() && rows_dev.scalar_type() == torch::kLong && rows_dev.numel() >= 1,
              "row-bounded ops: rows_dev int64 cuda [>=1]");
}
}  // namespace actb

void zero_rows_bounded(torch::Tensor buf, const torch::Tensor rows_dev) {
  actb::check_rows(rows_dev);
  TORCH_CHECK(buf.is_cuda() && buf.dim() == 2 && buf.is_contiguous(), "zero_rows_bounded: contiguous 2-D cuda buffer");
  const long long row_bytes = buf.size(1) * buf.element_size();
  TORCH_CHECK(row_bytes % 16 == 0 && reinterpret_cast<uintptr_t>(buf.data_ptr()) % 16 == 0,
              "zero_rows_bounded: 16-byte rows and alignment");
  const long long row_vecs = row_bytes / 16;
  actb::zero_rows_bounded_kernel<<<actb::blocks_for(buf.size(0) * row_vecs), actb::kThreads, 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint4 *>(buf.data_ptr()), row_vecs,
      reinterpret_cast<const long long *>(rows_dev.data_ptr<int64_t>()), (long long)buf.size(0));
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

void silu_mul_bounded(const torch::Tensor h, torch::Tensor out, int64_t ffn, const torch::Tensor rows_dev) {
  actb::check_rows(rows_dev);
  TORCH_CHECK(h.is_cuda() && out.is_cuda() && h.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16 && h.dim() == 2 && out.dim() == 2,
              "silu_mul_bounded: bf16 2-D cuda tensors");
  TORCH_CHECK(h.size(1) == 2 * ffn && out.size(1) == ffn && out.size(0) >= h.size(0) && ffn % 8 == 0,
              "silu_mul_bounded: h [m, 2*ffn], out [>=m, ffn], ffn % 8 == 0");
  TORCH_CHECK(h.stride(1) == 1 && out.stride(1) == 1 && h.stride(0) % 8 == 0 && out.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(h.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "silu_mul_bounded: 16-byte aligned rows");
  actb::silu_mul_bounded_kernel<<<actb::blocks_for(h.size(0) * (ffn / 8)), actb::kThreads, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(h.data_ptr()), reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()),
      (long long)ffn, (long long)h.stride(0), (long long)out.stride(0),
      reinterpret_cast<const long long *>(rows_dev.data_ptr<int64_t>()), (long long)h.size(0));
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

void graph_stage_in(const torch::Tensor x, const torch::Tensor ids, const torch::Tensor w, const torch::Tensor pad,
                    torch::Tensor xb, torch::Tensor idsb, torch::Tensor wb) {
  const long long n = x.size(0), S_b = xb.size(0);
  const int K = (int)idsb.size(1);
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && xb.is_contiguous() && x.scalar_type() == xb.scalar_type() &&
                  (n == 0 || x.size(1) == xb.size(1)) && n <= S_b,
              "graph_stage_in: x [n, H] and xb [S_b, H] contiguous, same dtype");
  TORCH_CHECK(ids.scalar_type() == torch::kInt && idsb.scalar_type() == torch::kInt && pad.scalar_type() == torch::kInt &&
                  w.scalar_type() == torch::kFloat && wb.scalar_type() == torch::kFloat && ids.is_contiguous() &&
                  idsb.is_contiguous() && pad.is_contiguous() && w.is_contiguous() && wb.is_contiguous(),
              "graph_stage_in: int32 ids / pad, float32 weights, contiguous");
  TORCH_CHECK(pad.numel() >= (S_b - n) * K && (n == 0 || (ids.size(1) == K && w.size(1) == K)),
              "graph_stage_in: pad rows [S_b - n, K]");
  const long long row_bytes = xb.size(1) * xb.element_size();
  TORCH_CHECK(row_bytes % 16 == 0 && reinterpret_cast<uintptr_t>(xb.data_ptr()) % 16 == 0 &&
                  (n == 0 || reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0),
              "graph_stage_in: 16-byte rows and alignment");
  const long long xvec = row_bytes / 16;
  actb::graph_stage_in_kernel<<<actb::blocks_for(S_b * xvec + 2 * S_b * K), actb::kThreads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4 *>(x.data_ptr()), ids.data_ptr<int>(), w.data_ptr<float>(), pad.data_ptr<int>(),
      reinterpret_cast<uint4 *>(xb.data_ptr()), idsb.data_ptr<int>(), wb.data_ptr<float>(), n, S_b, xvec, K);
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

void planner_tail(const torch::Tensor gather, torch::Tensor probs, torch::Tensor vce, int64_t R, int64_t n,
                  int64_t nlp, int64_t gpe) {
  TORCH_CHECK(gather.is_cuda() && gather.is_contiguous() && gather.scalar_type() == torch::kInt &&
                  gather.numel() >= 2 * R * n && probs.is_contiguous() && probs.element_size() == 4 &&
                  probs.numel() >= R * n && vce.is_contiguous() && vce.scalar_type() == torch::kInt &&
                  vce.numel() >= R * n && nlp > 0,
              "planner_tail: int32 gather [R, 2n], 4-byte probs [R n], int32 vce [R n], nlp > 0");
  actb::planner_tail_kernel<<<actb::blocks_for(R * n), actb::kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      gather.data_ptr<int>(), reinterpret_cast<int *>(probs.data_ptr()), vce.data_ptr<int>(), (long long)R,
      (long long)n, (int)nlp, (int)gpe);
  LANEDEV_CUDA_CHECK(cudaGetLastError());
}

// kernel registry (include/flux/cuda/kernel_registry.h): every kernel of this file
void lane_device_kernels(KernelList &out) {
  out.push_back(ZEPP_KERNEL_ENTRY(actb::planner_tail_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(actb::graph_stage_in_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(lanedev::lane_arm_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(lanedev::lane_push_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(lanedev::lane_commit_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(lanedev::pad_rebuild_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(lanedev::dual3_push_w1_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(lanedev::dual3_pull_w2_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(lanedev::dual3_wait_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(actb::zero_rows_bounded_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(actb::silu_mul_bounded_kernel));
}

void lane_device_preload() {
  // lazy module loading: a kernel's module loads at its first launch, and a load behind a resident
  // persistent GEMM never completes; query the attributes once while the device is idle
  KernelList list;
  lane_device_kernels(list);
  preload_kernels(list);
}

}  // namespace bytedance::flux
