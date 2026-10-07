//===- combine_kernels.cu -------------------------------------------- C++ ---===//
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
// Kernels for the a2av_hier combine: a persistent pack kernel that turns
// the split-major GEMM output into destination-major send-panel chunks behind
// the per-split cascade flags, and a per-split topk reduce at the destination.
// All transport between the two is host-issued (copy engines / NIC, zero SMs).

#include <cub/cub.cuh>
#include <cutlass/barrier.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <type_traits>

#include "flux/args/gemm_combine.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/flux.h"
#include "combine/topk_gather_rs.hpp"
#include "flux/cuda/kernel_registry.h"
#include "core/verdict.h"

namespace bytedance::flux {
namespace {

template <typename T>
union PackU {
  static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>);
  constexpr static int kElemsPerPack = sizeof(uint4) / sizeof(T);
  uint4 data;
  T elems[kElemsPerPack];
};

CUTLASS_DEVICE void
storePack(void *ptr, uint4 data) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
               :
               : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
}

CUTLASS_DEVICE uint4
loadPack(void const *ptr) {
  uint4 data;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
               : "l"(ptr));
  return data;
}

template <typename T>
CUTLASS_DEVICE float
elem_to_float(T v) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(v);
  } else {
    return __bfloat162float(v);
  }
}

template <typename T>
CUTLASS_DEVICE T
float_to_elem(float f) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half(f);
  } else {
    return __float2bfloat16(f);
  }
}

// Epoch flag publish: a system-scope release store of the run id (the waiters are 64-bit front-end
// waits for >= run id, so the flags never need a reset).
CUTLASS_DEVICE void
store_release_sys_u64(uint64_t *ptr, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;\n" : : "l"(ptr), "l"(v) : "memory");
}

// Block-completion counter that returns to zero by itself: the last of `nblocks` arrivals sees
// nblocks - 1 and wraps the counter to 0 (atomicInc), so the next launch starts from zero without a
// per-step memset. True for the last arrival.
CUTLASS_DEVICE bool
last_block_arrival(int *counter, unsigned nblocks) {
  return atomicInc(reinterpret_cast<unsigned *>(counter), nblocks - 1) == nblocks - 1;
}

// Spinning combine kernels can become resident before (or beside) the combine GEMM: they prefer the maximum
// shared-memory carveout so an SM they occupy stays configured for a GEMM block (dwire/dispatch_wire.cu).
inline void
prefer_max_shared_once(const void *fn) {
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
}

template <typename T, bool kHasVecScale>
__global__ void
__launch_bounds__(512, 2) a2av_combine_pack_kernel(CombinePackArguments args) {
  using Barrier = cutlass::Barrier;
  // piece relay (fused pack: the GEMM wrote the panel; this kernel
  // is a pure flag pipeline): per piece, wait the piece's chunk flags, then
  // flip per (node, piece) flags in production order; the own-node per-split
  // group flag flips at the last piece so the intra ladder keeps its gate.
  if (args.n_pieces > 0) {
    for (int p = 0; p < args.n_pieces; p++) {
      for (int c = args.piece_first_chunk[p]; c < args.piece_first_chunk[p + 1]; c++) {
        Barrier::wait_eq(args.barrier, threadIdx.x, c, 1);
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        for (int gi = 0; gi < args.nnodes; gi++) {
          const int g = args.node_order[gi];
          if (last_block_arrival(args.piece_group_counters + g * 8 + p, gridDim.x)) {
            store_release_sys_u64(args.piece_group_flags + g * 8 + p, args.run_id);
          }
          if (p == args.n_pieces - 1) {
            if (last_block_arrival(args.group_counters + g * args.n_split + 0, gridDim.x)) {
              store_release_sys_u64(args.group_flags + g * args.n_split + 0, args.run_id);
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }
  // no-split chunked GEMM — data completeness = ALL chunk flags
  // (chunk completion order is not guaranteed, so wait each one). With
  // pieces, the per-(tn, piece) gating above replaces this entry wait.
  if (args.n_chunk_flags > 0) {
    for (int c = 0; c < args.n_chunk_flags; c++) {
      Barrier::wait_eq(args.barrier, threadIdx.x, c, 1);
    }
  }
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int64_t packs_per_row = n_per / kElemsPerPack;
  // deferred verdict: the wave-adapt decision is on the device (0 = the collapsed single-gate GEMM: one
  // split gate, ring order, data pack with the row scale), and so are the destination row ranges
  const bool dec_on = args.msplit_dec == nullptr || *args.msplit_dec != 0;
  const int msplit = args.msplit && dec_on;
  const int relay_only = args.relay_only && dec_on;
  auto node_row = [&](int g) -> int64_t {
    return args.node_row_dev != nullptr ? args.node_row_dev[(int64_t)g * args.node_row_stride]
                                        : args.node_row_start[g];
  };
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < args.n_split; sid++) {
    // Without msplit: the GEMM's tile->problem->split cascade releases this flag once
    // every expert's rows of column window sid are complete -- the minimal
    // gate, since any destination's rows interleave across all local experts.
    // M-split waves (msplit): gate PER RING STEP below instead — each wave's
    // flag releases as soon as every expert's rows for that wave's dest nodes
    // are complete, so early nodes pack (and their ladders fire) mid-GEMM.
    if (!msplit) {
      Barrier::wait_eq(args.barrier, threadIdx.x, sid, 1);
    }
    for (int gi = 0; gi < args.nnodes; gi++) {
      // remote-node chunks first so the NIC-bound flags flip earliest; the host
      // ladders consume flags in this same production order (no head-of-line).
      // msplit: the host-built schedule replaces the hardcoded ring.
      int g = msplit ? args.node_order[gi] : (args.node_idx + 1 + gi) % args.nnodes;
      if (msplit) {
        // schedule steps map monotonically to waves; re-waiting a set flag is free
        Barrier::wait_eq(args.barrier, threadIdx.x, args.wave_of_node[gi], 1);
      }
      const int64_t row_lo = node_row(g);
      // relay_only (fused pack): the GEMM scattered the panel rows
      // directly — skip the data loop, keep the wave-wait + flag flips
      const bool push = args.plan != nullptr;
      const int64_t total = (relay_only || push) ? 0 : (node_row(g + 1) - row_lo) * packs_per_row;
      for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
           idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t p = row_lo + idx / packs_per_row;
        const int64_t col = (idx % packs_per_row) * kElemsPerPack;
        const int src_row = args.pack_index[p];
        PackU<T> pk;
        pk.data = loadPack(
            (T const *)args.gemm_out + (int64_t)src_row * args.n + (int64_t)sid * n_per + col);
        if constexpr (kHasVecScale) {
          const float s = args.vec_scale[src_row];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kElemsPerPack; i++) {
            pk.elems[i] = float_to_elem<T>(elem_to_float<T>(pk.elems[i]) * s);
          }
        }
        storePack(
            (T *)args.send_panel + ((int64_t)sid * args.panel_rows + p) * n_per + col, pk.data);
      }
      // publish the (g, sid) chunk to the host put ladders once every block is
      // done -- including g == node_idx: the intra-node ladder gates on it
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        if (last_block_arrival(args.group_counters + g * args.n_split + sid, gridDim.x)) {
          store_release_sys_u64(args.group_flags + g * args.n_split + sid, args.run_id);
        }
      }
      if (push) {
        // device-issued wire: node g's rows straight into their destinations (see CombinePackArguments)
        const uint64_t run = args.run_ptr != nullptr ? *args.run_ptr : args.run_id;
        const int64_t *P = args.plan;
        const int L = args.local_world_size;
        auto dst_sig_of = [&](int dl, T *&dst, uint64_t *&sig) {
          const int d = g * L + dl;
          if (g == args.node_idx) {
            dst = (T *)args.recv_peer[dl] + ((int64_t)sid * args.recv_cap + P[args.plan_dst_off + d]) * n_per;
            sig = args.recvsig_peer[dl] + (int64_t)args.rank * args.n_split + sid;
          } else {
            dst = (T *)args.conv_peer[dl] + ((int64_t)sid * args.conv_cap + P[args.plan_conv_dst + d]) * n_per;
            sig = args.convsig_peer[dl] + ((int64_t)args.my_lr * args.nnodes + g) * args.n_split + sid;
          }
        };
        // one pass over the rows of every destination of node g (destination-major), one warp per row, then one
        // system fence (it covers all of this block's stores) and the per-destination arrivals: the last arriving
        // block of a destination raises its signal, so every signal is still after its data, and every block still
        // arrives once per destination (also with zero rows)
        int64_t cum[9];
        cum[0] = 0;
        for (int dl = 0; dl < L; dl++) {
          cum[dl + 1] = cum[dl] + P[args.plan_send_rows + g * L + dl];
        }
        constexpr int kU = 4;
        const int lane = threadIdx.x & 31;
        const int64_t nwarps = ((int64_t)gridDim.x * blockDim.x) >> 5;
        for (int64_t row = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5; row < cum[L]; row += nwarps) {
          int dl = 0;
          while (row >= cum[dl + 1]) {
            ++dl;
          }
          const int64_t r = row - cum[dl];
          const int64_t lo = P[args.plan_send_off + g * L + dl];
          T *dst;
          uint64_t *sig;
          dst_sig_of(dl, dst, sig);
          T *drow = dst + r * n_per;
          T const *srow;
          float sc = 1.0f;
          if (relay_only) {
            // fused pack: the GEMM epilogue wrote the panel rows of this wave (its flag is set)
            srow = (T const *)args.send_panel + ((int64_t)sid * args.panel_rows + lo + r) * n_per;
          } else {
            const int src_row = args.pack_index[lo + r];
            srow = (T const *)args.gemm_out + (int64_t)src_row * args.n + (int64_t)sid * n_per;
            if constexpr (kHasVecScale) {
              sc = args.vec_scale[src_row];
            }
          }
          for (int64_t c0 = 0; c0 < packs_per_row; c0 += 32 * kU) {
            PackU<T> pk[kU];
            CUTLASS_PRAGMA_UNROLL
            for (int u = 0; u < kU; u++) {
              const int64_t c = c0 + u * 32 + lane;
              if (c < packs_per_row) {
                pk[u].data = loadPack(srow + c * kElemsPerPack);
              }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int u = 0; u < kU; u++) {
              const int64_t c = c0 + u * 32 + lane;
              if (c < packs_per_row) {
                if constexpr (kHasVecScale) {
                  if (!relay_only) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kElemsPerPack; i++) {
                      pk[u].elems[i] = float_to_elem<T>(elem_to_float<T>(pk[u].elems[i]) * sc);
                    }
                  }
                }
                storePack(drow + c * kElemsPerPack, pk[u].data);
              }
            }
          }
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
          for (int dl = 0; dl < L; dl++) {
            T *dst;
            uint64_t *sig;
            dst_sig_of(dl, dst, sig);
            if (last_block_arrival(reinterpret_cast<int *>(args.push_counters) + (g * L + dl) * args.n_split + sid,
                                   gridDim.x)) {
              __threadfence_system();
              store_release_sys_u64(sig, run);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

CUTLASS_DEVICE uint64_t
load_acquire_sys_u64(uint64_t const *ptr) {
  uint64_t v;
  asm volatile("ld.global.acquire.sys.b64 %0, [%1];\n" : "=l"(v) : "l"(ptr));
  return v;
}

// source lane of a recv-panel row: the unique s with cum[s] <= row < cum[s+1]
// (zero-row lanes have cum[s] == cum[s+1] and can never contain a row)
CUTLASS_DEVICE int
lane_of_row(int64_t const *cum, int world_size, int64_t row) {
  int lo = 0, hi = world_size - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (row < cum[mid + 1]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}


// Source-side gateway pre-reduce for compress: per (split, target node in
// inter-ladder rotation order) thread 0 of each block spins on the L per-peer
// convergence signals, then the block strides the segment's wire rows, merging
// each row's CSR conv contributions in fp32; group completion flips the
// (tn, sid) wire flag via the pack kernel's counter handshake.
template <typename T>
__global__ void
__launch_bounds__(512, 1) a2av_combine_prereduce_kernel(CombinePreReduceArguments args) {
  const uint64_t run_id = args.run_ptr != nullptr ? *args.run_ptr : args.run_id;  // device step state
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int64_t packs_per_row = n_per / kElemsPerPack;
  const int NN = args.nnodes;
  const int L = args.local_world_size;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < args.n_split; sid++) {
    T const *conv = (T const *)args.conv_panel + (int64_t)sid * args.conv_rows * n_per;
    T *wire = (T *)args.wire_panel + (int64_t)sid * args.wire_rows * n_per;
    // pieces: PIECE-OUTER visit order — the kernel drains (tn, piece)
    // cells serially, and piece flags fire piece-major globally; tn-outer
    // would park every later tn's early pieces behind the first tn's last
    // piece (~GEMM end). P == 1 with the per-split signals is the plain
    // per-tn visit.
    const int P = args.n_pieces > 0 ? args.n_pieces : 1;
    for (int p = 0; p < P; p++) {
      for (int gi = 0; gi < NN - 1; gi++) {
        // host-built schedule (ring by default, or size-sorted); the
        // panel/segment layout stays tn-ascending, only the visit order moves
        const int tn = args.node_order[gi];
        const int seg = tn < args.node_idx ? tn : tn - 1;
        if (threadIdx.x == 0) {
          for (int ls = 0; ls < L; ls++) {
            uint64_t const *sig =
                args.n_pieces > 0
                    ? args.piece_conv_sigs + ((int64_t)ls * NN + tn) * 8 + p
                    : args.conv_signals + ((int64_t)ls * NN + tn) * args.n_split + sid;
            uint64_t spins = 0;
            while (load_acquire_sys_u64(sig) < run_id) {
              __nanosleep(200);
              ++spins;
              // kill word (host memory, so polled every 64 rounds): give up the wait
              if ((spins & 63) == 0 && args.kill_word != nullptr &&
                  *reinterpret_cast<volatile uint64_t const *>(args.kill_word) != 0) {
                break;
              }
              if (args.spin_limit != 0 && spins >= args.spin_limit) {
                printf(
                    "[a2av-combine] prereduce conv-signal SPIN LIMIT: node %d "
                    "tn %d ls %d sid %d piece %d run_id %llu\n",
                    args.node_idx, tn, ls, sid, p,
                    (unsigned long long)run_id);
                __trap();
              }
            }
          }
        }
        __syncthreads();
        int64_t w_lo = args.wire_seg_dev != nullptr ? args.wire_seg_dev[seg] : args.wire_seg_start[seg];
        int64_t w_hi =
            args.wire_seg_dev != nullptr ? args.wire_seg_dev[seg + 1] : args.wire_seg_start[seg + 1];
        if (args.n_pieces > 0) {
          w_hi = w_lo + args.piece_start[seg * 9 + p + 1];
          w_lo = w_lo + args.piece_start[seg * 9 + p];
        }
        const int64_t total = (w_hi - w_lo) * packs_per_row;
        for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
             idx += (int64_t)gridDim.x * blockDim.x) {
          const int64_t w = w_lo + idx / packs_per_row;
          const int64_t col = (idx % packs_per_row) * kElemsPerPack;
          float acc[kElemsPerPack];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kElemsPerPack; i++) {
            acc[i] = 0.0f;
          }
          for (int32_t k = args.wire_ptr[w]; k < args.wire_ptr[w + 1]; k++) {
            PackU<T> pk;
            pk.data = loadPack(conv + (int64_t)args.wire_copy[k] * n_per + col);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kElemsPerPack; i++) {
              acc[i] += elem_to_float<T>(pk.elems[i]);
            }
          }
          PackU<T> out;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kElemsPerPack; i++) {
            out.elems[i] = float_to_elem<T>(acc[i]);
          }
          storePack(wire + w * n_per + col, out.data);
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
          if (args.n_pieces > 0) {
            if (last_block_arrival(args.piece_wire_counters + tn * 8 + p, gridDim.x)) {
              store_release_sys_u64(args.piece_wire_flags + tn * 8 + p, run_id);
            }
          } else {
            if (last_block_arrival(args.wire_counters + tn * args.n_split + sid, gridDim.x)) {
              store_release_sys_u64(args.wire_flags + tn * args.n_split + sid, run_id);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}



template <typename T>
__global__ void
a2av_combine_reduce_kernel(CombineReduceArguments args) {
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int64_t packs_per_row = n_per / kElemsPerPack;
  const int64_t total = args.ntokens_local * packs_per_row;
  T const *panel = (T const *)args.recv_panel + (int64_t)args.sid * args.panel_rows * n_per;
  // deferred verdict set: degenerate step, zero rows (nothing was routed)
  const int topk = (args.verdict != nullptr &&
                    *reinterpret_cast<volatile int64_t const *>(args.verdict + Verdict::kDead) != 0)
                       ? 0
                       : args.topk;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t t = idx / packs_per_row;
    const int64_t col = (idx % packs_per_row) * kElemsPerPack;
    float acc[kElemsPerPack];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerPack; i++) {
      acc[i] = 0.0f;
    }
    for (int j = 0; j < topk; j++) {
      const int64_t row = args.reduce_index[t * args.topk + j];
      PackU<T> pk;
      pk.data = loadPack(panel + row * n_per + col);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElemsPerPack; i++) {
        acc[i] += elem_to_float<T>(pk.elems[i]);
      }
    }
    PackU<T> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerPack; i++) {
      out.elems[i] = float_to_elem<T>(acc[i]);
    }
    storePack((T *)args.output + t * args.n + (int64_t)args.sid * n_per + col, out.data);
  }
}






// ---- completion-bucketed receiver kernels --------------

__global__ void
a2av_bucket_map_kernel(A2AVBucketMapArguments args) {
  for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < args.ntokens;
       t += (int64_t)gridDim.x * blockDim.x) {
    int pos = 0;
    for (int32_t k = args.red_ptr[t]; k < args.red_ptr[t + 1]; k++) {
      const int32_t row = args.red_row[k];
      // lane of this recv row: binary search over the C' per-rank prefixes
      int lo = 0, hi = args.world_size - 1;
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (args.lane_off[mid] <= row) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }
      const int p = args.chain_pos[lo];
      pos = p > pos ? p : pos;
    }
    args.comp[t] = pos;
    atomicAdd(args.bucket_cnt + pos, 1);
  }
}

__global__ void
a2av_bucket_scan_kernel(A2AVBucketScanArguments args) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int32_t acc = 0;
  args.bucket_ptr[0] = 0;
  for (int b = 0; b < args.n_chain; b++) {
    acc += args.bucket_cnt[b];
    args.bucket_ptr[b + 1] = acc;
    args.bucket_cur[b] = 0;
  }
}

__global__ void
a2av_bucket_scatter_kernel(A2AVBucketScatterArguments args) {
  for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < args.ntokens;
       t += (int64_t)gridDim.x * blockDim.x) {
    const int pos = args.comp[t];
    const int32_t idx = atomicAdd(args.bucket_cur + pos, 1);
    args.bucket_tok[args.bucket_ptr[pos] + idx] = (int32_t)t;
  }
}

// One completion bucket's tokens, folded with the register CSR reduce: all
// contributions of a bucket's tokens are resident once its lane wait fires
// (sequential waits on one stream give the chain-prefix guarantee), so each
// token is read once and written once — wait-all's byte budget with
// arrival-order start times. Token order inside a bucket is
// scatter-nondeterministic, but each token's fold walks its own CSR slice in
// order, so the output is bitwise-identical to the wait-all reduce.
// the lane wait alone (one thread), ahead of the non-waiting wide fold of a remote lane
__global__ void
a2av_lane_wait_kernel(uint64_t const *sig, uint64_t const *run_ptr, uint64_t run_host, uint64_t const *kill) {
  if (threadIdx.x != 0) {
    return;
  }
  const uint64_t run = run_ptr != nullptr ? *run_ptr : run_host;
  for (unsigned spin = 0; load_acquire_sys_u64(sig) < run; ++spin) {
    if ((spin & 63) == 0 && kill != nullptr && *(volatile uint64_t const *)kill != 0) {
      return;
    }
    __nanosleep(128);
  }
}

template <typename T>
__global__ void
a2av_combine_bucket_reduce_kernel(A2AVBucketReduceArguments args) {
  if (args.wait_sig != nullptr) {
    __shared__ int s_ok;
    if (threadIdx.x == 0) {
      const uint64_t run = args.run_ptr != nullptr ? *args.run_ptr : args.run_host;
      int ok = 1;
      for (unsigned spin = 0; load_acquire_sys_u64(args.wait_sig) < run; ++spin) {
        if ((spin & 63) == 0 && args.kill != nullptr && *(volatile uint64_t const *)args.kill != 0) {
          ok = 0;
          break;
        }
        __nanosleep(128);
      }
      s_ok = ok;
    }
    __syncthreads();
    if (!s_ok) {
      return;
    }
  }
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int64_t packs_per_row = n_per / kElemsPerPack;
  const int32_t lo = args.bucket_ptr[args.bucket];
  const int32_t hi = args.bucket_ptr[args.bucket + 1];
  const int64_t total = (int64_t)(hi - lo) * packs_per_row;
  T const *panel = (T const *)args.recv_panel + (int64_t)args.sid * args.panel_rows * n_per;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t t = args.bucket_tok[lo + idx / packs_per_row];
    const int64_t col = (idx % packs_per_row) * kElemsPerPack;
    float acc[kElemsPerPack];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerPack; i++) {
      acc[i] = 0.0f;
    }
    for (int32_t k = args.red_ptr[t]; k < args.red_ptr[t + 1]; k++) {
      PackU<T> pk;
      pk.data = loadPack(panel + (int64_t)args.red_row[k] * n_per + col);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kElemsPerPack; i++) {
        acc[i] += elem_to_float<T>(pk.elems[i]);
      }
    }
    PackU<T> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerPack; i++) {
      out.elems[i] = float_to_elem<T>(acc[i]);
    }
    storePack((T *)args.output + t * args.n + (int64_t)args.sid * n_per + col, out.data);
  }
}


}  // namespace

// Force-load every combine kernel at construction time. Under
// CUDA_MODULE_LOADING=LAZY (the launch.sh default) a kernel's module is loaded
// at its FIRST launch; the compress schedule puts a persistent spin kernel
// (the pre-reduce) on the device BEFORE the epoch's first launch of the
// remaining kernels, and a first-launch load that must complete behind a
// never-exiting resident kernel deadlocks the epoch. Attribute queries are the
// documented cudart way to preload a kernel under lazy loading, and the ctor
// runs with an idle device, so every load here is trivial. Every kernel of
// this file is loaded (both element types, every derive / receiver kernel:
// the overlapped derive runs beside the spinning dispatch GEMM), from the
// kernel registry list at the end of the file.
void
a2av_combine_preload(DataTypeEnum dtype) {
  FLUX_CHECK(dtype == DataTypeEnum::BF16 || dtype == DataTypeEnum::FP16)
      << "unsupported dtype for a2av combine preload: " << dtype;
  KernelList list;
  combine_kernels(list);
  combine_workspace_kernels(list);
  preload_kernels(list);
}

void
a2av_combine_pack(
    CombinePackArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  // 512-thread blocks (grid-stride work): a block of this kernel (<= 64 registers) must fit beside one
  // resident GEMM2 block, or the block scheduler holds it, and every later kernel, behind that GEMM
  constexpr int kThreads = 512;
  FLUX_CHECK_LE(args.nnodes, kA2AVMaxNodes);
  FLUX_CHECK(args.n_per % 8 == 0) << "n/n_split must be a multiple of the 8-elem pack width";
  FLUX_CHECK(args.plan == nullptr || args.n_pieces == 0) << "device-issued wire: no piece relay";
  dim3 grid(2 * args.threadblock_count), block(kThreads);
  const bool has_vec_scale = args.vec_scale != nullptr;
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(cute::true_type{}, cute::false_type{})),
      [&](auto tup) {
        auto [cdtype, has_vec_scale_] = tup;
        return cdtype == dtype && has_vec_scale_ == has_vec_scale;
      },
      [&](auto tup) {
        auto [cdtype, has_vec_scale_] = tup;
        using T = decltype(to_cuda_dtype(cdtype));
        constexpr bool kHasVecScale = decltype(has_vec_scale_){};
        static const bool carveout_set = (prefer_max_shared_once(reinterpret_cast<const void *>(&a2av_combine_pack_kernel<T, kHasVecScale>)), true);
        (void)carveout_set;
        a2av_combine_pack_kernel<T, kHasVecScale><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av combine pack: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}


void
a2av_combine_reduce(
    CombineReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kThreads = 512;
  FLUX_CHECK(args.n_per % 8 == 0) << "n/n_split must be a multiple of the 8-elem pack width";
  dim3 grid(args.threadblock_count), block(kThreads);
  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [&](auto cdtype) { return cdtype == dtype; },
      [&](auto cdtype) {
        using T = decltype(to_cuda_dtype(cdtype));
        a2av_combine_reduce_kernel<T><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av combine reduce: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_combine_prereduce(
    CombinePreReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kThreads = 512;
  FLUX_CHECK(args.n_per % 8 == 0) << "n/n_split must be a multiple of the 8-elem pack width";
  FLUX_CHECK_LE(args.nnodes, kA2AVMaxNodes);
  FLUX_CHECK_GT(args.nnodes, 1) << "compress pre-reduce is a multi-node stage";
  dim3 grid(args.threadblock_count), block(kThreads);
  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [&](auto cdtype) { return cdtype == dtype; },
      [&](auto cdtype) {
        using T = decltype(to_cuda_dtype(cdtype));
        static const bool carveout_set = (prefer_max_shared_once(reinterpret_cast<const void *>(&a2av_combine_prereduce_kernel<T>)), true);
        (void)carveout_set;
        a2av_combine_prereduce_kernel<T><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av pre-reduce: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}



namespace {
__global__ void
a2av_invert_index_kernel(A2AVInvertIndexArguments args) {
  const int64_t n = args.n_dev != nullptr ? *args.n_dev : args.n;
  for (int64_t p = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; p < n;
       p += (int64_t)gridDim.x * blockDim.x) {
    args.out[args.idx[p]] = (int32_t)p;
  }
}

// deferred verdict: K-side fold of the row scales into the combine GEMM input (args/gemm_combine.h)
template <typename T>
__global__ void
a2av_fold_scales_kernel(A2AVFoldScalesArguments args) {
  const int64_t rows = *args.rows_dev;
  if (args.verdict != nullptr && blockIdx.x == 0 && threadIdx.x == 0 && rows > args.rows_bound) {
    atomicOr(reinterpret_cast<unsigned long long *>(args.verdict + Verdict::kErr),
             (unsigned long long)Verdict::kErrCombineRows);
    atomicOr(reinterpret_cast<unsigned long long *>(args.verdict + Verdict::kMask),
             (unsigned long long)Verdict::kErrorBit);
  }
  if (!args.fold || (args.dec != nullptr && *args.dec == 0)) {
    return;
  }
  const int64_t m = rows < args.rows_bound ? rows : args.rows_bound;
  T *x = reinterpret_cast<T *>(args.input);
  if (args.vec8) {
    // 16-byte accesses, the same per-element arithmetic (one row per vector: k % 8 == 0)
    const int64_t kv = args.k / 8;
    uint4 *xv = reinterpret_cast<uint4 *>(x);
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < m * kv;
         i += (int64_t)gridDim.x * blockDim.x) {
      const float s = args.scales[i / kv];
      uint4 v = xv[i];
      T *e = reinterpret_cast<T *>(&v);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        e[j] = float_to_elem<T>(elem_to_float<T>(e[j]) * s);
      }
      xv[i] = v;
    }
    return;
  }
  const int64_t total = m * args.k;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    x[i] = float_to_elem<T>(elem_to_float<T>(x[i]) * args.scales[i / args.k]);
  }
}
}  // namespace

void
a2av_invert_index(A2AVInvertIndexArguments const &args, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = (int)std::min<int64_t>((args.n + kThreads - 1) / kThreads, 4096);
  a2av_invert_index_kernel<<<blocks > 0 ? blocks : 1, kThreads, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_fold_scales(A2AVFoldScalesArguments const &args_in, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kThreads = 256;
  A2AVFoldScalesArguments args = args_in;
  // 2-byte types: 8 elements = 16 bytes per access when every row starts 16-byte aligned
  args.vec8 = (args.k % 8 == 0 && reinterpret_cast<uintptr_t>(args.input) % 16 == 0) ? 1 : 0;
  const int64_t total = std::max<int64_t>(args.rows_bound, 1) * args.k / (args.vec8 ? 8 : 1);
  const int blocks = (int)std::min<int64_t>((total + kThreads - 1) / kThreads, 1024);
  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [&](auto cdtype) { return cdtype == dtype; },
      [&](auto cdtype) {
        using T = decltype(to_cuda_dtype(cdtype));
        a2av_fold_scales_kernel<T><<<blocks > 0 ? blocks : 1, kThreads, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for the scale fold: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}




void
a2av_bucket_map(A2AVBucketMapArguments const &args, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = (int)std::min<int64_t>((args.ntokens + kThreads - 1) / kThreads, 4096);
  a2av_bucket_map_kernel<<<blocks > 0 ? blocks : 1, kThreads, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_bucket_scan(A2AVBucketScanArguments const &args, cudaStream_t stream) {
  a2av_bucket_scan_kernel<<<1, 32, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_bucket_scatter(A2AVBucketScatterArguments const &args, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = (int)std::min<int64_t>((args.ntokens + kThreads - 1) / kThreads, 4096);
  a2av_bucket_scatter_kernel<<<blocks > 0 ? blocks : 1, kThreads, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_lane_wait(uint64_t const *sig, uint64_t const *run_ptr, uint64_t run_host, uint64_t const *kill,
               cudaStream_t stream) {
  a2av_lane_wait_kernel<<<1, 32, 0, stream>>>(sig, run_ptr, run_host, kill);
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_combine_bucket_reduce(
    A2AVBucketReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kThreads = 512;
  FLUX_CHECK(args.n_per % 8 == 0) << "n/n_split must be a multiple of the 8-elem pack width";
  dim3 grid(args.threadblock_count), block(kThreads);
  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [&](auto cdtype) { return cdtype == dtype; },
      [&](auto cdtype) {
        using T = decltype(to_cuda_dtype(cdtype));
        static const bool carveout_set = (prefer_max_shared_once(reinterpret_cast<const void *>(&a2av_combine_bucket_reduce_kernel<T>)), true);
        (void)carveout_set;
        a2av_combine_bucket_reduce_kernel<T><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av bucket reduce: " << dtype; });
  CUDA_CHECK(cudaGetLastError());
}


// ---- sort-free compress-plan derivation -----------------------
// Every ordering in the compress CSRs is arithmetic on the dispatch's stable
// scatter_index (A-order position per copy) plus host cnt/U prefix tables:
// - conv panel position = conv bucket base (seg, expert) + the copy's rank
//   inside its (expert, home) A-suborder = scd - expert_base - home_base;
// - wire slot = per-(seg, token) CSR base + the copy's rank among its token's
//   conv siblings (O(topk) compare loop, conv-position ascending == a
//   stable sort's tie-break);
// - red rows = C'-remapped recv positions (same scd arithmetic restricted to
//   home == me) interleaved own-slots-then-remote-nodes per token.
// Deterministic direct writes; bitwise-identical to the argsort formulation.
namespace {

// deferred verdict set (this layer or an earlier one of the forward): the step is degenerate, the CSRs stay
// empty (the launcher's memsets already zeroed the per-token scratch)
CUTLASS_DEVICE bool
compress_dead(A2AVCompressPlanArguments const &args) {
  return args.verdict != nullptr &&
         *reinterpret_cast<volatile int64_t const *>(args.verdict + Verdict::kDead) != 0;
}

__global__ void
compress_plan_token_kernel(A2AVCompressPlanArguments args) {
  if (compress_dead(args)) {
    return;
  }
  const int64_t ntokens = args.m_full / args.topk;
  const int64_t tpr = ntokens / args.world_size;
  const int L = args.local_world_size;
  const int my_node = args.rank / L;
  const int my_lr = args.rank % L;
  const int NN = args.nnodes;
  for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < ntokens;
       t += (int64_t)gridDim.x * blockDim.x) {
    const int h = (int)(t / tpr);
    const int hn = h / L;
    const int64_t tl = t - (int64_t)h * tpr;
    if (hn != my_node && h % L == my_lr) {
      // source side: I am the gateway lane for home rank h
      const int seg = hn - (hn > my_node ? 1 : 0);
      int cntc = 0;
      int rp = 0;  // pieces: ready piece = max contributor piece
      for (int k = 0; k < args.topk; k++) {
        const int64_t e = args.e_of_copy[t * args.topk + k];
        if ((int)(e / args.ep_nexperts) / L == my_node) {
          cntc++;
          if (args.piece_of_e != nullptr) {
            const int pe = (int)args.piece_of_e[e];
            rp = pe > rp ? pe : rp;
          }
        }
      }
      args.conv_count[(int64_t)seg * tpr + tl] = cntc;
      if (args.wire_piece != nullptr) {
        args.wire_piece[(int64_t)seg * tpr + tl] = rp;
      }
    }
    if (h == args.rank) {
      // destination side: my tokens' per-node contribution flags
      for (int k = 0; k < args.topk; k++) {
        const int64_t e = args.e_of_copy[t * args.topk + k];
        const int on = (int)(e / args.ep_nexperts) / L;
        if (on != my_node) {
          // pieces: store max contributor piece + 1 (single writer per tl;
          // 1 when pieces are off — phase B/C treat nonzero as the flag)
          const int32_t v =
              args.piece_of_e != nullptr ? (int32_t)args.piece_of_e[e] + 1 : 1;
          int32_t &f = args.red_flags[tl * NN + on];
          if (v > f) {
            f = v;
          }
        }
      }
    }
  }
}

// running scans over the small per-token arrays (tens of KB): wire row
// numbering + wire CSR bases (warp 0), red CSR bases (warp 1), remote
// one-cumsum columns (warps 2+, one node column per warp, strided). The
// three phases are data-independent, so they run in parallel warps;
// each is a lane-strided 32-chunk warp scan with a serial running offset
// (the reduce_utils.cuh pattern), fully deterministic.
__global__ void __launch_bounds__(256, 1)
compress_plan_scan_kernel(A2AVCompressPlanArguments args) {
  const int64_t ntokens = args.m_full / args.topk;
  const int64_t tpr = ntokens / args.world_size;
  const int64_t ntok_local = tpr;
  const int NN = args.nnodes;
  const int L = args.local_world_size;
  const int my_node = args.rank / L;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int nwarps = blockDim.x / 32;
  const unsigned kFull = 0xffffffffu;

  // phase A: wire rows over (seg, ready_piece, token asc) — P passes of the
  // exclusive flag scan per segment (P == 1 with pieces off gives the plain
  // (seg, token) order). wire_row_of is -1-prefilled by the
  // launcher, so each pass writes only its own piece's rows; wire_ptr slots
  // follow the SAME renumbering (conv positions are order-independent).
  if (warp == 0) {
    const int P = args.n_pieces > 0 ? args.n_pieces : 1;
    int64_t run_rows = 0;
    int64_t run_cnt = 0;
    if (lane == 0 && args.wire_ptr != nullptr) {
      args.wire_ptr[0] = 0;
    }
    const int64_t n_pad = (tpr + 31) / 32 * 32;
    for (int seg = 0; seg < NN - 1; seg++) {
      const int64_t seg_entry = run_rows;
      for (int p = 0; p < P; p++) {
        if (lane == 0 && args.wire_piece_start != nullptr) {
          args.wire_piece_start[seg * (P + 1) + p] =
              (int32_t)(run_rows - seg_entry);
        }
        for (int64_t i = lane; i < n_pad; i += 32) {
          const int64_t gi = (int64_t)seg * tpr + i;
          int c = 0;
          if (i < tpr) {
            c = args.conv_count[gi];
            if (c > 0 && P > 1 && args.wire_piece != nullptr &&
                args.wire_piece[gi] != p) {
              c = 0;  // another piece's row: skip in this pass
            }
          }
          const int f = c > 0 ? 1 : 0;
          int pf = f;
          int pc = c;
          for (int d = 1; d < 32; d <<= 1) {
            const int uf = __shfl_up_sync(kFull, pf, d);
            const int uc = __shfl_up_sync(kFull, pc, d);
            if (lane >= d) {
              pf += uf;
              pc += uc;
            }
          }
          if (i < tpr && f) {
            const int64_t row_excl = run_rows + pf - f;
            args.wire_row_of[gi] = (int32_t)row_excl;
            args.wire_ptr[row_excl + 1] = (int32_t)(run_cnt + pc);
          }
          run_rows += __shfl_sync(kFull, pf, 31);
          run_cnt += __shfl_sync(kFull, pc, 31);
        }
      }
      if (lane == 0 && args.wire_piece_start != nullptr) {
        args.wire_piece_start[seg * (P + 1) + P] =
            (int32_t)(run_rows - seg_entry);
      }
    }
  }

  // phase B: red_ptr over my tokens (own copies + contributing remote nodes)
  if (warp == 1) {
    const bool dead = compress_dead(args);
    int64_t run = 0;
    if (lane == 0) {
      args.red_ptr[0] = 0;
    }
    const int64_t n_pad = (ntok_local + 31) / 32 * 32;
    for (int64_t tl = lane; tl < n_pad; tl += 32) {
      int cnt = 0;
      if (tl < ntok_local && !dead) {
        const int64_t t = (int64_t)args.rank * tpr + tl;
        for (int k = 0; k < args.topk; k++) {
          const int64_t e = args.e_of_copy[t * args.topk + k];
          if ((int)(e / args.ep_nexperts) / L == my_node) {
            cnt++;
          }
        }
        for (int m = 0; m < NN; m++) {
          cnt += args.red_flags[tl * NN + m] != 0 ? 1 : 0;  // holds piece+1
        }
      }
      int p = cnt;
      for (int d = 1; d < 32; d <<= 1) {
        const int u = __shfl_up_sync(kFull, p, d);
        if (lane >= d) {
          p += u;
        }
      }
      if (tl < ntok_local) {
        args.red_ptr[tl + 1] = (int32_t)(run + p);
      }
      run += __shfl_sync(kFull, p, 31);
    }
  }

  // phase C: remote one-cumsum per node column (exclusive over my tokens),
  // one column per warp among warps 2..nwarps-1, strided
  if (warp < 2) {
    return;
  }
  const int Pc = args.n_pieces > 0 ? args.n_pieces : 1;
  for (int m = warp - 2; m < NN; m += nwarps - 2) {
    if (m == my_node) {
      continue;
    }
    // Pc passes: remote lane rows order (piece, token) — the dest twin of
    // phase A's renumbering (red_flags holds piece+1; pieces off => one
    // pass over value 1, the plain token-ascending order)
    int64_t run = 0;
    const int64_t n_pad = (ntok_local + 31) / 32 * 32;
    for (int pp = 1; pp <= Pc; pp++) {
      for (int64_t tl = lane; tl < n_pad; tl += 32) {
        const int32_t fv = tl < ntok_local ? args.red_flags[tl * NN + m] : 0;
        const int f = fv == pp ? 1 : 0;
        int p = f;
        for (int d = 1; d < 32; d <<= 1) {
          const int u = __shfl_up_sync(kFull, p, d);
          if (lane >= d) {
            p += u;
          }
        }
        if (tl < ntok_local && f) {
          args.rem_pos[tl * NN + m] = (int32_t)(run + p - f);
        }
        run += __shfl_sync(kFull, p, 31);
      }
    }
    __syncwarp();
  }
}

__global__ void
compress_plan_conv_kernel(A2AVCompressPlanArguments args) {
  if (compress_dead(args)) {
    return;
  }
  const int64_t cpr = args.m_full / args.world_size;
  const int64_t ntokens = args.m_full / args.topk;
  const int64_t tpr = ntokens / args.world_size;
  const int L = args.local_world_size;
  const int my_node = args.rank / L;
  const int my_lr = args.rank % L;
  const int64_t node_e0 = (int64_t)my_node * L * args.ep_nexperts;
  for (int64_t c = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; c < args.m_full;
       c += (int64_t)gridDim.x * blockDim.x) {
    const int h = (int)(c / cpr);
    const int hn = h / L;
    if (hn == my_node || h % L != my_lr) {
      continue;
    }
    const int64_t e = args.e_of_copy[c];
    if ((int)(e / args.ep_nexperts) / L != my_node) {
      continue;
    }
    const int seg = hn - (hn > my_node ? 1 : 0);
    const int64_t conv_pos =
        args.conv_base[(int64_t)seg * L * args.ep_nexperts + (e - node_e0)] +
        (int64_t)args.scatter_index[c] - args.expert_base[e] -
        args.home_base[e * args.world_size + h];
    // rank among my token's conv siblings, conv-position ascending (== a
    // stable sort's copy-index tie-break inside each (expert, home) block)
    const int64_t t = c / args.topk;
    int rank_in_group = 0;
    for (int k = 0; k < args.topk; k++) {
      const int64_t c2 = t * args.topk + k;
      if (c2 == c) {
        continue;
      }
      const int64_t e2 = args.e_of_copy[c2];
      if ((int)(e2 / args.ep_nexperts) / L != my_node) {
        continue;
      }
      const int64_t conv_pos2 =
          args.conv_base[(int64_t)seg * L * args.ep_nexperts + (e2 - node_e0)] +
          (int64_t)args.scatter_index[c2] - args.expert_base[e2] -
          args.home_base[e2 * args.world_size + h];
      if (conv_pos2 < conv_pos) {
        rank_in_group++;
      }
    }
    const int64_t tl = t - (int64_t)h * tpr;
    const int32_t wire_row = args.wire_row_of[(int64_t)seg * tpr + tl];
    args.wire_copy[args.wire_ptr[wire_row] + rank_in_group] = (int32_t)conv_pos;
  }
}

__global__ void
compress_plan_red_kernel(A2AVCompressPlanArguments args) {
  if (compress_dead(args)) {
    return;
  }
  const int64_t cpr = args.m_full / args.world_size;
  const int64_t ntokens = args.m_full / args.topk;
  const int64_t tpr = ntokens / args.world_size;
  const int L = args.local_world_size;
  const int my_node = args.rank / L;
  const int NN = args.nnodes;
  for (int64_t tl = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; tl < tpr;
       tl += (int64_t)gridDim.x * blockDim.x) {
    const int64_t t = (int64_t)args.rank * tpr + tl;
    int64_t slot = args.red_ptr[tl];
    for (int k = 0; k < args.topk; k++) {
      const int64_t c = t * args.topk + k;
      const int64_t e = args.e_of_copy[c];
      const int owner = (int)(e / args.ep_nexperts);
      if (owner / L != my_node) {
        continue;
      }
      // recv row under C, then the C' lane remap (intra-lane order preserved)
      const int64_t rows_c = args.my_cnt_cum[e] + (int64_t)args.scatter_index[c] -
                             args.expert_base[e] -
                             args.home_base[e * args.world_size + args.rank];
      args.red_row[slot++] =
          (int32_t)(rows_c - args.recv_off_C[owner] + args.recv_off_Cp[owner]);
    }
    for (int m = 0; m < NN; m++) {
      if (m == my_node || args.red_flags[tl * NN + m] == 0) {
        continue;
      }
      args.red_row[slot++] = (int32_t)(args.rem_base[m] + args.rem_pos[tl * NN + m]);
    }
  }
}

}  // namespace

void
a2av_compress_plan(A2AVCompressPlanArguments const &args, cudaStream_t stream) {
  const int64_t ntokens = args.m_full / args.topk;
  const int64_t tpr = ntokens / args.world_size;
  const int64_t seg_tokens = (int64_t)(args.nnodes - 1) * tpr;
  CUDA_CHECK(cudaMemsetAsync(args.conv_count, 0, seg_tokens * sizeof(int32_t), stream));
  CUDA_CHECK(
      cudaMemsetAsync(args.red_flags, 0, tpr * args.nnodes * sizeof(int32_t), stream));
  // pieces: the P-pass phase A only writes real rows — prefill the misses
  CUDA_CHECK(
      cudaMemsetAsync(args.wire_row_of, 0xFF, seg_tokens * sizeof(int32_t), stream));
  if (args.wire_piece != nullptr) {
    CUDA_CHECK(
        cudaMemsetAsync(args.wire_piece, 0, seg_tokens * sizeof(int32_t), stream));
  }
  constexpr int kThreads = 256;
  compress_plan_token_kernel<<<(int)((ntokens + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      args);
  compress_plan_scan_kernel<<<1, 256, 0, stream>>>(args);
  compress_plan_conv_kernel<<<
      (int)((args.m_full + kThreads - 1) / kThreads),
      kThreads,
      0,
      stream>>>(args);
  compress_plan_red_kernel<<<(int)((tpr + kThreads - 1) / kThreads), kThreads, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

namespace {

// upper_bound over an i64 prefix array: first g with cum[g] > v
CUTLASS_DEVICE int64_t
combine_plan_upper_bound(int64_t const *cum, int64_t n, int64_t v) {
  int64_t lo = 0;
  int64_t hi = n;
  while (lo < hi) {
    const int64_t mid = (lo + hi) >> 1;
    if (cum[mid] <= v) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < n ? lo : n - 1;  // clamp_max(n - 1), torch parity
}

// pack side: gemm row i (A-order) -> recv-panel slot. Identity family of the
// torch path: g = searchsorted(cumA, i, right), sgi = offR_of_A[g] + i -
// offA[g], pack_index[sgi] = i.
__global__ void
combine_plan_pack_kernel(CombinePlanArguments args) {
  const int64_t m = args.m_this_ep_dev != nullptr ? *args.m_this_ep_dev : args.m_this_ep;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < m;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t g = combine_plan_upper_bound(args.cumA, args.nexG, i);
    args.pack_index[args.offR_of_A[g] + i - args.offA[g]] = (int32_t)i;
  }
}

// reduce side: my copy j -> its rank among my copies in (expert, copy) ==
// recv-panel order. r = routing_idx[row0 + j]; e = searchsorted(expert_cum,
// r, right) (or the copy's routing id, the same expert: the scatter index is the
// stable sort of the ids); reduce = my_cum[e] + r - expert_base[e] - h_base[e], with
// expert_base[e] = expert_cum[e - 1] (0 at e == 0).
__global__ void
combine_plan_reduce_kernel(CombinePlanArguments args) {
  // deferred verdict set: degenerate step, every copy reads receive row 0 (in bounds; zero-row receiver)
  const bool dead = args.verdict != nullptr &&
                    *reinterpret_cast<volatile int64_t const *>(args.verdict + Verdict::kDead) != 0;
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < args.cpr;
       j += (int64_t)gridDim.x * blockDim.x) {
    if (dead) {
      args.reduce_index[j] = 0;
      continue;
    }
    const int64_t r = args.routing_idx[args.row0 + j];
    const int64_t e = args.e_of_copy != nullptr
                          ? (int64_t)args.e_of_copy[args.row0 + j]
                          : combine_plan_upper_bound(args.expert_cum, args.nex, r);
    const int64_t e_base = e > 0 ? args.expert_cum[e - 1] : 0;
    args.reduce_index[j] = (int32_t)(args.my_cum[e] + r - e_base - args.h_base[e]);
  }
}

}  // namespace

void
a2av_combine_plan(CombinePlanArguments const &args, cudaStream_t stream) {
  constexpr int kThreads = 256;
  if (args.m_this_ep_dev != nullptr) {
    // the row count lives on the device: the grid is sized from the host hint and strides over the
    // true count (zero rows: every thread exits at once)
    const int64_t hint = std::max<int64_t>(args.m_this_ep, 1);
    combine_plan_pack_kernel<<<
        (int)std::min<int64_t>((hint + kThreads - 1) / kThreads, 1024),
        kThreads,
        0,
        stream>>>(args);
  } else if (args.m_this_ep > 0) {
    combine_plan_pack_kernel<<<
        (int)((args.m_this_ep + kThreads - 1) / kThreads),
        kThreads,
        0,
        stream>>>(args);
  }
  combine_plan_reduce_kernel<<<
      (int)((args.cpr + kThreads - 1) / kThreads),
      kThreads,
      0,
      stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Device-side combine tables (see args/gemm_combine.h)
namespace {

// 128 threads, <= 128 registers each: the overlapped combine derive launches this kernel while the dispatch
// GEMM is resident on every SM and spinning on arrivals. A block that needs more registers than one GEMM block
// leaves free cannot be placed, and the work distributor then holds every later kernel of the same priority
// behind it, including the wire kernels the GEMM waits on (a deadlock); 512 threads x 84 registers do not fit.
constexpr int kCtThreads = 128;

__device__ __forceinline__ void
ct_block_exscan(
    int64_t *v, int n, typename cub::BlockScan<int64_t, kCtThreads>::TempStorage &tmp,
    int64_t *total) {
  const int per = (n + kCtThreads - 1) / kCtThreads;
  const int lo = threadIdx.x * per;
  const int hi = min(lo + per, n);
  int64_t sum = 0;
  for (int i = lo; i < hi; i++) {
    sum += v[i];
  }
  int64_t excl = 0, agg = 0;
  cub::BlockScan<int64_t, kCtThreads>(tmp).ExclusiveSum(sum, excl, agg);
  __syncthreads();
  int64_t run = excl;
  for (int i = lo; i < hi; i++) {
    const int64_t x = v[i];
    v[i] = run;
    run += x;
  }
  if (threadIdx.x == 0 && total != nullptr) {
    *total = agg;
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kCtThreads, 4)
a2av_combine_tables_kernel(A2AVCombineTablesArguments args) {
  extern __shared__ int64_t smem[];
  __shared__ typename cub::BlockScan<int64_t, kCtThreads>::TempStorage scan_tmp;
  __shared__ int64_t s_tot[4];
  const int W = args.W, E = args.ep_nexperts, L = args.L, NN = args.nnodes, rank = args.rank;
  const int64_t nex = (int64_t)W * E, nexG = nex;
  const int ucols = W + NN;
  const int my_node = rank / L, my_lr = rank % L;
  const int64_t ep_start = (int64_t)rank * E;
  int64_t *s_v = smem;              // [nexG]
  int64_t *s_col = s_v + nexG;      // [W] C[s][rank] then scans
  int64_t *s_cp = s_col + W;        // [W]
  const int tid = threadIdx.x;
  auto cnt = [&](int h, int64_t e) -> int64_t { return args.sps[(int64_t)h * nex + e]; };
  auto U_of = [&](int h, int n) -> int64_t { return args.uc[(int64_t)h * ucols + W + n]; };

  // A-order groups g = e_loc*W + h
  for (int64_t g = tid; g < nexG; g += kCtThreads) {
    s_v[g] = cnt((int)(g % W), ep_start + g / W);
  }
  __syncthreads();
  ct_block_exscan(s_v, (int)nexG, scan_tmp, &s_tot[0]);  // s_tot[0] = m_this_ep
  for (int64_t g = tid; g < nexG; g += kCtThreads) {
    args.offA[g] = s_v[g];
    args.cumA[g] = s_v[g] + cnt((int)(g % W), ep_start + g / W);
  }
  __syncthreads();
  // recv-order groups h*E + e_loc -> offR_of_A[e_loc*W + h]
  for (int64_t r = tid; r < nexG; r += kCtThreads) {
    s_v[r] = cnt((int)(r / E), ep_start + r % E);
  }
  __syncthreads();
  ct_block_exscan(s_v, (int)nexG, scan_tmp, nullptr);
  for (int64_t r = tid; r < nexG; r += kCtThreads) {
    args.offR_of_A[(int64_t)(r % E) * W + (r / E)] = s_v[r];
  }
  __syncthreads();
  // column sums over homes: e_base (exclusive), e_cum (inclusive), expert_base (= e_base)
  for (int64_t e = tid; e < nex; e += kCtThreads) {
    int64_t c = 0;
    for (int h = 0; h < W; h++) {
      c += cnt(h, e);
    }
    s_v[e] = c;
  }
  __syncthreads();
  for (int64_t e = tid; e < nex; e += kCtThreads) {
    args.e_cum[e] = s_v[e];  // temporarily the column sum
  }
  __syncthreads();
  ct_block_exscan(s_v, (int)nex, scan_tmp, nullptr);
  for (int64_t e = tid; e < nex; e += kCtThreads) {
    args.e_base[e] = s_v[e];
    args.expert_base[e] = s_v[e];
    args.e_cum[e] = s_v[e] + args.e_cum[e];
  }
  __syncthreads();
  // my_cum / my_cnt_cum: exclusive prefix of my home row
  for (int64_t e = tid; e < nex; e += kCtThreads) {
    s_v[e] = cnt(rank, e);
  }
  __syncthreads();
  ct_block_exscan(s_v, (int)nex, scan_tmp, nullptr);
  for (int64_t e = tid; e < nex; e += kCtThreads) {
    args.my_cum[e] = s_v[e];
    args.my_cnt_cum[e] = s_v[e];
  }
  // h_base[e] = rows of expert e from homes below mine; home_base[e*W + h] exclusive over h
  {
    const int lane = tid & 31, warp = tid >> 5, nwarps = kCtThreads / 32;
    for (int64_t e = warp; e < nex; e += nwarps) {
      int32_t carry = 0;
      for (int h0 = 0; h0 < W; h0 += 32) {
        const int h = h0 + lane;
        int32_t x = h < W ? (int32_t)cnt(h, e) : 0;
        int32_t incl = x;
        for (int o = 1; o < 32; o <<= 1) {
          const int32_t y = __shfl_up_sync(0xffffffffu, incl, o);
          if (lane >= o) {
            incl += y;
          }
        }
        if (h < W) {
          const int32_t excl = carry + incl - x;
          args.home_base[e * W + h] = excl;
          if (h == rank) {
            args.h_base[e] = excl;
          }
        }
        carry += __shfl_sync(0xffffffffu, incl, 31);
      }
    }
  }
  // compress: C[s][rank] per source s (rows homed at rank for s's experts), cp for recv_off_Cp
  for (int s = tid; s < W; s += kCtThreads) {
    int64_t c = 0;
    for (int64_t e = (int64_t)s * E; e < (int64_t)(s + 1) * E; e++) {
      c += cnt(rank, e);
    }
    s_col[s] = c;
    s_cp[s] = (s / L == my_node) ? c : ((s % L == my_lr) ? U_of(rank, s / L) : 0);
  }
  __syncthreads();
  if (tid == 0) {  // own_total before the scans consume s_col
    int64_t own = 0;
    for (int s = my_node * L; s < (my_node + 1) * L; s++) {
      own += s_col[s];
    }
    s_tot[1] = own;
  }
  __syncthreads();
  ct_block_exscan(s_col, W, scan_tmp, nullptr);
  ct_block_exscan(s_cp, W, scan_tmp, nullptr);
  for (int s = tid; s < W; s += kCtThreads) {
    args.recv_off_C[s] = s_col[s];
    args.recv_off_Cp[s] = s_cp[s];
  }
  // conv_base: exclusive prefix over (seg = other nodes ascending, j in [0, L*E)) of
  // cnt(tn*L + my_lr, my_node*L*E + j)
  const int64_t seg_len = (int64_t)L * E;
  const int64_t n_conv = (int64_t)(NN - 1) * seg_len;
  for (int64_t i = tid; i < n_conv; i += kCtThreads) {
    int seg = (int)(i / seg_len);
    const int64_t j = i % seg_len;
    const int tn = seg < my_node ? seg : seg + 1;
    s_v[i] = cnt(tn * L + my_lr, (int64_t)my_node * seg_len + j);
  }
  __syncthreads();
  if (n_conv > 0) {
    ct_block_exscan(s_v, (int)n_conv, scan_tmp, &s_tot[2]);  // conv_total
    for (int64_t i = tid; i < n_conv; i += kCtThreads) {
      args.conv_base[i] = s_v[i];
    }
  } else if (tid == 0) {
    s_tot[2] = 0;
  }
  __syncthreads();
  if (tid == 0) {
    int64_t wire = 0, rem = 0;
    for (int m = 0; m < NN; m++) {
      if (m == my_node) {
        args.rem_base[m] = 0;
        continue;
      }
      wire += U_of(m * L + my_lr, my_node);
      rem += U_of(rank, m);
      args.rem_base[m] = args.recv_off_Cp[m * L + my_lr];
    }
    args.totals[0] = s_tot[0];
    args.totals[1] = s_tot[1];
    args.totals[2] = s_tot[2];
    args.totals[3] = wire;
    args.totals[4] = rem;
    args.totals[5] = args.totals[6] = args.totals[7] = 0;
  }
}

}  // namespace

void
a2av_combine_tables(A2AVCombineTablesArguments const &args, cudaStream_t stream) {
  const int64_t nexG = (int64_t)args.W * args.ep_nexperts;
  const size_t smem = sizeof(int64_t) * (nexG + 2 * args.W);
  FLUX_CHECK_LE(smem, (size_t)96 * 1024) << "combine tables: shared scratch too large";
  if (smem > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(
        a2av_combine_tables_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  }
  a2av_combine_tables_kernel<<<1, kCtThreads, smem, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Combine plan block and msplit problem tables (see args/gemm_combine.h)
namespace {

// One block of 128 threads; launch bounds (128, 8) cap a thread at 64 registers (8K per block), so the
// block fits beside one resident dispatch-GEMM block on every SM (the combine derive overlaps that GEMM).
constexpr int kCpThreads = 128;
constexpr int kCpWarps = kCpThreads / 32;

// exclusive scan of one value per thread over the block; s4 = shared [kCpWarps]
__device__ __forceinline__ long long
cp_exscan(long long v, long long *s4, long long *total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  long long incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const long long y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) {
      incl += y;
    }
  }
  if (lane == 31) {
    s4[warp] = incl;
  }
  __syncthreads();
  long long base = 0, tot = 0;
#pragma unroll
  for (int w = 0; w < kCpWarps; w++) {
    base += w < warp ? s4[w] : 0;
    tot += s4[w];
  }
  __syncthreads();
  *total = tot;
  return base + incl - v;
}

constexpr int kCpSum = 0, kCpMax = 1, kCpMin = 2;
template <int kOp>
__device__ __forceinline__ long long
cp_op(long long a, long long b) {
  return kOp == kCpSum ? a + b : (kOp == kCpMax ? (a > b ? a : b) : (a < b ? a : b));
}
template <int kOp>
__device__ __forceinline__ long long
cp_reduce(long long v, long long *s4) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v = cp_op<kOp>(v, __shfl_xor_sync(0xffffffffu, v, o));
  }
  if (lane == 0) {
    s4[warp] = v;
  }
  __syncthreads();
  long long r = s4[0];
#pragma unroll
  for (int w = 1; w < kCpWarps; w++) {
    r = cp_op<kOp>(r, s4[w]);
  }
  __syncthreads();
  return r;
}

// Thread d (< W) owns home / source rank d. Integer arithmetic only: the send / receive / convergence /
// wire offsets, the overflow maxima, the bucket lane table, remote_rows and node_base.
__global__ void __launch_bounds__(kCpThreads, 8)
a2av_combine_plan_block_kernel(A2AVCombinePlanBlockArguments args) {
  __shared__ long long s4[kCpWarps];
  __shared__ long long s_node_c[kCpThreads];
  __shared__ long long s_part_c[kCpThreads];
  const int W = args.W, E = args.ep_nexperts, L = args.L, NN = args.nnodes, rank = args.rank;
  const int64_t nex = (int64_t)W * E;
  const int64_t ucols = W + NN;
  const int my_node = rank / L, my_lr = rank % L;
  const int64_t ep0 = (int64_t)rank * E;
  const int64_t seg_len = (int64_t)L * E;
  const int64_t node_e0 = (int64_t)my_node * seg_len;
  int32_t const *sps = args.sps;
  int32_t const *uc = args.uc;
  int64_t *blk = args.block;
  CombinePlanLayout const &lay = args.layout;
  const int d = threadIdx.x;
  const bool act = d < W;
  const int dn = d / L, dl = d % L;
  long long rows_to = 0;     // C[rank][d]
  long long rowsum = 0;      // sum_s C[s][d]
  long long cp_in = 0;       // cp(d, rank): receive lane d at me
  long long node_c = 0;      // sum over my node's experts of cnt[d][e]
  long long part_c = 0;      // ... over the experts of my node's ranks below me
  long long owner_rows = 0;  // sum_h C[d][h]
  long long conv_g = 0;      // convergence rows of gateway (dn, dl)
  long long wire_g = 0;      // wire rows of gateway (dn, dl)
  long long dst = 0;         // sum_{s < rank} cp(s, d)
  if (act) {
    int32_t const *row_d = sps + (int64_t)d * nex;
    for (int64_t e = 0; e < nex; e++) {
      rowsum += row_d[e];
    }
    for (int64_t e = 0; e < E; e++) {
      rows_to += row_d[ep0 + e];
    }
    const int64_t part_len = (int64_t)my_lr * E;
    for (int64_t j = 0; j < seg_len; j++) {
      const long long c = row_d[node_e0 + j];
      node_c += c;
      part_c += j < part_len ? c : 0;
    }
    if (dn == my_node) {
      int32_t const *row_r = sps + (int64_t)rank * nex + (int64_t)d * E;
      for (int64_t e = 0; e < E; e++) {
        cp_in += row_r[e];
      }
    } else if (dl == my_lr) {
      cp_in = uc[(int64_t)rank * ucols + W + dn];
    }
    for (int h = 0; h < W; h++) {
      int32_t const *row_h = sps + (int64_t)h * nex + (int64_t)d * E;
      for (int64_t e = 0; e < E; e++) {
        owner_rows += row_h[e];
      }
    }
    for (int tn = 0; tn < NN; tn++) {
      if (tn == dn) {
        continue;
      }
      const int h = tn * L + dl;
      int32_t const *row_h = sps + (int64_t)h * nex + (int64_t)dn * seg_len;
      for (int64_t j = 0; j < seg_len; j++) {
        conv_g += row_h[j];
      }
      wire_g += uc[(int64_t)h * ucols + W + dn];
    }
    for (int s = 0; s < rank; s++) {
      if (s / L == dn) {
        for (int64_t e = 0; e < E; e++) {
          dst += row_d[(int64_t)s * E + e];
        }
      } else if (s % L == dl) {
        dst += uc[(int64_t)d * ucols + W + s / L];
      }
    }
    s_node_c[d] = node_c;
    s_part_c[d] = part_c;
  }
  long long m_total = 0, lane_total = 0, wire_total = 0;
  const long long send_off = cp_exscan(rows_to, s4, &m_total);  // also publishes s_node_c / s_part_c
  const long long lane_off = cp_exscan(cp_in, s4, &lane_total);
  const bool remote = act && dn != my_node;
  const long long remote_rows = cp_reduce<kCpSum>(remote ? rows_to : 0, s4);
  const long long own = cp_reduce<kCpSum>(act && !remote ? cp_in : 0, s4);
  const long long rem = cp_reduce<kCpSum>(remote ? cp_in : 0, s4);
  const long long col_min = cp_reduce<kCpMin>(act ? rowsum : LLONG_MAX, s4);
  const long long col_max = cp_reduce<kCpMax>(act ? rowsum : 0, s4);
  const long long max_send = cp_reduce<kCpMax>(owner_rows, s4);
  const long long max_conv = cp_reduce<kCpMax>(conv_g, s4);
  const long long max_wire = cp_reduce<kCpMax>(wire_g, s4);
  const long long conv_total = cp_reduce<kCpSum>(remote && dl == my_lr ? node_c : 0, s4);
  // wire rows per remote target node, scanned in node order (my node contributes 0)
  long long wrow = 0;
  if (d < NN && d != my_node) {
    wrow = uc[(int64_t)(d * L + my_lr) * ucols + W + my_node];
  }
  const long long wseg = cp_exscan(wrow, s4, &wire_total);
  if (act) {
    blk[lay.send_off + d] = send_off;
    blk[lay.send_rows + d] = rows_to;
    blk[lay.dst_off + d] = dst;
    blk[lay.lane_off + d] = lane_off;
    long long cdst = 0;  // conv_off(dl, tn = dn, my_lr)
    if (dn != my_node) {
      for (int t2 = 0; t2 < dn; t2++) {
        cdst += t2 != my_node ? s_node_c[t2 * L + dl] : 0;
      }
      cdst += s_part_c[d];
    }
    blk[lay.conv_dst + d] = cdst;
    // bucket receiver: lane offsets and the chain position of every source (own node first in
    // (my_lr - dl) order, then the same-lr remote lanes in (my_node - n) order; others 0)
    args.lanes[d] = (int32_t)lane_off;
    int pos = 0;
    if (dn == my_node) {
      pos = (my_lr - dl + L) % L;
    } else if (dl == my_lr) {
      pos = L - 1 + (my_node - dn + NN) % NN;
    }
    args.lanes[W + 1 + d] = pos;
  }
  if (d < NN) {
    blk[lay.wire_rows + d] = wrow;
    if (d != my_node) {
      blk[lay.wire_seg + (d < my_node ? d : d - 1)] = wseg;
    }
  }
  for (int64_t i = d; i < E; i += kCpThreads) {
    long long acc = 0;
    int64_t *nb = blk + lay.node_base + i * (NN + 1);
    for (int n2 = 0; n2 < NN; n2++) {
      nb[n2] = acc;
      for (int lr = 0; lr < L; lr++) {
        acc += sps[(int64_t)(n2 * L + lr) * nex + ep0 + i];
      }
    }
    nb[NN] = acc;
  }
  if (d == 0) {
    blk[lay.send_off + W] = m_total;
    blk[lay.lane_off + W] = lane_total;
    args.lanes[W] = (int32_t)lane_total;
    blk[lay.wire_seg + NN - 1] = wire_total;
    blk[CombinePlanLayout::kM] = m_total;
    blk[CombinePlanLayout::kOwnTotal] = own;
    blk[CombinePlanLayout::kConvTotal] = conv_total;
    blk[CombinePlanLayout::kWireTotal] = wire_total;
    blk[CombinePlanLayout::kRemTotal] = rem;
    blk[CombinePlanLayout::kRemoteRows] = remote_rows;
    blk[CombinePlanLayout::kMaxSend] = max_send;
    blk[CombinePlanLayout::kMaxConv] = max_conv;
    blk[CombinePlanLayout::kMaxWire] = max_wire;
    blk[CombinePlanLayout::kColMin] = col_min;
    blk[CombinePlanLayout::kColMax] = col_max;
    blk[CombinePlanLayout::kSeq] = args.seq;
    for (int i = CombinePlanLayout::kSeq + 1; i < CombinePlanLayout::kHeader; i++) {
      blk[i] = 0;
    }
    // deferred verdict: the forward's host checks of these words, as device-assert bits (a degenerate step's
    // counts are zero and are not checked)
    if (args.verdict != nullptr &&
        *reinterpret_cast<volatile int64_t const *>(args.verdict + Verdict::kDead) == 0) {
      unsigned long long e = 0;
      if (col_min != args.cpr || col_max != args.cpr) {
        e |= (unsigned long long)Verdict::kErrCombineCols;
      }
      if (max_send > args.lim_send) {
        e |= (unsigned long long)Verdict::kErrCombineSend;
      }
      if (args.compress) {
        if (max_conv > args.lim_conv) {
          e |= (unsigned long long)Verdict::kErrCombineConv;
        }
        if (max_wire > args.lim_wire) {
          e |= (unsigned long long)Verdict::kErrCombineWire;
        }
        if (lane_total > args.cpr) {
          e |= (unsigned long long)Verdict::kErrCombineLanes;
        }
        if (conv_total == 0 && wire_total != 0) {
          e |= (unsigned long long)Verdict::kErrCombineUnion;
        }
      }
      if (e != 0) {
        atomicOr(reinterpret_cast<unsigned long long *>(args.verdict + Verdict::kErr), e);
        atomicOr(reinterpret_cast<unsigned long long *>(args.verdict + Verdict::kMask),
                 (unsigned long long)Verdict::kErrorBit);
      }
    }
  }
}

__global__ void __launch_bounds__(kCpThreads, 8)
a2av_msplit_tables_kernel(A2AVMsplitTablesArguments args) {
  __shared__ int32_t s_pos[kMsplitMaxExperts];  // position of expert e inside a wave
  __shared__ int s_dec;
  const int E = args.E, NN = args.NN, nw = args.n_waves;
  auto gate_of = [&](int e) -> int { return args.gate_dev != nullptr ? args.gate_dev[e] : (int)args.gate[e]; };
  if (threadIdx.x == 0) {
    int nu = 0;
    for (int e = 0; args.reorder && e < E; e++) {
      nu += gate_of(e) < 0 ? 1 : 0;
    }
    int iu = 0, ig = 0;
    for (int e = 0; e < E; e++) {
      s_pos[e] = !args.reorder ? e : (gate_of(e) < 0 ? iu++ : nu + ig++);
    }
    // deferred verdict: the wave-adapt rule of the host (gemm_combine.cc forward_impl), on the device
    int dec = 1;
    if (args.dec != nullptr) {
      const long long wire_bytes =
          (args.remote_rows != nullptr ? (long long)*args.remote_rows : 0ll) * (long long)args.adapt_row_bytes;
      dec = (args.adapt_ratio > 0 && args.adapt_reread > (long long)args.adapt_ratio * wire_bytes) ? 0 : 1;
      *args.dec = dec;
    }
    s_dec = dec;
  }
  __syncthreads();
  // collapsed (dec 0): the single-gate GEMM in the wave layout, every row of expert e in problem (0, e)
  const bool collapsed = s_dec == 0;
  const int64_t np = (int64_t)nw * E;
  if (args.out != nullptr) {
    for (int64_t i = threadIdx.x; i < np; i += kCpThreads) {
      const int w = (int)(i / E), e = (int)(i % E);
      const int64_t ip = (int64_t)w * E + s_pos[e];
      int64_t const *nb = args.node_base + (int64_t)e * (NN + 1);
      int64_t off, rows;
      if (collapsed) {
        off = 0;
        rows = w == 0 ? nb[NN] - nb[0] : 0;
      } else {
        off = nb[args.wave_lo[w]];
        rows = nb[args.wave_hi[w]] - off;
      }
      args.out[ip] = (int32_t)rows;
      args.out[np + ip] = (int32_t)off;
      args.out[2 * np + ip] = e;
      args.out[3 * np + ip] = w;
    }
    for (int w = threadIdx.x; w < nw; w += kCpThreads) {
      int ne = 0;
      for (int e = 0; e < E; e++) {
        int64_t const *nb = args.node_base + (int64_t)e * (NN + 1);
        if (collapsed) {
          ne += (w == 0 && nb[NN] > nb[0]) ? 1 : 0;
        } else {
          ne += nb[args.wave_hi[w]] > nb[args.wave_lo[w]] ? 1 : 0;
        }
      }
      args.out[4 * np + w] = ne;
    }
  }
  if (args.wgate != nullptr) {
    if (nw > 0) {
      for (int64_t i = threadIdx.x; i < np; i += kCpThreads) {
        const int w = (int)(i / E), e = (int)(i % E);
        args.wgate[(int64_t)w * E + s_pos[e]] = gate_of(e);
      }
    } else {
      for (int64_t i = threadIdx.x; i < args.n_wgate; i += kCpThreads) {
        args.wgate[i] = gate_of((int)(i % E));
      }
    }
  }
}

}  // namespace

void
a2av_combine_plan_block(A2AVCombinePlanBlockArguments const &args, cudaStream_t stream) {
  FLUX_CHECK_LE(args.W, kCpThreads) << "combine plan block: one thread per rank (at most 128 ranks)";
  FLUX_CHECK_LE(args.nnodes, kA2AVMaxNodes);
  FLUX_CHECK_EQ(args.layout.W, args.W);
  a2av_combine_plan_block_kernel<<<1, kCpThreads, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_msplit_tables(A2AVMsplitTablesArguments const &args, cudaStream_t stream) {
  FLUX_CHECK_LE(args.E, kMsplitMaxExperts) << "msplit tables: too many experts per rank";
  FLUX_CHECK_LE(args.n_waves, kA2AVMaxNodes);
  a2av_msplit_tables_kernel<<<1, kCpThreads, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

// Every kernel of the combine derive chain and the forward's table builders: several of them are first
// launched beside a resident spinning GEMM (the overlapped derive), where a lazy module load would hold
// all new GPU work until that GEMM ends.
void
a2av_combine_derive_preload() {
  cudaFuncAttributes attr;
  CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_tables_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_plan_block_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_msplit_tables_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, combine_plan_pack_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, combine_plan_reduce_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, compress_plan_token_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, compress_plan_scan_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, compress_plan_conv_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, compress_plan_red_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_invert_index_kernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_fold_scales_kernel<__nv_bfloat16>));
  CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_fold_scales_kernel<__half>));
}

// kernel registry (include/flux/cuda/kernel_registry.h): every kernel of this file. CUB's EmptyKernel
// is its PTX-version probe (block-scan headers), never launched; listed so the registry equals the
// library's kernel list.
void
combine_kernels(KernelList &out) {
  out.push_back(ZEPP_KERNEL_ENTRY_T(
      "a2av_combine_pack_kernel<__nv_bfloat16, true>", a2av_combine_pack_kernel<__nv_bfloat16, true>));
  out.push_back(ZEPP_KERNEL_ENTRY_T(
      "a2av_combine_pack_kernel<__nv_bfloat16, false>", a2av_combine_pack_kernel<__nv_bfloat16, false>));
  out.push_back(
      ZEPP_KERNEL_ENTRY_T("a2av_combine_pack_kernel<__half, true>", a2av_combine_pack_kernel<__half, true>));
  out.push_back(
      ZEPP_KERNEL_ENTRY_T("a2av_combine_pack_kernel<__half, false>", a2av_combine_pack_kernel<__half, false>));
  out.push_back(ZEPP_KERNEL_ENTRY_T(
      "a2av_combine_prereduce_kernel<__nv_bfloat16>", a2av_combine_prereduce_kernel<__nv_bfloat16>));
  out.push_back(
      ZEPP_KERNEL_ENTRY_T("a2av_combine_prereduce_kernel<__half>", a2av_combine_prereduce_kernel<__half>));
  out.push_back(ZEPP_KERNEL_ENTRY_T(
      "a2av_combine_reduce_kernel<__nv_bfloat16>", a2av_combine_reduce_kernel<__nv_bfloat16>));
  out.push_back(ZEPP_KERNEL_ENTRY_T("a2av_combine_reduce_kernel<__half>", a2av_combine_reduce_kernel<__half>));
  out.push_back(ZEPP_KERNEL_ENTRY_T(
      "a2av_combine_bucket_reduce_kernel<__nv_bfloat16>", a2av_combine_bucket_reduce_kernel<__nv_bfloat16>));
  out.push_back(ZEPP_KERNEL_ENTRY_T(
      "a2av_combine_bucket_reduce_kernel<__half>", a2av_combine_bucket_reduce_kernel<__half>));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_bucket_map_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_bucket_scan_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_lane_wait_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_bucket_scatter_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_invert_index_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY_T(
      "a2av_fold_scales_kernel<__nv_bfloat16>", a2av_fold_scales_kernel<__nv_bfloat16>));
  out.push_back(ZEPP_KERNEL_ENTRY_T("a2av_fold_scales_kernel<__half>", a2av_fold_scales_kernel<__half>));
  out.push_back(ZEPP_KERNEL_ENTRY(compress_plan_token_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(compress_plan_scan_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(compress_plan_conv_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(compress_plan_red_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(combine_plan_pack_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(combine_plan_reduce_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_combine_tables_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_combine_plan_block_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_msplit_tables_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY_T("EmptyKernel<void>", cub::EmptyKernel<void>));
}

}  // namespace bytedance::flux


