//===- sort_util.cu ----------------------------------------------- C++ ---===//
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

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cutlass/device_kernel.h>

#include <algorithm>
#include <climits>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <iterator>
#include <numeric>
#include <set>

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/reduce_utils.cuh"
#include "flux/flux.h"
#include "flux/utils.h"
#include "sort_util.h"
#include "flux/cuda/kernel_registry.h"
#include "core/verdict.h"

namespace bytedance::flux {
using namespace cute;









// a2av dispatch stage 1: single-pass decode + counts + pack keys. smem holds
// the inclusive splits prefix (thread 0, nexperts is small) and a block-local
// [W,W] histogram flushed with one global atomicAdd per nonzero bin.
__global__ void
a2av_stage1_kernel(A2AVStage1Arguments args) {
  extern __shared__ int a2av_smem[];
  int *splits_cum = a2av_smem;            // [nexperts], inclusive
  int *hist = a2av_smem + args.nexperts;  // [W*W]
  const int WW = args.world_size * args.world_size;
  // chunks/expert_base may be nullptr: the metadata path (host-provided
  // splits_per_source) derives both on the CPU and only needs the decode
  const bool count_chunks = args.chunks != nullptr;
  const bool decode = args.routing_ids == nullptr;
  // deferred verdict: degenerate step (its demands kernel zeroed the counts), no copy is kept
  const bool dead =
      args.verdict != nullptr &&
      *reinterpret_cast<volatile int64_t const *>(args.verdict + Verdict::kDead) != 0;
  if (threadIdx.x == 0 && (decode || args.expert_base != nullptr)) {
    int acc = 0;
    for (int e = 0; e < args.nexperts; e++) {
      acc += args.splits[e];
      splits_cum[e] = acc;
    }
  }
  if (count_chunks) {
    for (int i = threadIdx.x; i < WW; i += blockDim.x) {
      hist[i] = 0;
    }
  }
  __syncthreads();
  if (blockIdx.x == 0 && args.expert_base != nullptr) {
    for (int e = threadIdx.x; e < args.nexperts; e += blockDim.x) {
      args.expert_base[e] = splits_cum[e] - args.splits[e];
    }
  }
  const int64_t my_start = (int64_t)args.rank * args.copies_per_rank;
  for (int64_t p = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; p < args.n_copies;
       p += (int64_t)gridDim.x * blockDim.x) {
    int d = args.scatter_index[p];
    int e;
    if (!decode) {
      e = args.routing_ids[p];
    } else {
      int lo = 0, hi = args.nexperts - 1;  // first e with splits_cum[e] > d
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (splits_cum[mid] > d) {
          hi = mid;
        } else {
          lo = mid + 1;
        }
      }
      e = lo;
    }
    int owner = e / args.ep_nexperts;
    int64_t s = p / args.copies_per_rank;
    args.e_all[p] = e;
    args.s_all[p] = s;
    args.flat_dst[p] = d;
    args.not_mine[p] = dead || owner != args.rank;
    if (args.mine_token != nullptr && !dead) {
      const int L = args.local_world_size;
      const bool keep = ((int)(s / L) == args.node_idx || !args.union_bcast)
                            ? owner == args.rank
                            : owner / L == args.node_idx;
      if (keep) {
        // idempotent: every kept copy of a token stores the same 1
        args.mine_token[p / args.topk] = 1;
      }
    }
    if (count_chunks) {
      atomicAdd(&hist[(int)s * args.world_size + owner], 1);
    }
    int64_t lp = p - my_start;
    if (lp >= 0 && lp < args.copies_per_rank) {
      args.pack_key[lp] = (int64_t)e * args.copies_per_rank + lp;
      if (args.pack_flag != nullptr && !dead) {
        // compress send segment of this copy's destination: my node expanded
        // into L per-rank segments, each remote node one union segment
        // (mirrors the host seg_off_h layout). Seg-major flag write is
        // idempotent — duplicate (token, seg) copies write 1 again.
        const int L = args.local_world_size;
        int nd = owner / L;
        int seg = nd == args.node_idx ? owner - args.node_idx * (L - 1)
                                      : (nd < args.node_idx ? nd : nd + L - 1);
        const int64_t tokens = args.copies_per_rank / args.topk;
        args.pack_flag[(int64_t)seg * tokens + lp / args.topk] = 1;
      }
    }
  }
  if (count_chunks) {
    __syncthreads();
    for (int i = threadIdx.x; i < WW; i += blockDim.x) {
      if (hist[i] != 0) {
        atomicAdd(&args.chunks[i], hist[i]);
      }
    }
  }
}

void
a2av_stage1_impl(A2AVStage1Arguments const &args, cudaStream_t stream) {
  constexpr int kThreads = 256;
  int blocks = (int)std::min<int64_t>((args.n_copies + kThreads - 1) / kThreads, 432);
  blocks = std::max(blocks, 1);
  size_t smem = (args.nexperts + args.world_size * args.world_size) * sizeof(int);
  if (smem > 48 * 1024) {
    // W = 128 pushes the [W,W] block histogram past the 48KB default dynamic
    // smem window (W <= 64 fits); sm80 grants up to ~163KB via the explicit
    // opt-in. Set once per process at the high-water mark.
    static size_t granted = 0;
    if (smem > granted) {
      CUDA_CHECK(cudaFuncSetAttribute(
          a2av_stage1_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          (int)smem));
      granted = smem;
    }
  }
  a2av_stage1_kernel<<<blocks, kThreads, smem, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

// one block per segment; multi-tile exclusive scan with a running carry.
// nseg is tiny (L + NN - 1) and tokens_per_rank is a few thousand, so a
// single 256-thread block per segment is plenty.
__global__ void
a2av_pack_scan_kernel(A2AVPackScanArguments args) {
  constexpr int kThreads = 256;
  using BlockScan = cub::BlockScan<int, kThreads>;
  __shared__ typename BlockScan::TempStorage temp;
  __shared__ int carry;
  if (threadIdx.x == 0) {
    carry = 0;
  }
  __syncthreads();
  const int seg = blockIdx.x;
  int32_t const *flags = args.pack_flag + (int64_t)seg * args.tokens;
  const int64_t base = args.seg_off[seg];
  for (int64_t t0 = 0; t0 < args.tokens; t0 += kThreads) {
    const int64_t t = t0 + threadIdx.x;
    const int f = (t < args.tokens) ? flags[t] : 0;
    int pos, tile_total;
    BlockScan(temp).ExclusiveSum(f, pos, tile_total);
    if (f) {
      args.pack_gather[base + carry + pos] = t;  // carry still pre-tile here
    }
    __syncthreads();  // temp reuse + everyone read carry before the bump
    if (threadIdx.x == 0) {
      carry += tile_total;
    }
    __syncthreads();
  }
}

void
a2av_pack_scan_impl(A2AVPackScanArguments const &args, cudaStream_t stream) {
  if (args.nseg <= 0) {
    return;
  }
  a2av_pack_scan_kernel<<<args.nseg, 256, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

// fused compress consumer build: A row = offA[group] + atomic rank within the
// group. Plain gmem atomics — E * W counters over <= a few hundred thousand
// copies is uncontended enough that a smem pre-histogram (stage-1 style) has
// not been needed.
__global__ void
a2av_consumer_build_kernel(A2AVConsumerBuildArguments args) {
  const int W = args.world_size;
  for (int64_t p = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; p < args.n_copies;
       p += (int64_t)gridDim.x * blockDim.x) {
    if (args.not_mine[p]) {
      continue;
    }
    const int64_t e = args.e_all[p];
    int64_t recv_row = args.c_excl[p / args.topk];
    if (args.mm_off != nullptr) {
      // minimal-move partition: canonical row -> chunk-major row (see .h)
      const int ns = (int)(args.s_all[p] / args.local_world_size);
      if (ns != args.node_idx) {
        const int64_t base = args.mm_base[ns];
        const int64_t x = recv_row - base;
        int lo = (int)args.mm_off[ns], hi = (int)args.mm_off[ns + 1];
        if (hi > lo) {
          // last piece with mm_lo <= x (pieces sorted by canonical start)
          while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (args.mm_lo[mid] <= x) {
              lo = mid;
            } else {
              hi = mid;
            }
          }
          recv_row = base + args.mm_dst[lo] + (x - args.mm_lo[lo]);
        }
      }
    }
    int lane = 0;
    if (args.lane_end != nullptr) {
      // gating lane = first w with recv_row < lane_end[w] (rows always fall
      // below the last lane end == total dedup recv rows)
      int lo = 0, hi = W - 1;
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (args.lane_end[mid] > recv_row) {
          hi = mid;
        } else {
          lo = mid + 1;
        }
      }
      lane = lo;
    }
    if (args.hist_only) {
      // pass 1 : per-(expert, lane) counts only
      atomicAdd(&args.gate_hist[(e - args.ep_start) * W + lane], 1);
      continue;
    }
    int64_t row;
    if (args.offA_lane != nullptr) {
      // pass 3: LANE-keyed A order — rows of expert e are lane-contiguous, so
      // the A-position partition by gating_cumsum is exact (the tile gate's
      // invariant); interior order within a lane stays arbitrary, which no
      // consumer observes
      const int64_t g = (e - args.ep_start) * W + lane;
      row = args.offA_lane[g] + atomicAdd(&args.blk_cnt[g], 1);
    } else {
      // single pass: SOURCE-keyed groups (single-node gating compares
      // per-source boundaries, ssc)
      const int64_t g = (e - args.ep_start) * W + args.s_all[p];
      row = args.offA[g] + atomicAdd(&args.blk_cnt[g], 1);
      if (args.lane_end != nullptr && args.gate_hist != nullptr) {
        atomicAdd(&args.gate_hist[(e - args.ep_start) * W + lane], 1);
      }
    }
    args.gather[row] = (int32_t)recv_row;
    args.scatter[row] = (int32_t)(args.flat_dst[p] - args.expert_base[e]);
  }
}

void
a2av_consumer_build_impl(A2AVConsumerBuildArguments const &args, cudaStream_t stream) {
  constexpr int kThreads = 256;
  int blocks = (int)std::min<int64_t>((args.n_copies + kThreads - 1) / kThreads, 432);
  blocks = std::max(blocks, 1);
  a2av_consumer_build_kernel<<<blocks, kThreads, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

// one thread per expert row: W is tiny (= world size), the serial lane scan is
// nothing next to the launch latency
__global__ void
a2av_gating_cumsum_kernel(A2AVGatingCumsumArguments args) {
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < args.ep_nexperts;
       e += gridDim.x * blockDim.x) {
    int32_t acc = 0;
    const int64_t base = (args.offA_lane != nullptr) ? args.offA[(int64_t)e * args.world_size] : 0;
    for (int w = 0; w < args.world_size; w++) {
      if (args.offA_lane != nullptr) {
        args.offA_lane[(int64_t)e * args.world_size + w] = base + acc;  // exclusive
      }
      acc += args.gate_hist[e * args.world_size + w];
      args.gating_cumsum[e * args.world_size + w] = acc;
    }
  }
}

void
a2av_gating_cumsum_impl(A2AVGatingCumsumArguments const &args, cudaStream_t stream) {
  constexpr int kThreads = 128;
  int blocks = (args.ep_nexperts + kThreads - 1) / kThreads;
  a2av_gating_cumsum_kernel<<<blocks, kThreads, 0, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}











// ---------------------------------------------------------------------------
// Routed metadata in three launches — see sort_util.h for the contract.
// ---------------------------------------------------------------------------

namespace {

constexpr int kMetaThreads = 256;
constexpr int kMetaWarps = kMetaThreads / 32;

// tile b of the flat order: tokens [tok_lo, tok_hi) of source b / tiles_per_src
__device__ __forceinline__ void
meta_tile_bounds(A2AVMetaArguments const &args, int b, int64_t &tok_lo, int64_t &tok_hi) {
  const int src = b / args.tiles_per_src;
  const int t = b - src * args.tiles_per_src;
  const int64_t tt = kA2AVMetaTile / args.topk;
  tok_lo = (int64_t)src * args.tokens_per_rank + (int64_t)t * tt;
  tok_hi = min(tok_lo + tt, (int64_t)(src + 1) * args.tokens_per_rank);
}

__device__ __forceinline__ void
meta_pass1_body(A2AVMetaArguments const &args, int32_t *s_meta) {
  int32_t *s_hist = s_meta;                // [nexperts]
  int32_t *s_uc = s_meta + args.nexperts;  // [W + nnodes]
  const int32_t W = args.world_size, ucols = W + args.nnodes;
  for (int i = threadIdx.x; i < args.nexperts; i += blockDim.x) {
    s_hist[i] = 0;
  }
  for (int i = threadIdx.x; i < ucols; i += blockDim.x) {
    s_uc[i] = 0;
  }
  __syncthreads();
  int64_t tok_lo, tok_hi;
  meta_tile_bounds(args, blockIdx.x, tok_lo, tok_hi);
  for (int64_t i = tok_lo * args.topk + threadIdx.x; i < tok_hi * args.topk; i += blockDim.x) {
    atomicAdd(&s_hist[args.topk_ids[i]], 1);
  }
  for (int64_t t = tok_lo + threadIdx.x; t < tok_hi; t += blockDim.x) {
    uint64_t rank_mask[2] = {0, 0};  // W <= 128: one bit per owner rank
    for (int32_t k = 0; k < args.topk; ++k) {
      const int32_t r = args.topk_ids[t * args.topk + k] / args.ep_nexperts;
      rank_mask[r >> 6] |= (uint64_t)1 << (r & 63);
    }
    uint64_t node_mask = 0;  // nnodes <= 64
    for (int32_t w = 0; w < 2; ++w) {
      for (uint64_t m = rank_mask[w]; m != 0; m &= m - 1) {
        const int32_t d = (w << 6) + __ffsll((long long)m) - 1;
        atomicAdd(&s_uc[d], 1);
        node_mask |= (uint64_t)1 << (d / args.local_world);
      }
    }
    for (uint64_t m = node_mask; m != 0; m &= m - 1) {
      atomicAdd(&s_uc[W + __ffsll((long long)m) - 1], 1);
    }
  }
  __syncthreads();
  int32_t *bh = args.block_hist + (int64_t)blockIdx.x * args.nexperts;
  for (int i = threadIdx.x; i < args.nexperts; i += blockDim.x) {
    bh[i] = s_hist[i];
  }
  int32_t *ub = args.uc_blk + (int64_t)blockIdx.x * ucols;
  for (int i = threadIdx.x; i < ucols; i += blockDim.x) {
    ub[i] = s_uc[i];
  }
}

// the scan (one block). stage != nullptr: block_hist [nb, E] and uc_blk [nb, ucols] are first copied there (one
// coalesced pass), so the serial per-expert walks read shared memory
__device__ __forceinline__ void
meta_scan_body(A2AVMetaArguments const &args, int32_t *stage) {
  using BlockScan = cub::BlockScan<int32_t, kMetaThreads>;
  __shared__ typename BlockScan::TempStorage s_tmp;
  const int32_t E = args.nexperts, W = args.world_size, ucols = W + args.nnodes;
  const int32_t tps = args.tiles_per_src;
  const int32_t *bh = args.block_hist, *ubk = args.uc_blk;
  if (stage != nullptr) {
    const int64_t nb = (int64_t)W * tps;
    for (int64_t i = threadIdx.x; i < nb * E; i += blockDim.x) {
      stage[i] = args.block_hist[i];
    }
    for (int64_t i = threadIdx.x; i < nb * ucols; i += blockDim.x) {
      stage[nb * E + i] = args.uc_blk[i];
    }
    __syncthreads();
    bh = stage;
    ubk = stage + nb * E;
  }
  // per-expert exclusive scan over the blocks in flat order; per-source sums
  for (int32_t e = threadIdx.x; e < E; e += blockDim.x) {
    int32_t run = 0;
    for (int32_t src = 0; src < W; ++src) {
      int32_t s_run = 0;
      for (int32_t t = 0; t < tps; ++t) {
        const int64_t b = (int64_t)src * tps + t;
        const int32_t c = bh[b * E + e];
        args.block_offset[b * E + e] = run;
        run += c;
        s_run += c;
      }
      args.sps[(int64_t)src * E + e] = s_run;
    }
    args.splits[e] = run;
  }
  for (int32_t c = threadIdx.x; c < ucols; c += blockDim.x) {
    for (int32_t src = 0; src < W; ++src) {
      int32_t v = 0;
      for (int32_t t = 0; t < tps; ++t) {
        v += ubk[((int64_t)src * tps + t) * ucols + c];
      }
      args.uc[(int64_t)src * ucols + c] = v;
    }
  }
  __syncthreads();
  // expert_base = exclusive scan of splits (contiguous chunk per thread)
  const int32_t per = (E + kMetaThreads - 1) / kMetaThreads;
  const int32_t lo = threadIdx.x * per, hi = min(lo + per, E);
  int32_t sum = 0;
  for (int32_t e = lo; e < hi; ++e) {
    sum += args.splits[e];
  }
  int32_t excl = 0, total = 0;
  BlockScan(s_tmp).ExclusiveSum(sum, excl, total);
  int32_t r = excl;
  for (int32_t e = lo; e < hi; ++e) {
    args.expert_base[e] = r;
    r += args.splits[e];
  }
  if (threadIdx.x == 0) {
    args.expert_base[E] = total;
  }
}

// pass1 in every block, then the scan in the LAST-arriving block (every block fences its histogram rows and arrives
// on a self-resetting counter; no block waits for another, so partial residency cannot deadlock); the scan reads the
// rows from shared memory when stage_bytes > 0
__global__ void __launch_bounds__(kMetaThreads)
a2av_meta_front_fused_kernel(A2AVMetaArguments args, unsigned *counter, int stage) {
  extern __shared__ int32_t s_meta[];
  __shared__ int s_last;
  meta_pass1_body(args, s_meta);
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    s_last = atomicInc(counter, gridDim.x - 1) == gridDim.x - 1 ? 1 : 0;
  }
  __syncthreads();
  if (!s_last) {
    return;
  }
  __threadfence();
  meta_scan_body(args, stage ? s_meta : nullptr);
}

__global__ void __launch_bounds__(kMetaThreads)
a2av_meta_pass2_kernel(A2AVMetaArguments args) {
  extern __shared__ int32_t s_meta[];  // [kMetaWarps, nexperts]: counts, then running bases
  const int32_t E = args.nexperts;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t lanemask_lt = (1u << lane) - 1;
  int64_t tok_lo, tok_hi;
  meta_tile_bounds(args, blockIdx.x, tok_lo, tok_hi);
  const int64_t lo = tok_lo * args.topk, hi = tok_hi * args.topk;
  for (int i = threadIdx.x; i < kMetaWarps * E; i += blockDim.x) {
    s_meta[i] = 0;
  }
  __syncthreads();
  int32_t *row = s_meta + warp * E;
  const int64_t slice = lo + (int64_t)warp * 256;  // the warp's slice, flat order
  // phase 1: per-(warp, expert) counts over the slice
  for (int s = 0; s < 8; ++s) {
    const int64_t i = slice + s * 32 + lane;
    const uint32_t act = __ballot_sync(0xffffffffu, i < hi);
    if (i < hi) {
      const int32_t e = args.topk_ids[i];
      const uint32_t m = __match_any_sync(act, e);
      if ((__ffs(m) - 1) == lane) {
        atomicAdd(&row[e], __popc(m));
      }
    }
  }
  __syncthreads();
  // exclusive prefix over the warps, folded with the expert and block bases (in place)
  const int32_t *bo = args.block_offset + (int64_t)blockIdx.x * E;
  for (int32_t e = threadIdx.x; e < E; e += blockDim.x) {
    int32_t run = args.expert_base[e] + bo[e];
    for (int w = 0; w < kMetaWarps; ++w) {
      const int32_t c = s_meta[w * E + e];
      s_meta[w * E + e] = run;
      run += c;
    }
  }
  __syncthreads();
  // phase 2: stable emission in flat order; the leader lane advances the running base
  for (int s = 0; s < 8; ++s) {
    const int64_t i = slice + s * 32 + lane;
    const uint32_t act = __ballot_sync(0xffffffffu, i < hi);
    if (i < hi) {
      const int32_t e = args.topk_ids[i];
      const uint32_t m = __match_any_sync(act, e);
      args.scatter_index[i] = row[e] + __popc(m & lanemask_lt);
      __syncwarp(act);
      if ((__ffs(m) - 1) == lane) {
        row[e] += __popc(m);
      }
      __syncwarp(act);
    }
  }
}

}  // namespace

void
a2av_meta_impl(A2AVMetaArguments const &args, cudaStream_t stream) {
  a2av_meta_front_impl(args, stream);
  a2av_meta_pass2_impl(args, stream);
}

void
a2av_meta_front_impl(A2AVMetaArguments const &args, cudaStream_t stream) {
  FLUX_CHECK(args.world_size <= 128) << "owner bitmask is 2x u64";
  FLUX_CHECK(args.nnodes <= 64) << "node bitmask is one u64";
  FLUX_CHECK(args.topk >= 1 && args.topk <= kA2AVMetaTile);
  FLUX_CHECK(args.ntokens == args.tokens_per_rank * args.world_size);
  const int num_blocks = args.world_size * args.tiles_per_src;
  const size_t smem1 =
      sizeof(int32_t) * ((size_t)args.nexperts + args.world_size + args.nnodes);
  const size_t smem2 = sizeof(int32_t) * (size_t)kMetaWarps * args.nexperts;
  static size_t granted2 = 0;  // sm80 opt-in above the 48KB window
  if (smem2 > 48 * 1024 && smem2 > granted2) {
    CUDA_CHECK(cudaFuncSetAttribute(
        a2av_meta_pass2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2));
    granted2 = smem2;
  }
  // one counter word per process (self-reset by the last arrival of every launch)
  static unsigned *counter = [] {
    unsigned *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(p, 0, sizeof(unsigned)));
    return p;
  }();
  const size_t stage_bytes = sizeof(int32_t) * (size_t)num_blocks *
                             ((size_t)args.nexperts + args.world_size + args.nnodes);
  const int stage = stage_bytes <= (size_t)64 * 1024 ? 1 : 0;
  const size_t smem = std::max(smem1, stage ? stage_bytes : (size_t)0);
  static size_t granted = 0;
  if (smem > 48 * 1024 && smem > granted) {
    CUDA_CHECK(cudaFuncSetAttribute(
        a2av_meta_front_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    granted = smem;
  }
  a2av_meta_front_fused_kernel<<<num_blocks, kMetaThreads, smem, stream>>>(args, counter, stage);
  CUDA_CHECK(cudaGetLastError());
}

void
a2av_meta_pass2_impl(A2AVMetaArguments const &args, cudaStream_t stream) {
  const int num_blocks = args.world_size * args.tiles_per_src;
  const size_t smem2 = sizeof(int32_t) * (size_t)kMetaWarps * args.nexperts;
  static size_t granted2 = 0;  // sm80 opt-in above the 48KB window
  if (smem2 > 48 * 1024 && smem2 > granted2) {
    CUDA_CHECK(cudaFuncSetAttribute(
        a2av_meta_pass2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2));
    granted2 = smem2;
  }
  a2av_meta_pass2_kernel<<<num_blocks, kMetaThreads, smem2, stream>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Device-side metadata arena (see sort_util.h)
namespace {

// the one-block planning kernels read the step's counts (sps [W, nex], uc [W, W + NN]) from shared memory when they
// fit (kPlanSmemBudget): one coalesced copy, then every (mostly serial, latency-bound) read hits shared memory; the
// same values
__device__ __forceinline__ void
plan_stage_counts(int32_t const *&sps, int32_t const *&uc, int32_t *dst, int64_t n_sps, int64_t n_uc) {
  for (int64_t i = threadIdx.x; i < n_sps; i += blockDim.x) {
    dst[i] = sps[i];
  }
  for (int64_t i = threadIdx.x; i < n_uc; i += blockDim.x) {
    dst[n_sps + i] = uc[i];
  }
  __syncthreads();
  sps = dst;
  uc = dst + n_sps;
}
constexpr size_t kPlanSmemBudget = 96 * 1024;
// a kernel's static shared memory (bytes); dynamic + static above 48 KB needs the opt-in attribute
template <typename K>
size_t
plan_static_smem(K kernel) {
  cudaFuncAttributes at{};
  CUDA_CHECK(cudaFuncGetAttributes(&at, kernel));
  return at.sharedSizeBytes;
}
template <typename K>
void
plan_dyn_smem_optin(K kernel, size_t static_bytes, size_t dyn_bytes) {
  if (static_bytes + dyn_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_bytes));
  }
}

constexpr int kArenaThreads = 512;
constexpr int kArenaMaxL = 8;

// in-place exclusive scan of v[0, n) by the whole block (chunk per thread)
__device__ __forceinline__ void
arena_block_exscan(
    int64_t *v, int n, typename cub::BlockScan<int64_t, kArenaThreads>::TempStorage &tmp) {
  const int per = (n + kArenaThreads - 1) / kArenaThreads;
  const int lo = threadIdx.x * per;
  const int hi = min(lo + per, n);
  int64_t sum = 0;
  for (int i = lo; i < hi; i++) {
    sum += v[i];
  }
  int64_t excl = 0;
  cub::BlockScan<int64_t, kArenaThreads>(tmp).ExclusiveSum(sum, excl);
  __syncthreads();
  int64_t run = excl;
  for (int i = lo; i < hi; i++) {
    const int64_t x = v[i];
    v[i] = run;
    run += x;
  }
  __syncthreads();
}

// lb_minmove water-fill of the n -> m stream (reference: python/zepp/routing.py _waterfill, same piece order).
// Fills bd[0..L] (chunk bounds). With o_lo != nullptr also emits the pieces at
// [pbase, ...) using bd_ref (the final bounds) for the destination coordinate.
__device__ int64_t
arena_waterfill(
    A2AVMetaArenaArguments const &args, int n, int m, int L, int64_t *bd, int64_t const *bd_ref,
    int64_t *o_lo, int64_t *o_hi, int64_t *o_dst, int64_t pbase) {
  int64_t V[kArenaMaxL], cap[kArenaMaxL], keep[kArenaMaxL], room[kArenaMaxL], off[kArenaMaxL];
  int64_t cst[kArenaMaxL + 1];
  const int W = args.W, ucols = W + args.nnodes;
  int64_t tot = 0;
  cst[0] = 0;
  for (int k = 0; k < L; k++) {
    V[k] = args.uc[(int64_t)(n * L + k) * ucols + W + m];
    tot += V[k];
    cst[k + 1] = cst[k] + V[k];
  }
  for (int k = 0; k < L; k++) {
    cap[k] = tot / L + (k < tot % L ? 1 : 0);
    keep[k] = min(V[k], cap[k]);
    room[k] = cap[k] - keep[k];
    off[k] = keep[k];
  }
  int64_t pc = 0;
  int imp = 0;
  for (int sl = 0; sl < L; sl++) {
    if (keep[sl] > 0) {
      if (o_lo != nullptr) {
        o_lo[pbase + pc] = cst[sl];
        o_hi[pbase + pc] = cst[sl] + keep[sl];
        o_dst[pbase + pc] = bd_ref[sl];
      }
      pc++;
    }
    int64_t j = keep[sl];
    while (j < V[sl]) {
      while (imp < L && room[imp] == 0) {
        imp++;
      }
      if (imp >= L) {
        __trap();  // the water-fill ran out of room
      }
      const int64_t take = min(V[sl] - j, room[imp]);
      if (o_lo != nullptr) {
        o_lo[pbase + pc] = cst[sl] + j;
        o_hi[pbase + pc] = cst[sl] + j + take;
        o_dst[pbase + pc] = bd_ref[imp] + off[imp];
      }
      off[imp] += take;
      room[imp] -= take;
      j += take;
      pc++;
    }
  }
  bd[0] = 0;
  for (int k = 0; k < L; k++) {
    bd[k + 1] = bd[k] + off[k];
  }
  return pc;
}

__global__ void __launch_bounds__(kArenaThreads, 1)
a2av_meta_arena_kernel(A2AVMetaArenaArguments args) {
  extern __shared__ int64_t smem[];
  __shared__ typename cub::BlockScan<int64_t, kArenaThreads>::TempStorage scan_tmp;
  const int W = args.W, E = args.ep_nexperts, nex = args.nexperts, L = args.L, NN = args.nnodes;
  const int ucols = W + NN;
  const int64_t nexG = (int64_t)E * W;
  int64_t *s_v = smem;                      // [nexG] scan scratch
  int64_t *s_recv = s_v + nexG;             // [W + 1] recv_off_u
  int64_t *s_bound = s_recv + W + 1;        // [NN * (L + 1)] chunk bounds, target my node
  int64_t *s_pc = s_bound + NN * (L + 1);   // [NN + 1] piece counts -> offsets
  const int tid = threadIdx.x;
  if (args.only_if_dead != nullptr && *reinterpret_cast<volatile int64_t const *>(args.only_if_dead) == 0) {
    return;  // the speculative arena of a live step stands (written from the same counts)
  }
  if (args.stage) {
    plan_stage_counts(args.sps, args.uc, reinterpret_cast<int32_t *>(s_pc + NN + 1), (int64_t)W * nex,
                      (int64_t)W * (W + NN));
  }
  auto cnt = [&](int s, int64_t e) -> int64_t { return args.sps[(int64_t)s * nex + e]; };
  auto u_of = [&](int s, int d) -> int64_t { return args.uc[(int64_t)s * ucols + d]; };
  auto U_of = [&](int s, int n) -> int64_t { return args.uc[(int64_t)s * ucols + W + n]; };
  int64_t *cumA = reinterpret_cast<int64_t *>(args.arena);
  int64_t *offA = cumA + nexG;
  int64_t *offR_of_A = offA + nexG;
  int64_t *expert_base = offR_of_A + nexG;
  int32_t *ssc = reinterpret_cast<int32_t *>(expert_base + nex);

  // (1) A-order groups g = e_loc*W + s: offA exclusive, cumA inclusive
  for (int64_t g = tid; g < nexG; g += kArenaThreads) {
    s_v[g] = cnt((int)(g % W), args.ep_start + g / W);
  }
  __syncthreads();
  arena_block_exscan(s_v, (int)nexG, scan_tmp);
  for (int64_t g = tid; g < nexG; g += kArenaThreads) {
    offA[g] = s_v[g];
    cumA[g] = s_v[g] + cnt((int)(g % W), args.ep_start + g / W);
  }
  __syncthreads();
  // (2) recv-order groups h = s*E + e_loc; offR_of_A[e_loc*W + s] = offR[h]
  for (int64_t h = tid; h < nexG; h += kArenaThreads) {
    s_v[h] = cnt((int)(h / E), args.ep_start + h % E);
  }
  __syncthreads();
  arena_block_exscan(s_v, (int)nexG, scan_tmp);
  for (int64_t h = tid; h < nexG; h += kArenaThreads) {
    offR_of_A[(int64_t)(h % E) * W + (h / E)] = s_v[h];
  }
  __syncthreads();
  // (3) expert_base[e] = exclusive prefix of the column sums over all experts
  for (int64_t e = tid; e < nex; e += kArenaThreads) {
    int64_t c = 0;
    for (int s = 0; s < W; s++) {
      c += cnt(s, e);
    }
    s_v[e] = c;
  }
  __syncthreads();
  arena_block_exscan(s_v, nex, scan_tmp);
  for (int64_t e = tid; e < nex; e += kArenaThreads) {
    expert_base[e] = s_v[e];
  }
  // (4) ssc[e_loc*W + s]: inclusive cumsum over sources, one warp per expert
  {
    const int lane = tid & 31, warp = tid >> 5, nwarps = kArenaThreads / 32;
    for (int e_loc = warp; e_loc < E; e_loc += nwarps) {
      int32_t carry = 0;
      for (int s0 = 0; s0 < W; s0 += 32) {
        const int s = s0 + lane;
        int32_t x = s < W ? (int32_t)cnt(s, args.ep_start + e_loc) : 0;
        for (int o = 1; o < 32; o <<= 1) {
          const int32_t y = __shfl_up_sync(0xffffffffu, x, o);
          if (lane >= o) {
            x += y;
          }
        }
        if (s < W) {
          ssc[(int64_t)e_loc * W + s] = carry + x;
        }
        carry += __shfl_sync(0xffffffffu, x, 31);
      }
    }
  }
  __syncthreads();

  // ---- compress block: seg_off, (NN > 1) gate_q, (lb_minmove) receiver piece tables
  int64_t *cmp = reinterpret_cast<int64_t *>(args.arena + args.compress_off);
  const int nseg = L + NN - 1;
  int64_t *seg_off = cmp;
  int64_t *gate_q = cmp + nseg + 1;
  int64_t *mm = gate_q + (int64_t)E * (W + 1);
  const int my_node = args.my_node, rank = args.rank;
  auto region_rows = [&](int s, int d) -> int64_t {
    return (s / L != d / L) ? U_of(s, d / L) : u_of(s, d);
  };
  if (tid == 0) {
    seg_off[0] = 0;
    int seg = 0;
    for (int n = 0; n < NN; n++) {
      if (n == my_node) {
        for (int dl = 0; dl < L; dl++, seg++) {
          seg_off[seg + 1] = seg_off[seg] + u_of(rank, n * L + dl);
        }
      } else {
        seg_off[seg + 1] = seg_off[seg] + U_of(rank, n);
        seg++;
      }
    }
  }
  for (int s = tid; s < W; s += kArenaThreads) {
    s_recv[s] = region_rows(s, rank);
  }
  __syncthreads();
  arena_block_exscan(s_recv, W, scan_tmp);  // recv_off_u
  if (NN > 1) {
    const int64_t P = (int64_t)NN * 2 * L;
    int64_t *o_off = mm;
    int64_t *o_base = o_off + NN + 1;
    int64_t *o_lo = o_base + NN;
    int64_t *o_hi = o_lo + P;
    int64_t *o_dst = o_hi + P;
    // pass 1: chunk bounds of every ns -> my_node stream (+ piece counts)
    if (tid < NN) {
      const int ns = tid;
      int64_t *bd = s_bound + ns * (L + 1);
      int64_t npieces = 0;
      if (ns == my_node) {
        for (int k = 0; k <= L; k++) {
          bd[k] = 0;
        }
      } else if (args.lb_minmove) {
        npieces = arena_waterfill(args, ns, my_node, L, bd, nullptr, nullptr, nullptr, nullptr, 0);
      } else {
        int64_t total = 0;
        for (int k = 0; k < L; k++) {
          total += U_of(ns * L + k, my_node);
        }
        for (int k = 0; k <= L; k++) {
          bd[k] = (total / L) * k + min((int64_t)k, total % L);
        }
      }
      s_pc[ns] = npieces;
    }
    __syncthreads();
    if (args.lb_minmove) {
      if (tid == 0) {
        int64_t pc = 0;
        for (int ns = 0; ns < NN; ns++) {
          o_off[ns] = pc;
          o_base[ns] = s_recv[ns * L];
          const int64_t c = s_pc[ns];
          s_pc[ns] = pc;
          pc += c;
        }
        o_off[NN] = pc;
      }
      __syncthreads();
      // clear the unused tail of the piece tables: the arena is a function of this step's counts only
      for (int64_t i = tid; i < P; i += kArenaThreads) {
        if (i >= o_off[NN]) {
          o_lo[i] = 0;
          o_hi[i] = 0;
          o_dst[i] = 0;
        }
      }
      __syncthreads();
      // pass 2: emit the pieces in canonical order at o_off[ns]
      if (tid < NN && tid != my_node) {
        const int ns = tid;
        int64_t bd_scratch[kArenaMaxL + 1];
        arena_waterfill(
            args, ns, my_node, L, bd_scratch, s_bound + ns * (L + 1), o_lo, o_hi, o_dst, s_pc[ns]);
      }
      __syncthreads();
    }
    // gating queries: gate_q[e*(W+1)] = e*R_key; gate_q[e*(W+1)+1+s] = e*R_key + lane_end[s]
    for (int64_t i = tid; i < (int64_t)E * (W + 1); i += kArenaThreads) {
      const int64_t e = i / (W + 1);
      const int c = (int)(i % (W + 1));
      if (c == 0) {
        gate_q[i] = e * args.recv_key;
        continue;
      }
      const int s = c - 1;
      int64_t lane_end;
      if (s / L == my_node) {
        lane_end = s_recv[s] + region_rows(s, rank);
      } else {
        const int ns = s / L, gl = s % L;
        lane_end = s_recv[ns * L] + s_bound[ns * (L + 1) + gl + 1];
      }
      gate_q[i] = e * args.recv_key + lane_end;
    }
  }
}

}  // namespace

void
a2av_meta_arena_impl(A2AVMetaArenaArguments const &args, cudaStream_t stream) {
  FLUX_CHECK(args.L >= 1 && args.L <= kArenaMaxL) << "device meta arena: L out of range";
  FLUX_CHECK(args.nnodes >= 1 && args.nnodes <= kArenaThreads) << "device meta arena: nnodes";
  FLUX_CHECK_EQ((int64_t)args.nexperts, (int64_t)args.ep_nexperts * args.W)
      << "device meta arena: nexperts must equal ep_nexperts * W";
  const int64_t nexG = (int64_t)args.ep_nexperts * args.W;
  size_t smem = sizeof(int64_t) *
                (nexG + args.W + 1 + (int64_t)args.nnodes * (args.L + 1) + args.nnodes + 1);
  FLUX_CHECK_LE(smem, (size_t)96 * 1024) << "device meta arena: shared scratch too large";
  A2AVMetaArenaArguments a = args;
  const size_t stage_bytes =
      sizeof(int32_t) * ((size_t)args.W * args.nexperts + (size_t)args.W * (args.W + args.nnodes));
  static const size_t static_bytes = plan_static_smem(a2av_meta_arena_kernel);
  if (static_bytes + smem + stage_bytes <= kPlanSmemBudget) {
    a.stage = 1;
    smem += stage_bytes;
  }
  plan_dyn_smem_optin(a2av_meta_arena_kernel, static_bytes, smem);
  a2av_meta_arena_kernel<<<1, kArenaThreads, smem, stream>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Device demand check (see sort_util.h)
namespace {

constexpr int kDemThreads = 512;

__device__ __forceinline__ void
dem_max(int64_t *slot, int64_t v) {
  atomicMax(reinterpret_cast<unsigned long long *>(slot), (unsigned long long)v);
}

__global__ void __launch_bounds__(kDemThreads, 1)
a2av_demands_kernel(A2AVDemandsArguments args) {
  extern __shared__ int32_t chunks[];  // [W * W] rows source s -> destination d
  __shared__ int64_t red[8];
  __shared__ int s_dead;
  const int W = args.W, E = args.ep_nexperts, L = args.L, NN = args.nnodes;
  const int64_t nex = (int64_t)W * E;
  const int ucols = W + NN;
  const int tid = threadIdx.x;
  if (args.stage) {
    plan_stage_counts(args.sps, args.uc, chunks + W * W, (int64_t)W * nex, (int64_t)W * ucols);
  }
  auto u_of = [&](int s, int d) -> int64_t { return args.uc[(int64_t)s * ucols + d]; };
  auto U_of = [&](int s, int n) -> int64_t { return args.uc[(int64_t)s * ucols + W + n]; };
  if (tid < 8) {
    red[tid] = 0;
  }
  for (int i = tid; i < W * W; i += kDemThreads) {
    const int s = i / W, d = i % W;
    int32_t acc = 0;
    for (int64_t e = (int64_t)d * E; e < (int64_t)(d + 1) * E; e++) {
      acc += args.sps[(int64_t)s * nex + e];
    }
    chunks[i] = acc;
  }
  __syncthreads();
  for (int d = tid; d < W; d += kDemThreads) {  // recv_rows: rows landing on destination d
    int64_t col = 0;
    for (int s = 0; s < W; s++) {
      col += chunks[s * W + d];
    }
    dem_max(&red[0], col);
  }
  for (int i = tid; i < W * W; i += kDemThreads) {  // pair_rows
    dem_max(&red[6], chunks[i]);
  }
  if (NN > 1) {
    for (int d = tid; d < W; d += kDemThreads) {  // dispatch_recv: node-union regions per destination
      int64_t col = 0;
      for (int s = 0; s < W; s++) {
        col += (s / L == d / L) ? u_of(s, d) : U_of(s, d / L);
      }
      dem_max(&red[1], col);
    }
    for (int idx = tid; idx < NN * L; idx += kDemThreads) {  // stage (n, k) and the relay chunk max
      const int n = idx / L, k = idx % L;
      int64_t st = 0, mx = 0;
      for (int ns = 0; ns < NN; ns++) {
        if (ns == n) {
          continue;
        }
        int64_t T = 0;
        for (int l = 0; l < L; l++) {
          T += U_of(ns * L + l, n);
        }
        const int64_t cr = T / L + (k < T % L ? 1 : 0);
        st += cr;
        mx = max(mx, cr);
      }
      dem_max(&red[2], st);
      dem_max(&red[3], mx);
    }
    for (int idx = tid; idx < L * NN; idx += kDemThreads) {  // conv (dl, n2): rows from other nodes
      const int dl = idx / NN, n2 = idx % NN;
      int64_t acc = 0;
      for (int tn = 0; tn < NN; tn++) {
        if (tn == n2) {
          continue;
        }
        for (int ls = 0; ls < L; ls++) {
          acc += chunks[(tn * L + dl) * W + n2 * L + ls];
        }
      }
      dem_max(&red[4], acc);
    }
    for (int idx = tid; idx < L * NN; idx += kDemThreads) {  // wire (l, n): unions from other nodes
      const int l = idx / NN, n = idx % NN;
      int64_t acc = 0;
      for (int ns = 0; ns < NN; ns++) {
        if (ns != n) {
          acc += U_of(ns * L + l, n);
        }
      }
      dem_max(&red[5], acc);
    }
  }
  __syncthreads();
  if (tid == 0) {
    const int64_t recv = max(red[0], (int64_t)1);
    int64_t drecv, stage, relay, conv, wire;
    if (NN == 1) {
      drecv = recv;
      stage = relay = conv = wire = 1;
    } else {
      drecv = max(max(recv, red[1]), (int64_t)1);
      stage = max(red[2], (int64_t)1);
      relay = max((int64_t)args.relay_slots * red[3], (int64_t)1);
      conv = max(red[4], (int64_t)1);
      wire = max(red[5], (int64_t)1);
    }
    const int64_t pair = max(red[6], (int64_t)1);
    args.out[0] = recv;
    args.out[1] = drecv;
    args.out[2] = stage;
    args.out[3] = relay;
    args.out[4] = conv;
    args.out[5] = wire;
    args.out[6] = pair;
    int64_t mask = 0;
    if (args.direct) {
      mask |= (recv > args.caps[0]) ? 1 : 0;
      mask |= (pair > args.caps[7]) ? (1 << 7) : 0;
    } else {
      mask |= (recv > args.caps[0]) ? 1 : 0;
      mask |= (recv > args.caps[4]) ? (1 << 1) : 0;
      mask |= (drecv > args.caps[1]) ? (1 << 2) : 0;
      mask |= (stage > args.caps[2]) ? (1 << 3) : 0;
      mask |= (relay > args.caps[3]) ? (1 << 4) : 0;
      mask |= (conv > args.caps[5]) ? (1 << 5) : 0;
      mask |= (wire > args.caps[6]) ? (1 << 6) : 0;
    }
    args.out[7] = mask;
    if (args.verdict != nullptr) {
      // deferred verdict: sticky OR, first violating layer latches its demands
      unsigned long long *vm = reinterpret_cast<unsigned long long *>(args.verdict + Verdict::kMask);
      const unsigned long long m =
          (unsigned long long)mask | (args.force ? (unsigned long long)Verdict::kForceBit : 0ull);
      const unsigned long long prev = atomicOr(vm, m);
      if (m != 0 &&
          atomicCAS(reinterpret_cast<unsigned long long *>(args.verdict + Verdict::kLatch), 0ull,
                    (unsigned long long)(args.layer + 1)) == 0ull) {
        for (int i = 0; i < 7; i++) {
          args.verdict[Verdict::kDem + i] = args.out[i];
        }
      }
      s_dead = (prev | m) != 0 ? 1 : 0;
    }
  }
  if (args.verdict == nullptr) {
    return;
  }
  __syncthreads();
  if (s_dead) {
    // degenerate step: every builder after this kernel sizes its work from zero counts
    const int64_t n_sps = (int64_t)W * nex, n_uc = (int64_t)W * ucols;
    for (int64_t i = tid; i < n_sps; i += kDemThreads) {
      args.zero_sps[i] = 0;
    }
    for (int64_t i = tid; i < n_uc; i += kDemThreads) {
      args.zero_uc[i] = 0;
    }
    for (int64_t i = tid; i < nex; i += kDemThreads) {
      args.zero_splits[i] = 0;
    }
    if (args.zero_plan != nullptr) {
      for (int64_t i = tid; i < args.plan_words; i += kDemThreads) {
        if (i != DispatchPlan::kSeq) {  // the sequence word is not a count: it keeps its value
          args.zero_plan[i] = 0;
        }
      }
    }
    if (tid == 0) {
      args.verdict[Verdict::kDead] = 1;
    }
  }
}

}  // namespace

void
a2av_demands_impl(A2AVDemandsArguments const &args, cudaStream_t stream) {
  FLUX_CHECK(args.W >= 1 && args.W <= 256) << "device demands: W out of range";
  size_t smem = sizeof(int32_t) * (size_t)args.W * args.W;
  A2AVDemandsArguments a = args;
  const size_t stage_bytes = sizeof(int32_t) * ((size_t)args.W * args.W * args.ep_nexperts +
                                                (size_t)args.W * (args.W + args.nnodes));
  static const size_t static_bytes = plan_static_smem(a2av_demands_kernel);
  if (static_bytes + smem + stage_bytes <= kPlanSmemBudget) {
    a.stage = 1;
    smem += stage_bytes;
  }
  plan_dyn_smem_optin(a2av_demands_kernel, static_bytes, smem);
  a2av_demands_kernel<<<1, kDemThreads, smem, stream>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Dispatch plan block (sort_util.h DispatchPlan): one block computes, for this rank, every size and
// offset the dispatch wire needs (reference: python/zepp/routing.py dispatch_plan_ref,
// tests/test_dispatch_plan_device.py).
namespace {

constexpr int kPlanThreads = 256;
constexpr int kPlanMaxL = 8;
constexpr int kPlanMaxNN = 16;

__global__ void __launch_bounds__(kPlanThreads, 1)
a2av_dispatch_plan_kernel(A2AVDispatchPlanArguments args) {
  using DP = DispatchPlan;
  const int W = args.W, L = args.L, NN = args.nnodes, my_node = args.my_node, rank = args.rank;
  const int my_lr = rank - my_node * L;
  const int ucols = W + NN;
  const int64_t nex = args.nexperts, E = args.ep_nexperts;
  const int tid = threadIdx.x;
  int64_t *plan = args.plan;
  __shared__ int64_t s_bd[kPlanMaxNN * kPlanMaxNN * (kPlanMaxL + 1)];  // chunk bounds of every n -> m stream
  __shared__ int64_t s_keep[kPlanMaxNN * kPlanMaxL];                  // my_node -> tn: kept rows per relay
  __shared__ int64_t s_imp[kPlanMaxNN * kPlanMaxL];                   // my_node -> tn: imported rows per relay
  __shared__ int64_t s_pz[kPlanMaxNN * 2 * kPlanMaxL * 5];            // my_node -> tn pieces {sl, j_lo, j_hi, k, off}
  __shared__ int s_npz[kPlanMaxNN];
  __shared__ int64_t s_seg[kPlanMaxNN + kPlanMaxL];
  __shared__ unsigned long long s_err, s_m;
  __shared__ long long s_maxcol, s_maxstage, s_maxrelay;
  // prefix tables (built in parallel once, so the lookups below are table reads, not O(W) / O(NN L) loops):
  // s_rp[dl * (W + 1) + s] = sum_{sq < s} region(sq, my_node * L + dl) (only this node's destinations are queried)
  // s_psb[plr * NN + tn]   = local peer plr's send-segment base for target node tn
  __shared__ int64_t s_rp[kPlanMaxL * (kPlanMaxL * kPlanMaxNN + 1)];
  __shared__ int64_t s_psb[kPlanMaxL * kPlanMaxNN];

  if (args.stage) {
    extern __shared__ int32_t plan_dsm[];
    plan_stage_counts(args.sps, args.uc, plan_dsm, (int64_t)W * nex, (int64_t)W * ucols);
  }
  auto u = [&](int s, int d) -> int64_t { return args.uc[(int64_t)s * ucols + d]; };
  auto U = [&](int s, int n) -> int64_t { return args.uc[(int64_t)s * ucols + W + n]; };
  // dedup recv region of source s at destination d (remote-node sources hold their node union)
  auto region = [&](int s, int d) -> int64_t { return (s / L != d / L) ? U(s, d / L) : u(s, d); };
  auto recv_off_of_u_scan = [&](int s, int d) -> int64_t {
    int64_t acc = 0;
    for (int sq = 0; sq < s; sq++) {
      acc += region(sq, d);
    }
    return acc;
  };
  // d is always a rank of this node (round 0, self, forwards): the table built after the checks
  auto recv_off_of_u = [&](int s, int d) -> int64_t { return s_rp[(int64_t)(d - my_node * L) * (W + 1) + s]; };
  auto chunk_rows = [&](int n, int m, int k) -> int64_t {
    if (n == m) {
      return 0;
    }
    const int64_t *bd = &s_bd[((int64_t)n * NN + m) * (L + 1)];
    return bd[k + 1] - bd[k];
  };
  // a local peer's send-segment base for target node tn (its send layout is the replicated u / U rows)
  auto peer_seg_base_scan = [&](int plr, int tn) -> int64_t {
    const int prank = my_node * L + plr;
    int64_t acc = 0;
    for (int n = 0; n < tn; n++) {
      if (n == my_node) {
        for (int dl = 0; dl < L; dl++) {
          acc += u(prank, n * L + dl);
        }
      } else {
        acc += U(prank, n);
      }
    }
    return acc;
  };

  auto peer_seg_base = [&](int plr, int tn) -> int64_t { return s_psb[plr * NN + tn]; };

  const int64_t words = DP::words(L, NN);
  for (int64_t i = tid; i < words; i += blockDim.x) {
    plan[i] = 0;
  }
  if (tid == 0) {
    s_err = 0;
    s_m = 0;
    s_maxcol = 0;
    s_maxstage = 0;
    s_maxrelay = 0;
  }
  __syncthreads();

  // rows this rank computes (a warp-reduced sum: one shared atomic per warp instead of one per entry), and the
  // unique-count consistency checks
  {
    unsigned long long msum = 0;
    for (int64_t i = tid; i < (int64_t)W * E; i += blockDim.x) {
      const int s = (int)(i / E);
      msum += (unsigned long long)args.sps[(int64_t)s * nex + args.ep_start + i % E];
    }
    for (int o = 16; o > 0; o >>= 1) {
      msum += __shfl_xor_sync(0xffffffffu, msum, o);
    }
    if ((tid & 31) == 0 && msum != 0) {
      atomicAdd(&s_m, msum);
    }
  }
  // the prefix tables (read after the barrier that ends the water-fill)
  for (int i = tid; i < L * (W + 1); i += blockDim.x) {
    const int dl = i / (W + 1), s = i - dl * (W + 1);
    s_rp[i] = recv_off_of_u_scan(s, my_node * L + dl);
  }
  for (int i = tid; i < L * NN; i += blockDim.x) {
    s_psb[i] = peer_seg_base_scan(i / NN, i % NN);
  }
  for (int64_t i = tid; i < (int64_t)W * W; i += blockDim.x) {
    const int s = (int)(i / W), d = (int)(i % W);
    int64_t cv = 0;
    for (int64_t e = (int64_t)d * E; e < (int64_t)(d + 1) * E; e++) {
      cv += args.sps[(int64_t)s * nex + e];
    }
    const int64_t uv = u(s, d);
    if (!(uv >= 0 && uv <= cv && (uv > 0) == (cv > 0))) {
      atomicOr(&s_err, (unsigned long long)DP::kErrUnique);
    }
    if (uv > U(s, d / L)) {
      atomicOr(&s_err, (unsigned long long)DP::kErrUnion);
    }
  }
  for (int64_t i = tid; i < (int64_t)W * NN; i += blockDim.x) {
    const int s = (int)(i / NN), n = (int)(i % NN);
    int64_t su = 0;
    for (int d = n * L; d < (n + 1) * L; d++) {
      su += u(s, d);
    }
    if (U(s, n) > su) {
      atomicOr(&s_err, (unsigned long long)DP::kErrUnion);
    }
  }
  for (int d = tid; d < W; d += blockDim.x) {
    int64_t col = 0;
    for (int s = 0; s < W; s++) {
      col += region(s, d);
    }
    atomicMax(&s_maxcol, (long long)col);
  }
  // lb_minmove water-fill of every n -> m stream (reference: python/zepp/routing.py _waterfill, same piece order)
  if (NN > 1) {
    for (int p = tid; p < NN * NN; p += blockDim.x) {
      const int n = p / NN, m = p % NN;
      int64_t *bd = &s_bd[(int64_t)p * (L + 1)];
      for (int k = 0; k <= L; k++) {
        bd[k] = 0;
      }
      if (n == m) {
        continue;
      }
      int64_t V[kPlanMaxL], cap[kPlanMaxL], keep[kPlanMaxL], room[kPlanMaxL], off[kPlanMaxL];
      int64_t tot = 0;
      for (int k = 0; k < L; k++) {
        V[k] = U(n * L + k, m);
        tot += V[k];
      }
      const int64_t tq = tot / L, tr = tot % L;
      for (int k = 0; k < L; k++) {
        cap[k] = tq + (k < tr ? 1 : 0);
        keep[k] = min(V[k], cap[k]);
        room[k] = cap[k] - keep[k];
        off[k] = keep[k];
      }
      const bool mine = n == my_node;
      int pc = 0, imp = 0;
      bool bad = false;
      auto emit = [&](int sl, int64_t j_lo, int64_t j_hi, int k, int64_t o) {
        if (mine && pc < 2 * L) {
          int64_t *z = &s_pz[((int64_t)m * 2 * L + pc) * 5];
          z[0] = sl;
          z[1] = j_lo;
          z[2] = j_hi;
          z[3] = k;
          z[4] = o;
        }
        pc++;
      };
      for (int sl = 0; sl < L && !bad; sl++) {
        if (keep[sl] > 0) {
          emit(sl, 0, keep[sl], sl, 0);
        }
        int64_t j = keep[sl];
        while (j < V[sl]) {
          while (imp < L && room[imp] == 0) {
            imp++;
          }
          if (imp >= L) {
            atomicOr(&s_err, (unsigned long long)DP::kErrRoom);
            bad = true;
            break;
          }
          const int64_t take = min(V[sl] - j, room[imp]);
          emit(sl, j, j + take, imp, off[imp]);
          off[imp] += take;
          room[imp] -= take;
          j += take;
        }
      }
      if (pc > 2 * L) {
        atomicOr(&s_err, (unsigned long long)DP::kErrPieces);
      }
      for (int k = 0; k < L; k++) {
        bd[k + 1] = bd[k] + off[k];
      }
      if (mine) {
        for (int k = 0; k < L; k++) {
          s_keep[m * L + k] = keep[k];
          s_imp[m * L + k] = off[k] - keep[k];
        }
        s_npz[m] = min(pc, 2 * L);
      }
    }
  }
  // my send segments: nodes ascending, my node expanded into L per-rank segments
  if (tid == 0) {
    s_seg[0] = 0;
    for (int n = 0, seg = 0; n < NN; n++) {
      if (n == my_node) {
        for (int dl = 0; dl < L; dl++, seg++) {
          s_seg[seg + 1] = s_seg[seg] + u(rank, n * L + dl);
        }
      } else {
        s_seg[seg + 1] = s_seg[seg] + U(rank, n);
        seg++;
      }
    }
  }
  __syncthreads();

  const int64_t nseg = DP::nseg(L, NN);
  for (int i = tid; i <= nseg; i += blockDim.x) {
    plan[DP::seg_off() + i] = s_seg[i];
  }
  // round 0: intra-node puts, by local destination
  for (int dlg = tid; dlg < L; dlg += blockDim.x) {
    const int d = my_node * L + dlg;
    int64_t *r = plan + DP::round0(L, NN) + 3 * dlg;
    r[0] = u(rank, d);
    r[1] = s_seg[my_node + dlg];
    r[2] = recv_off_of_u(rank, d);
  }
  if (NN > 1) {
    // staging bounds: gateway staging per (node, relay) and relay staging (S-slot ring)
    for (int i = tid; i < NN * L; i += blockDim.x) {
      const int n = i / L, k = i % L;
      int64_t srows = 0, rrows = 0;
      for (int ns = 0; ns < NN; ns++) {
        if (ns != n) {
          srows += chunk_rows(ns, n, k);
        }
      }
      for (int dn = 1; dn < NN; dn++) {
        rrows = max(rrows, (int64_t)args.relay_slots * chunk_rows(n, (n - dn + NN) % NN, k));
      }
      atomicMax(&s_maxstage, (long long)srows);
      atomicMax(&s_maxrelay, (long long)rrows);
    }
    // per target node: my relay chunk, its pieces, and my wire put
    for (int t = tid; t < NN; t += blockDim.x) {
      if (t == my_node) {
        continue;
      }
      int64_t *r = plan + DP::tn(L, NN, t);
      const int64_t *bd = &s_bd[((int64_t)my_node * NN + t) * (L + 1)];
      r[DP::kAme] = bd[my_lr];
      r[DP::kBme] = bd[my_lr + 1];
      const bool own_only = s_imp[t * L + my_lr] == 0;
      r[DP::kOwnOnly] = own_only ? 1 : 0;
      r[DP::kOwnSrc] = s_seg[t < my_node ? t : t + L - 1];
      int64_t wd = 0;
      for (int n = 0; n < my_node; n++) {
        if (n != t) {
          wd += chunk_rows(n, t, my_lr);
        }
      }
      r[DP::kWireDst] = wd;
      int np = 0;
      if (!own_only) {
        auto put_piece = [&](int sl, int64_t src, int64_t dst, int64_t rows) {
          int64_t *q = r + DP::kPieces + 4 * np;
          q[0] = sl;
          q[1] = peer_seg_base(sl, t) + src;
          q[2] = dst;
          q[3] = rows;
          np++;
        };
        const int64_t keep = s_keep[t * L + my_lr];
        if (keep > 0) {
          put_piece(my_lr, 0, 0, keep);
        }
        for (int q = 0; q < s_npz[t]; q++) {
          const int64_t *z = &s_pz[((int64_t)t * 2 * L + q) * 5];
          if ((int)z[3] == my_lr && (int)z[0] != my_lr && np < 2 * L) {
            put_piece((int)z[0], z[1], z[4], z[2] - z[1]);
          }
        }
      }
      r[DP::kNpieces] = np;
    }
    // per source node: my gateway window, its staging offset, and the local forward destinations
    for (int s = tid; s < NN; s += blockDim.x) {
      if (s == my_node) {
        continue;
      }
      int64_t *r = plan + DP::ns(L, NN, s);
      const int64_t *bd = &s_bd[((int64_t)s * NN + my_node) * (L + 1)];
      r[DP::kWinA] = bd[my_lr];
      r[DP::kWinB] = bd[my_lr + 1];
      int64_t ws = 0;
      for (int n = 0; n < s; n++) {
        if (n != my_node) {
          ws += chunk_rows(n, my_node, my_lr);
        }
      }
      r[DP::kWstage] = ws;
      for (int dlg = 0; dlg < L; dlg++) {
        r[DP::kFwd + dlg] = recv_off_of_u(s * L, my_node * L + dlg);
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    unsigned long long err = s_err;
    if (s_maxcol > args.max_recv) {
      err |= DP::kErrRecv;
    }
    if (s_seg[nseg] > args.copies_per_rank) {
      err |= DP::kErrSend;
    }
    if (NN > 1 && s_maxstage > args.max_stage) {
      err |= DP::kErrStage;
    }
    if (NN > 1 && s_maxrelay > args.max_relay) {
      err |= DP::kErrRelay;
    }
    plan[DP::kM] = (int64_t)s_m;
    plan[DP::kSendRows] = s_seg[nseg];
    plan[DP::kSelfRows] = u(rank, rank);
    plan[DP::kSelfSend] = s_seg[my_node + my_lr];
    plan[DP::kSelfRecv] = recv_off_of_u(rank, rank);
    plan[DP::kErr] = (int64_t)err;
    // deferred verdict: inconsistent counts become a device assert (the capacity bits recv / stage / relay
    // are the demands kernel's, which runs next and makes the step degenerate)
    const unsigned long long hard =
        err & (unsigned long long)(DP::kErrRoom | DP::kErrPieces | DP::kErrUnique | DP::kErrUnion | DP::kErrSend);
    if (args.verdict != nullptr && hard != 0) {
      atomicOr(reinterpret_cast<unsigned long long *>(args.verdict + Verdict::kErr), hard);
      atomicOr(reinterpret_cast<unsigned long long *>(args.verdict + Verdict::kMask),
               (unsigned long long)Verdict::kErrorBit);
    }
    plan[DP::kSeq] = (int64_t)args.seq;
  }
}

}  // namespace

void
a2av_dispatch_plan_impl(A2AVDispatchPlanArguments const &args, cudaStream_t stream) {
  FLUX_CHECK(args.L >= 1 && args.L <= kPlanMaxL) << "dispatch plan: L out of range";
  FLUX_CHECK(args.nnodes >= 1 && args.nnodes <= kPlanMaxNN) << "dispatch plan: nnodes out of range";
  FLUX_CHECK_EQ((int64_t)args.W, (int64_t)args.L * args.nnodes) << "dispatch plan: W != L * nnodes";
  A2AVDispatchPlanArguments a = args;
  const size_t stage_bytes =
      sizeof(int32_t) * ((size_t)args.W * args.nexperts + (size_t)args.W * (args.W + args.nnodes));
  size_t smem = 0;
  static const size_t static_bytes = plan_static_smem(a2av_dispatch_plan_kernel);  // ~30 KB of scratch
  if (static_bytes + stage_bytes <= kPlanSmemBudget) {
    a.stage = 1;
    smem = stage_bytes;
  }
  plan_dyn_smem_optin(a2av_dispatch_plan_kernel, static_bytes, smem);
  a2av_dispatch_plan_kernel<<<1, kPlanThreads, smem, stream>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

// kernel registry (include/flux/cuda/kernel_registry.h): every kernel of this file. CUB's EmptyKernel
// is its PTX-version probe, compiled in by the block-scan headers and never launched; it is listed so
// the registry equals the library's kernel list.
void
dispatch_meta_kernels(KernelList &out) {
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_stage1_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_pack_scan_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_consumer_build_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_gating_cumsum_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_meta_front_fused_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_meta_pass2_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_meta_arena_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_demands_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(a2av_dispatch_plan_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY_T("EmptyKernel<void>", cub::EmptyKernel<void>));
}

}  // namespace bytedance::flux
