//===- sort_util.h --------------------------------------------- C++ ------===//
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

#pragma once
#include "flux/utils.h"

namespace bytedance::flux {






// a2av dispatch stage 1, fused: one grid-stride pass over all global copies
// decodes expert/source/owner, produces every tensor stage 2 needs, the [W,W]
// chunk-count matrix (must be pre-zeroed), and the producer pack keys
// (e * copies_per_rank + local_p — pack tie-break is the global copy index).
struct A2AVStage1Arguments {
  int32_t const *scatter_index;  // [n_copies] global dst rows
  int32_t const *splits;         // [nexperts]
  // [n_copies] expert of copy p (the routed id the scatter index was built from: the scatter is the
  // stable scatter of these ids, so this IS the expert the search below recovers); nullptr = decode
  // by binary search over the splits prefix
  int32_t const *routing_ids = nullptr;
  int nexperts;
  int ep_nexperts;
  int world_size;
  int rank;
  int64_t copies_per_rank;
  int64_t n_copies;
  int64_t *e_all;        // [n_copies] global expert id
  int64_t *s_all;        // [n_copies] source rank
  int64_t *flat_dst;     // [n_copies] dst row (int64 copy of scatter_index)
  bool *not_mine;        // [n_copies] owner != rank
  int64_t *expert_base;  // [nexperts] exclusive row base per expert; nullptr = skip
  int32_t *chunks;       // [world_size * world_size], pre-zeroed; nullptr = skip
                         // counting (metadata path derives it host-side)
  int64_t *pack_key;     // [copies_per_rank]
  // compress pack fusion: per-(segment, local token) flags, SEG-MAJOR
  // [nseg, tokens_per_rank] (contiguous per segment for the pack scan),
  // pre-zeroed; nullptr = skip. topk/local_world_size/node_idx only read
  // when pack_flag != nullptr.
  int32_t *pack_flag;
  int topk;
  int local_world_size;
  int node_idx;
  // fused consumer build: per-global-token keep flag (dedup recv rows are its
  // exclusive cumsum), pre-zeroed incl. the +1 garbage slot; nullptr = skip.
  // Keep rule mirrors the compress consumer: same-node source -> owner == rank;
  // remote source -> union_bcast ? dst node == my node : owner == rank.
  int32_t *mine_token;
  bool union_bcast;
  // deferred capacity verdict (core/verdict.h; nullptr = off): once its mask word is non-zero the step is
  // degenerate, every copy is foreign and no pack flag or keep flag is written (zero rows everywhere)
  int64_t const *verdict = nullptr;
};
void a2av_stage1_impl(A2AVStage1Arguments const &args, cudaStream_t stream);

// a2av compress consumer build, fused (no ATen key/argsort/index_select
// chain): one grid-stride pass over all copies assigns each kept
// copy its A row via the host block-start table offA (A-order groups
// g = e_loc * W + s) plus a per-group atomic rank — interior order within a
// group is arbitrary, which no consumer observes (gather/scatter are per-row
// indirections and the tile gating compares only group boundaries; same
// design as the dense sort). Writes gather_A[row] = c_excl[token] (the
// dedup recv row) and scatter_D[row] = flat_dst - expert_base[e]. Rows past
// M_this_ep are never read by the GEMM and stay untouched. When lane_end !=
// nullptr it also histograms rows into gating lanes by
// upper-bounding the recv row in lane_end[0..W-1]; a second tiny kernel turns
// the histogram into the inclusive per-expert lane cumsum the claimer reads.
// blk_cnt and gate_hist are [E * W] i32 scratch, pre-zeroed.
struct A2AVConsumerBuildArguments {
  int64_t n_copies;
  int topk;
  int ep_start;
  int ep_nexperts;            // E
  int world_size;             // W
  int64_t const *e_all;       // [n_copies] global expert id
  int64_t const *s_all;       // [n_copies] source rank
  int64_t const *flat_dst;    // [n_copies]
  bool const *not_mine;       // [n_copies]
  int64_t const *c_excl;      // [ntokens + 1] exclusive cumsum of mine_token
  int64_t const *offA;        // [E * W] A-order group starts (meta arena)
  int64_t const *expert_base; // [nexperts] (meta arena)
  int32_t *blk_cnt;           // [E * W], pre-zeroed
  int32_t *gather;            // [>= M_this_ep] out: A row -> dedup recv row
  int32_t *scatter;           // [>= M_this_ep] out: A row -> per-expert D row
  // gating (nullptr = skip): lane_end = gate_q row 0 shifted by one
  // (i.e. end(0..W-1)); gate_hist accumulates per-(e_loc, lane) row counts
  int64_t const *lane_end;    // [W]
  int32_t *gate_hist;         // [E * W], pre-zeroed
  // LANE-KEYED A order: the tile gate partitions an
  // expert's A rows by LANE (window) via gating_cumsum, so the A order must be
  // lane-monotone within each expert; source-keyed groups (offA) are NOT —
  // windows cut through source regions (a tile could read a row before its
  // window has landed). Two-pass protocol:
  //   hist_only = true            -> only gate_hist is accumulated (pass 1)
  //   offA_lane != nullptr        -> row = offA_lane[e_loc*W + lane] + atomic
  //                                  in-lane rank (pass 3; gate_hist untouched)
  //   both unset                  -> source-keyed single pass (single node)
  bool hist_only;
  int64_t const *offA_lane;   // [E * W] lane-keyed A-order group starts
  // minimal-move relay partition (paper
  // §4.2 eq. 2/3): the remote-node union regions are laid out CHUNK-major
  // (relay k's window = its kept own-segment prefix + the excess pieces it
  // imports), so the canonical dedup recv row (source-ascending cumsum) is
  // remapped through per-source-node piece tables, sorted by canonical
  // start: for region-relative row x in piece i (mm_lo[i] <= x < mm_hi[i])
  // the recv row is mm_base[ns] + mm_dst[i] + (x - mm_lo[i]). Own-node
  // sources are never remapped. mm_off == nullptr -> identity (equal-cut
  // canonical windows).
  int local_world_size;       // L
  int node_idx;
  int64_t const *mm_off;      // [NN + 1] piece range of source node ns
  int64_t const *mm_lo;       // [NN * 2L] canonical (region-relative) piece start
  int64_t const *mm_hi;       // [NN * 2L] piece end
  int64_t const *mm_dst;      // [NN * 2L] chunk-major (region-relative) piece start
  int64_t const *mm_base;     // [NN] region base (recv_off_u[ns * L])
};
void a2av_consumer_build_impl(A2AVConsumerBuildArguments const &args, cudaStream_t stream);

// finalize: gating_cumsum[e][w] = sum_{w' <= w} gate_hist[e][w']
// (inclusive over lanes; empty lanes repeat the previous value)
struct A2AVGatingCumsumArguments {
  int ep_nexperts;             // E
  int world_size;              // W
  int32_t const *gate_hist;    // [E * W]
  int32_t *gating_cumsum;      // [E * W] out
  // optional lane-keyed A offsets: offA_lane[e*W + w] = offA[e*W] (the
  // expert's first A row, source-keyed table) + exclusive lane prefix
  int64_t const *offA;         // [E * W] or nullptr
  int64_t *offA_lane;          // [E * W] out or nullptr
};
void a2av_gating_cumsum_impl(A2AVGatingCumsumArguments const &args, cudaStream_t stream);

// compress pack fusion stage 2: one block per send segment; a multi-tile
// block-wide exclusive scan over the segment's flag row assigns each flagged
// token its exclusive rank, writing pack_gather[seg_off[seg] + rank] = token.
// One kernel instead of an ATen scatter/cumsum/scatter chain (the cost there
// is launch count, not bandwidth: total work is nseg * tokens_per_rank int32).
struct A2AVPackScanArguments {
  int32_t const *pack_flag;  // [nseg, tokens] seg-major (from a2av_stage1)
  int64_t const *seg_off;    // [nseg] exclusive send-segment row offsets
  int64_t *pack_gather;      // [copies_per_rank + 1] send row -> local token
  int64_t tokens;            // tokens_per_rank
  int nseg;
};
void a2av_pack_scan_impl(A2AVPackScanArguments const &args, cudaStream_t stream);


// ---------------------------------------------------------------------------
// Routed metadata derivation: derive_routed_meta derives splits /
// splits_per_source / unique_counts / a stable scatter_index ON DEVICE from
// the raw replicated topk routing, inside the step.
// ---------------------------------------------------------------------------

// Routed metadata in three launches (no memsets, no global atomics): splits,
// splits_per_source (sps), unique_counts (uc = u_mat | U_mat) and the
// DETERMINISTIC stable scatter index, bit-identical to the python
// argsort(stable).argsort() reference (upstream Flux's calc_scatter_index
// is non-deterministic and must NEVER produce
// replicated cross-rank data). Tiles hold whole tokens of one source rank
// (tile = kA2AVMetaTile / topk tokens); block b = src * tiles_per_src + t
// enumerates the tiles in flat (token, k) order, so per-block prefix sums
// reproduce the flat stable rank exactly:
//   pass1 (grid = num_blocks): per-tile expert histogram -> block_hist,
//          per-tile dedup owner/node counts -> uc_blk
//   scan  (the last pass1 block to finish): block_offset = per-expert exclusive scan over blocks,
//          sps = per-source sums, splits = totals, uc = per-source sums of
//          uc_blk, expert_base = exclusive scan of splits (cub block scan)
//   pass2 (grid = num_blocks): one warp per 256-copy slice, in-warp stable
//          rank via __match_any_sync; emits expert_base + block_offset +
//          warp prefix + running + lane rank.
constexpr int32_t kA2AVMetaTile = 2048;  // copies per tile (>= topk)

struct A2AVMetaArguments {
  int32_t const *topk_ids;  // [ntokens_global, topk] global (virtual) expert ids
  int64_t ntokens;          // ntokens_global (= tokens_per_rank * W)
  int32_t topk;
  int32_t nexperts;         // E_virt
  int32_t ep_nexperts;      // experts per rank (owner = e / ep_nexperts)
  int32_t world_size;       // W (<= 128: per-token owner bitmask in 2x u64)
  int32_t nnodes;           // (<= 64)
  int32_t local_world;
  int64_t tokens_per_rank;
  int32_t tiles_per_src;    // = a2av_meta_tiles_per_src(tokens_per_rank, topk)
  int32_t *block_hist;      // [num_blocks, nexperts] scratch
  int32_t *block_offset;    // [num_blocks, nexperts] scratch
  int32_t *uc_blk;          // [num_blocks, W + nnodes] scratch
  int32_t *expert_base;     // [nexperts + 1] scratch (exclusive scan)
  int32_t *splits;          // [nexperts] out
  int32_t *sps;             // [W, nexperts] out (splits_per_source)
  int32_t *uc;              // [W, W + nnodes] out (u_mat | U_mat)
  int32_t *scatter_index;   // [ntokens * topk] out
};
inline int32_t
a2av_meta_tile_tokens(int32_t topk) {
  return kA2AVMetaTile / topk;
}
inline int32_t
a2av_meta_tiles_per_src(int64_t tokens_per_rank, int32_t topk) {
  const int32_t tt = a2av_meta_tile_tokens(topk);
  return (int32_t)((tokens_per_rank + tt - 1) / tt);
}
void a2av_meta_impl(A2AVMetaArguments const &args, cudaStream_t stream);
// the same chain in two parts (the planning branches): pass1 + scan (splits, sps, uc, expert_base), then pass2
// (scatter_index; reads only the block offsets and expert_base of the first part)
void a2av_meta_front_impl(A2AVMetaArguments const &args, cudaStream_t stream);
void a2av_meta_pass2_impl(A2AVMetaArguments const &args, cudaStream_t stream);

struct ProblemSchedule {
  int32_t expert_id;
  int32_t m_start;
  int32_t m_end;
  int32_t source_rank_start;
  int32_t source_rank_end;

  friend std::ostream &
  operator<<(std::ostream &os, ProblemSchedule const &sched) {
    os << "expert_id:" << sched.expert_id;
    os << ",m_start:" << sched.m_start;
    os << ",m_end:" << sched.m_end;
    os << ",source_rank_start:" << sched.source_rank_start;
    os << ",source_rank_end:" << sched.source_rank_end;
    return os;
  }
};





// we shift the computation rank order to comply with the order of gathering data.
// for ranks of the same nodes, circular shift the ranks to make the current local rank
// to be processed first. for different nodes, do the same shifting strategy as the local ranks.
// e.g. two nodes with ranks [0,1,2,3], [4,5,6,7]:
//  for rank #1, the order is: (1,2,3,0,5,6,7,4)
//  for rank #6, the order is: (6,7,4,5,2,3,0,1)
CUTLASS_HOST_DEVICE
int
shift_rank_to_order(int rank, DistEnv const &dist_env) {
  auto [node_idx, local_rank] = dist_env.global_rank_to_node_idx_local_rank(rank);
  int node_idx_shift = (node_idx - dist_env.node_idx + dist_env.nnodes) % dist_env.nnodes;
  int local_rank_shift =
      (local_rank - dist_env.local_rank + dist_env.local_world_size) % dist_env.local_world_size;
  return dist_env.local_rank_to_global_rank(local_rank_shift, node_idx_shift);
}



CUTLASS_HOST_DEVICE
int
revert_order_to_rank(int order, DistEnv const &dist_env) {
  auto [node_idx, local_rank] = dist_env.global_rank_to_node_idx_local_rank(order);
  int node_idx_origin = (node_idx + dist_env.node_idx) % dist_env.nnodes;
  int local_rank_origin = (local_rank + dist_env.local_rank) % dist_env.local_world_size;
  return dist_env.local_rank_to_global_rank(local_rank_origin, node_idx_origin);
}

// ---------------------------------------------------------------------------
// Device-side derivation of the a2av metadata arena. One block builds,
// from the device sps [W, nexperts] and uc [W, W + nnodes] counts, the arena
// DispatchGemmOpImpl::a2av_dispatch reads:
//   cumA/offA/offR_of_A i64[E*W], expert_base i64[nexperts], ssc i32[E*W], then
//   at compress_off: seg_off i64[L+NN], gate_q i64[E*(W+1)] (NN > 1), minmove
//   receiver tables i64[(NN+1)+NN+3*NN*2L] (lb_minmove; target = my node).
struct A2AVMetaArenaArguments {
  int32_t const *sps;  // [W, nexperts] device (splits per source)
  int32_t const *uc;   // [W, W + nnodes] device (u_mat | U_mat)
  int32_t W, nexperts, ep_nexperts, ep_start, L, nnodes, my_node, rank;
  int32_t lb_minmove;  // 1: water-fill chunk bounds (set when nnodes > 1), 0: equal cut
  int64_t recv_key;    // R_key = max recv rows (gating query stride)
  char *arena;         // device arena slice, host layout
  int64_t compress_off;
  int32_t stage = 0;   // set by the launcher when they fit: sps / uc copied to shared memory first
  // planning branches: the re-run after the demands kernel on a degenerate step only; the kernel returns at once
  // unless *only_if_dead != 0 (the verdict block's kDead word). nullptr = always run.
  int64_t const *only_if_dead = nullptr;
};
void a2av_meta_arena_impl(A2AVMetaArenaArguments const &args, cudaStream_t stream);

// Exact per-step buffer demands on device (python/zepp/capacity.py
// `demands_from_meta`, same expressions), compared against the capacities.
// out[0..6] = recv_rows, dispatch_recv, dispatch_stage, dispatch_relay,
// combine_conv, combine_wire, pair_rows (each >= 1); out[7] = violation mask
// (bit i = i-th (demand, capacity) pair of capacity.py _FIT; direct: bits 0, 7).
struct A2AVDemandsArguments {
  int32_t const *sps;  // [W, W * ep_nexperts] device
  int32_t const *uc;   // [W, W + nnodes] device
  int32_t W, ep_nexperts, L, nnodes, relay_slots, direct;
  int64_t caps[8];     // recv_cap, dispatch_recv, dispatch_stage, dispatch_relay,
                       // combine_send, combine_conv, combine_wire, pair_cap
  int64_t *out;        // [8] device
  // deferred capacity verdict (core/verdict.h; nullptr = off): the mask (plus the force bit) is ORed into
  // the block, the first violating layer latches its demands, and once the block is non-zero the step's
  // counts and dispatch plan block are zeroed, so every later builder of the step emits zero sizes
  int64_t *verdict = nullptr;
  int64_t layer = 0;              // MoE layer ordinal within the forward
  int64_t force = 0;              // forced abort (serving_check --force-abort): set the verdict at this layer
  int32_t *zero_sps = nullptr;    // [W, W * ep_nexperts] (the sps above)
  int32_t *zero_uc = nullptr;     // [W, W + nnodes] (the uc above)
  int32_t *zero_splits = nullptr;  // [W * ep_nexperts]
  int64_t *zero_plan = nullptr;   // the dispatch plan block (DispatchPlan::words), or nullptr
  int32_t stage = 0;              // set by the launcher when they fit: sps / uc copied to shared memory first
  int64_t plan_words = 0;
};
void a2av_demands_impl(A2AVDemandsArguments const &args, cudaStream_t stream);

// Dispatch plan block: every size and offset the dispatch wire of THIS rank needs, derived from the
// replicated counts (sps, uc). The same fixed int64 layout is produced by the device
// plan kernel and by the reference python/zepp/routing.py dispatch_plan_ref; the wire code reads only the
// block. Layout:
//   header[8]         m_this_ep, total_send_rows, self_rows, self_send_off, self_recv_off, err, seq, 0
//                     (seq: the launcher's sequence word, 0)
//   seg_off[nseg + 1] my send segments: nodes ascending, my node expanded into L per-rank segments
//   round0[L][3]      by local destination dlg: rows, src row (send buffer), dst row (peer recv buffer)
//   tn[NN][6 + 8L]    by target node: a_me, b_me, own_only, own_src, wire_dst, npieces, then 2L pieces
//                     {source local rank, src row (that source's send buffer), dst row (my chunk), rows}
//   ns[NN][3 + L]     by source node: win_a, win_b, wstage, fwd_dst[L] (by local destination dlg)
// err bits are identical on every rank (the inputs are replicated).
struct DispatchPlan {
  static constexpr int kHeader = 8;
  enum Hdr : int { kM = 0, kSendRows = 1, kSelfRows = 2, kSelfSend = 3, kSelfRecv = 4, kErr = 5, kSeq = 6 };
  enum Err : int64_t {
    kErrRoom = 1,     // water-fill ran out of room
    kErrPieces = 2,   // more than 2L pieces in a stream
    kErrUnique = 4,   // unique counts inconsistent with splits_per_source
    kErrUnion = 8,    // node-union counts out of range
    kErrRecv = 16,    // dispatch recv rows above capacity
    kErrSend = 32,    // send rows above copies per rank
    kErrStage = 64,   // gateway staging above capacity
    kErrRelay = 128,  // relay staging above capacity
  };
  enum Tn : int { kAme = 0, kBme = 1, kOwnOnly = 2, kOwnSrc = 3, kWireDst = 4, kNpieces = 5, kPieces = 6 };
  enum Ns : int { kWinA = 0, kWinB = 1, kWstage = 2, kFwd = 3 };
  CUTLASS_HOST_DEVICE static int64_t
  nseg(int L, int NN) {
    return (int64_t)L + NN - 1;
  }
  CUTLASS_HOST_DEVICE static int64_t
  seg_off() {
    return kHeader;
  }
  CUTLASS_HOST_DEVICE static int64_t
  round0(int L, int NN) {
    return kHeader + nseg(L, NN) + 1;
  }
  CUTLASS_HOST_DEVICE static int64_t
  tn_stride(int L) {
    return kPieces + 8 * (int64_t)L;
  }
  CUTLASS_HOST_DEVICE static int64_t
  tn(int L, int NN, int t) {
    return round0(L, NN) + 3 * (int64_t)L + (int64_t)t * tn_stride(L);
  }
  CUTLASS_HOST_DEVICE static int64_t
  ns_stride(int L) {
    return kFwd + (int64_t)L;
  }
  CUTLASS_HOST_DEVICE static int64_t
  ns(int L, int NN, int s) {
    return tn(L, NN, NN) + (int64_t)s * ns_stride(L);
  }
  CUTLASS_HOST_DEVICE static int64_t
  words(int L, int NN) {
    return ns(L, NN, NN);
  }
};

struct A2AVDispatchPlanArguments {
  int32_t const *sps;  // [W, nexperts] device
  int32_t const *uc;   // [W, W + nnodes] device
  int32_t W, nexperts, ep_nexperts, ep_start, L, nnodes, my_node, rank, relay_slots;
  int64_t copies_per_rank, max_recv, max_stage, max_relay;
  int64_t *plan;  // [DispatchPlan::words(L, nnodes)] device
  // deferred capacity verdict (core/verdict.h; nullptr = off): the non-capacity error bits (inconsistent
  // counts) become device-assert bits of the block instead of a host check (the capacity bits are the
  // demands kernel's)
  int64_t *verdict = nullptr;
  uint64_t seq = 0;  // written to the header
  int32_t stage = 0;  // set by the launcher when they fit: sps / uc copied to shared memory first
};
// one block; part of the planning chain (enqueued before the dispatch GEMM)
void a2av_dispatch_plan_impl(A2AVDispatchPlanArguments const &args, cudaStream_t stream);

}  // namespace bytedance::flux


