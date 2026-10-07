//===- gemm_combine.h -------------------------------------------- C++ ---===//
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

#include "flux/flux.h"

#pragma once
namespace bytedance::flux {

constexpr int kMaxNumGroups = 2;
constexpr int kMaxExpertCount = 1024;





struct GemmCombineArguments {
  void *problem_sizes;  // cutlass::gemm::GemmCoord*
  int problem_count;
  int *non_empty_problem_count;  // a pointer in GPU memory
  float alpha;
  float beta;
  void **ptr_A;
  void **ptr_B;
  void **ptr_C;
  void **ptr_D;
  void *lda;  // to support split_n
  void *ldb;
  void *ldc;
  void *ldd;
  void *ldr;
  // for FP8 arguments
  void **ptr_Aux = nullptr;     // m * n
  void **ptr_Vector = nullptr;  // bias: 1 * n
  float *abs_max_Aux = nullptr;
  float *abs_max_D = nullptr;
  // scaling tensors
  float const **scaleA = nullptr;
  float const **scaleB = nullptr;
  float const *scaleC = nullptr;
  float const *scaleD = nullptr;    // require if D is fp8
  float const *scaleAux = nullptr;  // require if Aux is fp8
  int32_t topk;
  int32_t *barrier;
  int32_t *routing_idx;
  int32_t n_split;
  // following args are for expert parallel
  int sm_margin;
  // M-split waves: device [n_waves] non-empty problem count per
  // cascade group (a wave can lack rows for an expert that is non-empty
  // elsewhere, so the uniform division is wrong there). nullptr = the
  // uniform per-split division (column splits).
  int const *non_empty_per_group = nullptr;
  // chunked combine: device [problem_count]
  // problem -> cascade group (wave) map for chunk-ordered problem lists;
  // nullptr = the uniform division.
  int const *prob_group_map = nullptr;
  // epilogue-fused pack: per-problem D scatter-index pointers (built by
  // make_workspace; identity iota when fused pack is off)
  int **scatter_D_ptr = nullptr;
  // swap: combine-side weight gate — device
  // [problem_count] problem -> index into weight_signal_ptr (-1 ungated);
  // gated problems spin at tile start until signal >= expected. nullptr = off.
  int const *prob_wgate_map = nullptr;
  uint64_t const *weight_signal_ptr = nullptr;
  uint64_t weight_signal_expected = 0;
  // device step state (core/step_state.h): when set, the expected epoch is read from this word instead
  uint64_t const *weight_signal_expected_ptr = nullptr;
};



// ---- a2av_hier combine (all-to-all-v): pack + reduce kernel arguments ----

constexpr int kA2AVMaxNodes = 32;

// Persistent pack kernel: split-major outer loop gated on the GEMM's per-split
// ready flag; per split it gathers each outgoing copy's n_per column window from
// gemm_out into the symmetric send panel in (home_rank, expert, copy) order
// (== the a2av dispatch's recv layout), applying output_vec_scale per source
// row. Chunk completion per (dest_node, sid) is published to the host put
// ladders via the group_counters/group_flags handshake -- including the OWN
// node's chunk (the intra-node ladder gates on it).
struct CombinePackArguments {
  void const *gemm_out;         // [m_this_ep, n] of dtype, expert-major rows
  float const *vec_scale;       // [m_this_ep] per-row topk weight, nullptr if absent
  int32_t const *pack_index;    // [m_this_ep]: send-panel row -> gemm_out row
  void *send_panel;             // [n_split, panel_rows, n_per] symmetric
  int *barrier;                 // local per-split GEMM ready flags (cascade output)
  uint64_t *group_flags;        // [nnodes * n_split] chunk-ready epoch words: set to run_id when ready
  int *group_counters;          // [nnodes * n_split] per-block completion counters (self-resetting)
  uint64_t run_id;              // value the flags are set to (the op's monotonic run id)
  int64_t node_row_start[kA2AVMaxNodes + 1];  // dest-node row ranges in the send panel
  int64_t panel_rows;           // send panel row capacity per split
  int n;
  int n_per;                    // n / n_split
  int n_split;
  int nnodes;
  int node_idx;
  int threadblock_count;
  // M-split waves: the GEMM completes in
  // destination-wave order (ring waves of dest-node row segments, n_split==1),
  // and the pack gates PER RING STEP on the wave's cascade flag instead of the
  // single split flag — the whole per-node ladder then pipelines under the
  // remaining GEMM. msplit == 0 keeps one gate per split.
  int msplit;                       // 0 = one gate per split
  int wave_of_node[kA2AVMaxNodes];  // schedule step -> barrier wave-flag index
  int node_order[kA2AVMaxNodes];    // schedule step -> dest node (ring or size-sorted)
  // epilogue-fused pack: the GEMM already wrote the send panel via
  // ScatterD, so the pack degenerates to a pure FLAG RELAY — wait each wave
  // flag and flip the per-node chunk flags, moving no data.
  int relay_only;
  // pieces: number of per-expert-chunk cascade flags to wait at entry
  // (the no-split build fires barrier[0..n) per chunk; wave_of_node stays 0).
  // 0 = per-wave gating.
  int n_chunk_flags;
  // piece relay (requires relay_only + msplit): wait each piece's
  // chunk-flag range, then flip per (node, piece) flags (depth-8 slots);
  // the own-node per-split group flag flips at the last piece for the intra
  // ladder.
  // 0 = off.
  int n_pieces;
  int piece_first_chunk[9];   // [p] = first chunk flag of piece p; [P] = end
  uint64_t *piece_group_flags;  // [nnodes * 8] epoch words (run_id)
  int *piece_group_counters;    // [nnodes * 8]
  // deferred verdict (core/verdict.h), nullptr = host values above:
  // node_row_dev: the combine plan block's send_off [W + 1]; dest node g's rows are
  //   [node_row_dev[g * node_row_stride], node_row_dev[(g + 1) * node_row_stride]) (stride = L)
  int64_t const *node_row_dev = nullptr;
  int node_row_stride = 0;
  // msplit_dec: the device wave-adapt decision (1 = destination waves, 0 = collapsed to the single-gate
  //   GEMM): msplit and relay_only above hold only when it is 1
  int32_t const *msplit_dec = nullptr;
  // device-issued wire (dwire/dwire.h), plan == nullptr = the host lanes: after each node's rows
  // are complete, the kernel stores them straight into their destinations -- my node's homes d: peer d's
  // receive panel at dst_off[d]; homes d = (g, dl) on a remote node g: my node's gateway dl's convergence
  // panel at conv_dst[g * L + dl] -- and the last arriving block per destination release-stores its signal
  // (recv_sig[rank * n_split + sid] at d / conv_sig[(my_lr * NN + g) * n_split + sid] at the gateway).
  // Node g's rows of all L destinations go in one pass, one warp per row (its index, source row and destination
  // computed once), every lane issuing four independent 16-byte loads before its four stores, then ONE system fence
  // and the L per-destination arrivals.
  // Collapsed waves (no fused pack): rows are gathered from gemm_out x vec_scale directly (no send panel).
  int64_t const *plan = nullptr;
  int64_t plan_send_off = 0, plan_send_rows = 0, plan_dst_off = 0, plan_conv_dst = 0;  // layout word offsets
  char *const *recv_peer = nullptr;         // [L] node peers' receive panels (split 0 base)
  char *const *conv_peer = nullptr;         // [L] node peers' convergence panels
  uint64_t *const *recvsig_peer = nullptr;  // [L]
  uint64_t *const *convsig_peer = nullptr;  // [L]
  int64_t recv_cap = 0, conv_cap = 0;       // rows per split
  int local_world_size = 0, my_lr = 0, rank = 0;
  unsigned *push_counters = nullptr;        // [world * n_split] per destination (self-resetting)
  uint64_t const *run_ptr = nullptr;        // device step state: the run id read from the step slot
  uint64_t const *kill = nullptr;           // process kill word
};

// invert an int32 permutation-ish map (out[idx[p]] = p) — builds the
// pack inverse (gemm row -> send-panel row) from pack_index at plan time.
struct A2AVInvertIndexArguments {
  int32_t const *idx;  // [n]
  int32_t *out;        // [n]
  int64_t n;           // deferred verdict: the grid hint, the count is *n_dev
  int64_t const *n_dev = nullptr;
};

// Deferred verdict: the K-side fold of the gate coefficients into the combine GEMM's input (the fused pack:
// rows [0, *rows_dev) of input [*, k] are scaled by scales[row] with the arithmetic of a torch in-place
// multiply: fp32 product rounded to the element type), only when the device wave-adapt decision *dec is 1
// (destination waves; the collapsed GEMM scales in the pack instead). Also the rows-bound device assert:
// *rows_dev above rows_bound sets a verdict error bit.
struct A2AVFoldScalesArguments {
  void *input;              // [rows_bound, k] of dtype
  float const *scales;      // [rows_bound]
  int64_t const *rows_dev;  // rows this rank computes (combine plan block kM)
  int32_t const *dec;       // wave-adapt decision, nullptr = always fold
  int32_t fold;             // 0: the rows-bound assert only (no fused pack)
  int64_t k;
  int64_t rows_bound;
  int64_t *verdict;         // nullptr = no assert
  int32_t vec8;             // set by a2av_fold_scales: 8 elements (16 bytes) per thread, k % 8 == 0, aligned rows
};
void a2av_fold_scales(A2AVFoldScalesArguments const &args, DataTypeEnum dtype, cudaStream_t stream);
void a2av_invert_index(A2AVInvertIndexArguments const &args, cudaStream_t stream);


// Sort-free compress-plan derivation: every ordering in the
// compress CSRs is arithmetic on the dispatch's stable scatter_index plus host
// cnt/U prefix tables (the conv panel is the destination A-order restricted
// per (segment, owner-lane) block; wire/red groupings are per-token O(topk)
// ranks). 4 kernels, no sorts, deterministic direct writes, bitwise equal to
// the argsort formulation.
struct A2AVCompressPlanArguments {
  // per-copy inputs (device)
  int32_t const *scatter_index;  // [m_full] global A-order position per copy
  int32_t const *e_of_copy;      // [m_full] expert of each copy
  // prefix tables (device; built host-side from cnt/U, no device sync)
  int32_t const *home_base;    // [nex * W] exclusive per-expert home prefix
  int64_t const *expert_base;  // [nex] exclusive A-order expert base
  int64_t const *conv_base;    // [(NN-1) * L * E_loc] conv bucket bases
  int64_t const *my_cnt_cum;   // [nex] exclusive prefix of cnt[rank][e]
  int64_t const *recv_off_C;   // [W]
  int64_t const *recv_off_Cp;  // [W]
  int64_t const *rem_base;     // [NN]
  // scratch (device, zeroed by the launcher where required)
  int32_t *conv_count;   // [(NN-1) * tokens_per_rank] conv copies per (seg, t)
  int32_t *wire_row_of;  // [(NN-1) * tokens_per_rank] (seg, t) -> wire row, -1 if none
  int32_t *red_flags;    // [ntok_local * NN] token contributes from node m
  int32_t *rem_pos;      // [ntok_local * NN] column-exclusive one-cumsum
  // outputs (device int32, sized by the host from cnt/U totals)
  int32_t *wire_ptr;   // [wire_total + 1]
  int32_t *wire_copy;  // [conv_total]
  int32_t *red_ptr;    // [ntok_local + 1]
  int32_t *red_row;    // [own_total + rem_total]
  // pieces (nullptr / 0 = off: rows in (seg, token) order):
  // deterministic piece of each global expert (shared chunk merge); wire
  // rows renumber (seg, ready_piece, token) on BOTH sides, where
  // ready_piece = max contributor piece. red_flags then stores piece+1
  // (0 = no contribution) instead of 0/1.
  int32_t const *piece_of_e = nullptr;  // [nexperts] device
  int n_pieces = 0;
  int32_t *wire_piece = nullptr;        // scratch [(NN-1) * tokens_per_rank]
  // out [(NN-1) * (n_pieces + 1)]: per-seg RELATIVE piece row starts
  // ([seg][P] = seg total) — equal on sender and dest by construction
  int32_t *wire_piece_start = nullptr;
  // geometry
  int64_t m_full;
  int topk;
  int world_size;
  int nnodes;
  int local_world_size;
  int rank;
  int64_t nexperts;
  int64_t ep_nexperts;  // experts per owner rank
  // deferred verdict (core/verdict.h; nullptr = off): once its mask word is set, the step is degenerate and
  // the CSRs are empty (no wire row, no reduce contribution); the per-copy kernels write nothing else
  int64_t const *verdict = nullptr;
};

void a2av_compress_plan(A2AVCompressPlanArguments const &args, cudaStream_t stream);

// Kernel-side a2av combine pack/reduce index build: the sort-free identities
// of build_a2av_combine_indices as two direct-write kernels over host prefix
// tables, in place of a ~15-op torch dispatcher chain
// (arange/searchsorted/index_select/scatter + two pageable H2Ds) that
// serializes on the host. The torch chain stays as the bitwise reference.
struct CombinePlanArguments {
  int32_t const *routing_idx;  // [m_full] dispatch stable scatter index (device)
  // prefix tables (device; host-built from cnt, ONE pinned async H2D)
  int64_t const *cumA;       // [nexG] inclusive A-order (e_loc, h) group cum
  int64_t const *offA;       // [nexG] exclusive A-order group base
  int64_t const *offR_of_A;  // [nexG] recv-panel base of A-order group
  int64_t const *expert_cum;  // [nex] inclusive global per-expert row cum
  int64_t const *my_cum;      // [nex] exclusive prefix of cnt[rank][e]
  int64_t const *h_base;      // [nex] rank's exclusive home base within e
  // outputs (device int32)
  int32_t *pack_index;    // [m_this_ep]
  int32_t *reduce_index;  // [cpr]
  int64_t m_this_ep;
  int64_t cpr;   // copies per rank (m_full / W)
  int64_t row0;  // rank * cpr slice offset into routing_idx
  int64_t nexG;  // E_loc * W A-order groups
  int64_t nex;   // total experts
  // device-table mode: the pack row count is read from the device (m_this_ep is then only the
  // launch-size hint), and the expert of each copy is taken from the routing ids instead of a search
  // over expert_cum. nullptr = host value / search.
  int64_t const *m_this_ep_dev = nullptr;
  int32_t const *e_of_copy = nullptr;  // [m_full] expert id of every copy (device)
  // deferred verdict (nullptr = off): a degenerate step's reduce index points at row 0 (in bounds; its
  // receiver writes zeros)
  int64_t const *verdict = nullptr;
};

void a2av_combine_plan(CombinePlanArguments const &args, cudaStream_t stream);

// Device-side derivation of the combine host tables (gemm_combine.cc
// build_a2av_combine_indices / build_a2av_compress_indices_fast), from the
// device counts sps [W, nexperts] and uc [W, W + nnodes]. Segments (i64):
// cumA/offA/offR_of_A [nexG], my_cum/e_base/h_base/e_cum [nex] (rtab rows),
// expert_base/my_cnt_cum [nex], conv_base [(NN-1)*L*E], recv_off_C/recv_off_Cp
// [W], rem_base [NN], totals[8] = m_this_ep, own_total, conv_total, wire_total,
// rem_total; (i32) home_base [nex * W]. Same formulas as the host, integer only.
struct A2AVCombineTablesArguments {
  int32_t const *sps;
  int32_t const *uc;
  int32_t W, ep_nexperts, L, nnodes, rank;
  int64_t *cumA, *offA, *offR_of_A;
  int64_t *my_cum, *e_base, *h_base, *e_cum;
  int64_t *expert_base, *my_cnt_cum, *conv_base, *recv_off_C, *recv_off_Cp, *rem_base;
  int64_t *totals;
  int32_t *home_base;
};
void a2av_combine_tables(A2AVCombineTablesArguments const &args, cudaStream_t stream);

// Combine plan block: every size, offset and scalar the combine's forward reads from the routing
// counts, as int64 words in a fixed layout that depends only on (W, NN, E). Notation:
// cnt[h][e] = splits_per_source, U[h][n] = unique_counts, C[s][d] = sum over s's experts of cnt[d][e]
// (rows owner s returns to home d), cp(s, d) = C[s][d] when s and d share a node, U[d][node of s]
// when s is remote with d's local rank, else 0 (the compressed receive image).
struct CombinePlanLayout {
  // header words
  static constexpr int kM = 0;           // rows this rank computes = sum_d C[rank][d]
  static constexpr int kOwnTotal = 1;    // receive rows from my node's ranks
  static constexpr int kConvTotal = 2;   // convergence rows this rank gathers as a gateway
  static constexpr int kWireTotal = 3;   // wire rows this rank sends
  static constexpr int kRemTotal = 4;    // receive rows from remote nodes
  static constexpr int kRemoteRows = 5;  // rows this rank computes for homes on other nodes
  static constexpr int kMaxSend = 6;     // max over owners s of sum_d C[s][d]
  static constexpr int kMaxConv = 7;     // max over gateways of convergence rows
  static constexpr int kMaxWire = 8;     // max over gateways of wire rows
  static constexpr int kColMin = 9;      // min over homes d of sum_s C[s][d]
  static constexpr int kColMax = 10;     // max over homes d of sum_s C[s][d]
  static constexpr int kSeq = 11;        // derive sequence number (stale-block check)
  static constexpr int kHeader = 16;
  int64_t W, NN, E;
  int64_t send_off;   // [W + 1] exclusive prefix of C[rank][d]; [W] = kM
  int64_t send_rows;  // [W] C[rank][d]
  int64_t dst_off;    // [W] cp-prefix of the rows I put at home d: sum_{s < rank} cp(s, d)
  int64_t lane_off;   // [W + 1] receive lane offsets: sum_{s' < s} cp(s', rank); [W] = own + rem
  int64_t conv_dst;   // [W] index tn * L + dl: my convergence offset at gateway (my_node, dl) for tn
  int64_t wire_seg;   // [NN] wire segment starts (remote nodes ascending); [NN - 1] = wire total
  int64_t wire_rows;  // [NN] U[tn * L + my_lr][my_node] (0 for my node)
  int64_t node_base;  // [E * (NN + 1)] per local expert: exclusive home-node prefix of its rows
  int64_t total;
};
inline CombinePlanLayout
combine_plan_layout(int64_t W, int64_t NN, int64_t E) {
  CombinePlanLayout l{};
  l.W = W;
  l.NN = NN;
  l.E = E;
  l.send_off = CombinePlanLayout::kHeader;
  l.send_rows = l.send_off + W + 1;
  l.dst_off = l.send_rows + W;
  l.lane_off = l.dst_off + W;
  l.conv_dst = l.lane_off + W + 1;
  l.wire_seg = l.conv_dst + W;
  l.wire_rows = l.wire_seg + NN;
  l.node_base = l.wire_rows + NN;
  l.total = l.node_base + E * (NN + 1);
  return l;
}

// One block of 128 threads (W <= 128), at most 64 registers per thread: it runs on the combine derive's
// side stream beside the spinning dispatch GEMM, so it must fit beside one GEMM block on any SM.
struct A2AVCombinePlanBlockArguments {
  int32_t const *sps;  // [W, W * E] device counts
  int32_t const *uc;   // [W, W + NN] device counts (U at column W + n)
  int32_t W, ep_nexperts, L, nnodes, rank;
  int64_t seq;
  CombinePlanLayout layout;
  int64_t *block;  // [layout.total]
  int32_t *lanes;  // [2 W + 1]: lane_off [W + 1] | chain_pos [W] for the bucketed receiver
  // deferred verdict (nullptr = off): the forward's host checks of the block become device-assert bits of
  // the verdict block (skipped on a degenerate step, whose counts are zero): rows this rank computes vs
  // lim_send (send panel), column sums vs cpr, the panel maxima vs lim_send / lim_conv / lim_wire, the receive
  // image vs cpr, wire rows without convergence rows
  int64_t *verdict = nullptr;
  int64_t lim_send = 0, lim_conv = 0, lim_wire = 0, cpr = 0;
  int32_t compress = 0;
};
void a2av_combine_plan_block(A2AVCombinePlanBlockArguments const &args, cudaStream_t stream);

// Per-forward GEMM problem tables of the destination-wave (msplit) combine and the per-problem
// weight-gate map, built from the plan block's node_base. One block of 128 threads, small registers.
// Problem order: wave-outer; inside a wave the experts ascending, or with `reorder` the ungated
// experts first and the gated ones last (both ascending).
constexpr int kMsplitMaxExperts = 512;
struct A2AVMsplitTablesArguments {
  int64_t const *node_base;  // [E * (NN + 1)] (device, plan block)
  int32_t *out;      // [4 * n_waves * E + n_waves]: wave_M | wave_off | eid | grp | non-empty per wave
  int32_t *wgate;    // [n_wgate] per-problem weight-signal index (-1 ungated); nullptr = none
  int32_t E, NN;
  int32_t n_waves;   // 0: no wave tables (single-gate problem list, wgate[i] = gate[i % E])
  int32_t reorder;
  int32_t n_wgate;
  int32_t wave_lo[kA2AVMaxNodes];  // wave w covers home nodes [wave_lo[w], wave_hi[w])
  int32_t wave_hi[kA2AVMaxNodes];
  int16_t gate[kMsplitMaxExperts];  // per local expert: weight-signal index, -1 ungated
  // deferred verdict (nullptr = off):
  // gate_dev: the per-local-expert gate map on the device (written by the swap lane's arm kernel) instead
  //   of `gate`
  int32_t const *gate_dev = nullptr;
  // dec: out, the wave-adapt decision of gemm_combine.cc (1 = destination waves, 0 = collapse to the
  //   single-gate GEMM when adapt_reread > adapt_ratio * (*remote_rows) * adapt_row_bytes); with 0 the tables
  //   describe the single-gate GEMM in the wave layout: wave 0 holds every row of each expert, the other
  //   waves are empty (same tiles, same rows, same pointers; group 0 completes with the whole GEMM)
  int32_t *dec = nullptr;
  int64_t const *remote_rows = nullptr;  // combine plan block kRemoteRows
  int64_t adapt_reread = 0, adapt_ratio = 0, adapt_row_bytes = 0;
};
void a2av_msplit_tables(A2AVMsplitTablesArguments const &args, cudaStream_t stream);
// force-load every kernel of the combine derive chain (called by a2av_combine_preload)
void a2av_combine_derive_preload();


constexpr int kA2AVMaxWorld = 128;

// ---- compress (dedup) combine: pre-reduce + CSR reduce kernel arguments ----

// Source-side gateway pre-reduce (persistent, one launch per forward): per
// (split, target node in inter-ladder rotation order) it spins on the L
// per-peer convergence signals, merges each wire row's contributing conv-panel
// rows (CSR) in fp32, writes the wire panel, and flips the (tn, sid) wire flag
// the host inter ladder gates on -- the pack kernel's counter/flag handshake.
// single-node receiver: per-split gather-reduce of the top-k recv rows of every local token
// (the compress wire's bucketed receiver replaces it whenever nnodes > 1)
struct CombineReduceArguments {
  void const *recv_panel;       // [n_split, panel_rows, n_per] symmetric
  int32_t const *reduce_index;  // [ntokens_local * topk]: local copy -> recv-panel row
  void *output;                 // [ntokens_local, n]
  int64_t panel_rows;           // recv panel row capacity per split
  int64_t ntokens_local;
  int n;
  int n_per;
  int topk;
  int sid;
  int threadblock_count;
  // deferred verdict (nullptr = off): a degenerate step writes zero rows (its reduce index is not routed)
  int64_t const *verdict = nullptr;
};

struct CombinePreReduceArguments {
  void const *conv_panel;        // [n_split, conv_rows, n_per] symmetric
  void *wire_panel;              // [n_split, wire_rows, n_per] symmetric
  int32_t const *wire_ptr;       // [wire_rows_local + 1] CSR offsets
  int32_t const *wire_copy;      // [conv_rows_local] wire row -> conv-panel rows
  uint64_t const *conv_signals;  // [(L * nnodes) * n_split], slot (ls*NN+tn)*n_split+sid
  uint64_t run_id;
  uint64_t *wire_flags;          // [nnodes * n_split] wire-ready epoch words: set to run_id when ready
  int *wire_counters;            // [nnodes * n_split] per-block completion counters (self-resetting)
  int node_order[kA2AVMaxNodes];  // schedule step -> remote target node (ring default)
  int64_t wire_seg_start[kA2AVMaxNodes + 1];  // wire-row start per segment (tn asc skip own)
  // pieces (0 = off): per-(tn, piece) consumption — wait the L conv
  // piece signals, process the piece's contiguous wire-row range (per-seg
  // RELATIVE starts, stride 9), flip the per-piece wire flag.
  int n_pieces = 0;
  int32_t piece_start[kA2AVMaxNodes * 9] = {};  // [seg * 9 + p], rel rows
  uint64_t const *piece_conv_sigs = nullptr;    // [(L * NN) * 8]
  uint64_t *piece_wire_flags = nullptr;         // [NN * 8] epoch words (run_id)
  int *piece_wire_counters = nullptr;           // [NN * 8]
  int64_t conv_rows;             // conv panel row capacity per split
  int64_t wire_rows;             // wire panel row capacity per split
  int n_per;
  int n_split;
  int nnodes;
  int node_idx;
  int local_world_size;
  int threadblock_count;
  // 0 = unlimited (default). >0: trap after this many no-progress sleep
  // iterations in the conv-signal wait — converts a missing-signal bug into
  // a loud abort instead of a hang . LAST field
  // with default: launch sites use positional aggregate init.
  uint64_t spin_limit = 0;
  // process kill word (pinned host memory, core/wire_runtime.h), polled in the conv-signal spin:
  // once non-zero the spin gives up (null = not polled)
  uint64_t const *kill_word = nullptr;
  // deferred verdict: the wire segment starts on the device (combine plan block wire_seg [NN], [NN - 1] =
  // total) instead of wire_seg_start; nullptr = host values
  int64_t const *wire_seg_dev = nullptr;
  // device step state: the run id read from the step slot instead of run_id (nullptr = run_id)
  uint64_t const *run_ptr = nullptr;
};








// ---- completion-bucketed register receiver ---------------------------------
// Arrival-order folding at wait-all's 1x bytes: tokens bucket by the chain
// position of their LAST-arriving contribution lane (plan-time, on-stream,
// ~us); each front-end lane wait then releases a register-CSR fold of exactly
// the tokens that lane completes. No fp32 scratch RMW (a per-lane scatter-add
// into fp32 moves 4-5x the bytes), no atomics in the fold, no finalize: every
// token is read once and written once, as early as legality allows. The
// post-last-arrival tail is one bucket instead of the whole reduce.
struct A2AVBucketMapArguments {
  int32_t const *red_ptr;    // [ntokens + 1]
  int32_t const *red_row;    // [red_total] recv-panel rows per token
  int32_t const *lane_off;   // [world_size + 1] C' recv-row prefix by source rank
  int32_t const *chain_pos;  // [world_size] source rank -> chain position
  int world_size;
  int n_chain;               // chain length S (= L + NN - 1 materializing lanes)
  int64_t ntokens;
  int32_t *comp;             // [ntokens] out: completion chain position
  int32_t *bucket_cnt;       // [n_chain] out: bucket sizes (pre-zeroed)
};
struct A2AVBucketScanArguments {
  int32_t const *bucket_cnt;  // [n_chain]
  int n_chain;
  int32_t *bucket_ptr;        // [n_chain + 1] out: exclusive prefix
  int32_t *bucket_cur;        // [n_chain] out: zeroed scatter cursors
};
struct A2AVBucketScatterArguments {
  int32_t const *comp;        // [ntokens]
  int32_t const *bucket_ptr;  // [n_chain + 1]
  int64_t ntokens;
  int32_t *bucket_cur;        // [n_chain] scatter cursors
  int32_t *bucket_tok;        // [ntokens] out: tokens grouped by completion bucket
};
struct A2AVBucketReduceArguments {
  void const *recv_panel;     // [n_split, panel_rows, n_per] symmetric (C' image)
  int32_t const *red_ptr;     // [ntokens_local + 1]
  int32_t const *red_row;     // [red_total]
  int32_t const *bucket_ptr;  // [n_chain + 1] (device; sizes unknown to host)
  int32_t const *bucket_tok;  // [ntokens_local]
  int bucket;                 // which completion bucket this launch folds
  void *output;               // [ntokens_local, n]
  int64_t panel_rows;
  int n;
  int n_per;
  int sid;
  int threadblock_count;
  // device-issued wire: the lane's arrival signal is waited INSIDE the kernel (thread 0 acquire spin, kill word)
  // instead of a front-end stream wait before the launch; nullptr = no wait
  uint64_t const *wait_sig = nullptr;
  uint64_t const *run_ptr = nullptr;  // device step slot (nullptr -> run_host)
  uint64_t run_host = 0;
  uint64_t const *kill = nullptr;
};
void a2av_bucket_map(A2AVBucketMapArguments const &args, cudaStream_t stream);
void a2av_bucket_scan(A2AVBucketScanArguments const &args, cudaStream_t stream);
// one thread waits until *sig >= the run id (device step slot run_ptr, else run_host; kill word polled), so that the
// wide fold of a remote lane launched behind it on the same stream never waits on the device
void a2av_lane_wait(uint64_t const *sig, uint64_t const *run_ptr, uint64_t run_host, uint64_t const *kill,
                    cudaStream_t stream);
void a2av_bucket_scatter(A2AVBucketScatterArguments const &args, cudaStream_t stream);
void a2av_combine_bucket_reduce(
    A2AVBucketReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream);


}  // namespace bytedance::flux
