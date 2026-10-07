//===- gemm_combine.cc -------------------------------------------- C++ ---===//
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

#include <algorithm>
#include "combine/ths_op/gemm_combine.h"
#include "flux/cuda/kernel_registry.h"

#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cuda_runtime_api.h>
#include <cutlass/fast_math.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <torch/all.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <nvshmemx.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>
#include <vector>

#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"

#include "flux/args/gemm_combine.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/cuda/ce_batch.h"
#include "core/wire_runtime.h"
#include "core/verdict.h"
#include "core/step_state.h"
#include "dwire/dwire.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "flux/utils.h"
#include "core/tuning.h"
#include "combine/topk_gather_rs.hpp"
#include "combine/workspace_helper.h"

namespace {
// the copy tile size for CombineWire. has nothing to do with the GEMM tile_size_m
static constexpr int kTileSizeM = 128, kTileSizeN = 1024;
// The dense-combine tile N drops to 512 when n_per_split is not
// 1024-aligned (e.g. H = 3584 = 7*512). Single policy point: the tile-barrier
// sizing and the split check derive from it.
static constexpr int kTileSizeNMin = 512;
static inline int
combine_tile_n(int n_dim, int n_split) {
  return (n_dim / n_split) % kTileSizeN == 0 ? kTileSizeN : kTileSizeNMin;
}
long
get_args_workspace_size(int problem_count) {
  using bytedance::flux::pad_to;
  constexpr int kAlignment = 128;
  // the workspace size
  int bytes =
      pad_to(sizeof(cutlass::gemm::GemmCoord) * problem_count, kAlignment) * 1  // problem_sizes
      + pad_to(sizeof(void *) * problem_count, kAlignment) * 4   // ptr_A/ptr_B/ptr_C/ptr_D
      + pad_to(sizeof(int64_t) * problem_count, kAlignment) * 5  // lda/ldb/ldc/ldd/ldr
      + pad_to(sizeof(float *) * problem_count, kAlignment) * 2  // scale_A/scale_B
      + pad_to(sizeof(int *) * problem_count, kAlignment) * 1    // scatter_D
      + pad_to(sizeof(int) * 1, kAlignment) * 1;                 // non_empty_problem_count
  return bytes;
}
c10::optional<std::vector<torch::Tensor>>
as_optional_vec(c10::optional<torch::Tensor> &t) {
  if (t.has_value()) {
    return c10::optional<std::vector<torch::Tensor>>{{t.value()}};
  }
  return {};
}

// Routed copies (rows of routing_idx) of the serving token bucket covering a step with m_full copies:
// tokens per rank t = m_full / (W * topk) rounded up to topk * 2^i, capped at the per-rank maximum
// (max_m / (W * topk)), i.e. python/zepp/serving.py token_buckets. Used only as the key of the GEMM
// hparams lookup, so an exact-size bucket shares the kernel of its covering power-of-two bucket.
static int64_t
covering_bucket_rows(int64_t m_full, int64_t world_size, int64_t topk, int64_t max_m) {
  const int64_t per_bucket_row = world_size * topk;
  const int64_t t = (m_full + per_bucket_row - 1) / per_bucket_row;
  const int64_t t_max = max_m / per_bucket_row;
  int64_t b = topk;
  while (b < t && b < t_max) {
    b *= 2;
  }
  return std::min(b, std::max(t_max, t)) * per_bucket_row;
}

// Wire-ordering rule: on libfabric/CXI the nbi put_signal exposes the flag before the data, so
// every inter-node combine put a consumer gates on is BLOCKING.
static inline void
flux_rs_put_signal(
    void *dst,
    const void *src,
    size_t bytes,
    uint64_t *sig,
    uint64_t val,
    int sig_op,
    int pe,
    cudaStream_t stream,
    int local_world_size,
    int my_node) {
  // INTER-NODE only: the ordering hazard is the libfabric/CXI proxy path.
  // Intra-node (P2P) puts stay nbi — CE-ordered, and the blocking on-stream
  // variant runs a device kernel that dereferences host-staged sources
  // (unspecified launch failure).
  const bool inter_node = (pe / local_world_size) != my_node;
  if (inter_node) {
    nvshmemx_putmem_signal_on_stream(dst, src, bytes, sig, val, sig_op, pe, stream);
  } else {
    nvshmemx_putmem_signal_nbi_on_stream(dst, src, bytes, sig, val, sig_op, pe, stream);
  }
}


void *
data_ptr_or(c10::optional<torch::Tensor> &t, void *other) {
  return t.has_value() ? t->data_ptr() : other;
}
int
get_rs_threadblock_count() {
  static int rs_num_blocks = 3;
  return rs_num_blocks;
}
// a2av_hier combine: SM budget of the pack / reduce kernels. Both are reserved
// out of the GEMM via sm_margin (the pack kernel is persistent and the per-split
// reduce must find free SMs while the GEMM still spins on later splits).
int
get_a2av_pack_blocks() {
  static int v = bytedance::flux::tuning::kCombinePackBlocks;
  return v;
}
int
get_a2av_reduce_blocks() {
  static int v = bytedance::flux::tuning::kCombineReduceBlocks;
  return v;
}
// The receivers of the REMOTE lanes (chain positions >= L) run with one block per SM of the device (capped at 128)
// instead of the reserved kCombineReduceBlocks. They are launched on the same reduce stream (the bucket-prefix order
// of the sequential lane waits stands) behind an event recorded after the combine GEMM, so a wide grid never shares
// the device with GEMM 2's resident persistent grid (a block that cannot be placed beside a resident GEMM block
// holds every later kernel of its priority); their in-kernel wait reads the device step slot
// (graph-safe). After GEMM 2 the remote returns are still in flight, so the wide fold only starts earlier relative to
// its own data, never before it.
int
get_a2av_tail_reduce_blocks() {
  static int v = [] {
    int dev = 0, x = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&x, cudaDevAttrMultiProcessorCount, dev);
    return x > 128 ? 128 : x;
  }();
  return v;
}
int
get_a2av_prered_blocks() {
  static int v = bytedance::flux::tuning::kCombinePreReduceBlocks;
  return v;
}
// Ring-consecutive dest nodes per wave (tile-quantization dial; waves never
// cross the ring wrap). 1 = per-node waves (finest pipelining; default — the
// tile waste hides under the wire whenever the wire dominates).
int
get_a2av_rs_wave_nodes() {
  static int v =
      std::max<int>(1, bytedance::flux::tuning::kCombineWaveNodes);
  return v;
}
// Epilogue-fused pack: the combine GEMM writes
// the dest-major send panel DIRECTLY through a ScatterD epilogue (pack inverse
// indices) with the topk gate coefficients pre-folded into the intermediate on
// the K side (cheaper than the pack's N-side pass when K < N; mathematically
// identical). Removes the pack kernel's full M x N read+write round-trip and
// one gating hop per wave (the pack degenerates to a 1-effective-CTA flag
// relay). Requires msplit.
int
get_a2av_rs_fused_pack() {
  static int v = 1;
  return v;
}
// Byte-adaptive wave collapse: the msplit dest-node row-split streams EVERY expert's w2 panel
// from HBM once per wave, so msplit pays (n_waves-1) extra weight passes for its early wire
// release; at small budgets the wire is too small to repay that. Per-iteration host-side rule
// from the cnt table the wave build already reads: collapse to the single-gate GEMM (one weight
// pass, n_waves = 0 downstream) when
//   (n_waves_planned - 1) * E * N * K * elt_w  >  ratio * remote_wire_bytes,
// ratio = tuning::kCombineWaveAdapt.
int
get_a2av_rs_wave_adapt() {
  static int v = bytedance::flux::tuning::kCombineWaveAdapt;
  return v;
}
// Effective L2 budget (MiB) for the auto chunk width: A100 has 40; leave
// headroom for the activation stream + pack/prered/reduce residents.
int
get_a2av_rs_chunk_l2_mib() {
  static int v = 30;
  return v;
}
// Own wave first: compute the OWN-node
// wave FIRST instead of last. Costs ~1/NN of the GEMM in wire-start delay;
// buys the receiver early own-node contributions, so tokens complete at
// their last REMOTE arrival (spread over the drain) instead of at the final
// own-node fold — the structural precondition for any arrival-order
// receiver to beat wait-all. 0 = own last.
int
get_a2av_rs_own_wave_first() {
  static int v = 0;
  return v;
}
// Rows claimed per cursor bump in the arrival-dynamic receiver (fold spread
// vs cursor-atomic traffic dial; folds trigger inside row processing, so a
// smaller chunk spreads token folds across more warps).
int
get_a2av_rs_recv_dyn_chunk() {
  static int v =
      std::max(1, 4);
  return v;
}

}  // namespace

namespace bytedance::flux::ths_op {

using torch::Tensor;

// The two a2av_hier routing-plan index builders are free functions of the derive
// (GemmCombineOpImpl::derive_combine_meta).
namespace {
  // Builds the mirror-layout gather indices for the a2av_hier combine, sharing
  // the a2av dispatch's exact ordering contract (same (.., expert, dst_row) keys,
  // same copy-index tie-break):
  // - pack_index [M_this_ep]: send-panel row -> gemm row. The send panel is
  //   (home_rank, expert, copy)-ordered == the dispatch's recv layout on this
  //   rank, so this is the inverse of the dispatch's sorted_gather_index
  //   arithmetic identity -- derived from the same offA/cumA/offR_of_A tables,
  //   NO sort.
  // - reduce_index [cpr]: local copy (t_local * topk + j) -> recv-panel row. The
  //   recv panel is (owner_rank, expert, copy)-ordered == the dispatch's
  //   send-buffer layout on this rank, i.e. globally (expert, copy)-sorted -- the
  //   inverse of the dispatch's pack permutation.
  // device-built combine tables (a2av_combine_tables): the plan kernels read the device arena and the
  // forward reads the device plan block.
  // Arena i64 layout: [cumA|offA|offR_of_A](3*nexG) [my_cum|e_base|h_base|e_cum](4*nex)
  // [expert_base|my_cnt_cum|conv_base|recv_off_C|recv_off_Cp|rem_base](n_i64)
  // totals[8] = m_this_ep, own_total, conv_total, wire_total, rem_total; i32: home_base[nex*W].
  struct CombineDevMeta {
    int64_t *i64 = nullptr;
    int32_t *i32 = nullptr;
    int64_t nexG = 0, nex = 0, n_i64 = 0;
    int64_t const *rtab() const { return i64 + 3 * nexG; }
    int64_t const *t64() const { return i64 + 3 * nexG + 4 * nex; }
    int64_t const *totals() const { return t64() + n_i64; }
  };

  // Persistent derive buffers: allocated once per op, sized from (W, E, NN) and max_m (provable
  // bounds, independent of the routing and of the resizable panel capacities), so no step allocates.
  // The builders return the whole buffers (the consumers read the true counts on the device).
  struct CombineDeriveBuffers {
    torch::Tensor pack_index;    // [max_m]: rows this rank computes <= every copy
    torch::Tensor reduce_index;  // [max_cpr]
    torch::Tensor scratch;       // compress plan scratch [2 (NN - 1) max_tpr + 2 max_tpr NN]
    torch::Tensor wire_ptr;      // [(NN - 1) max_tpr + 1]: one wire row per (remote home, token)
    torch::Tensor wire_copy;     // [(NN - 1) max_cpr]: the copies of NN - 1 homes
    torch::Tensor red_ptr;       // [max_tpr + 1]
    torch::Tensor red_row;       // [max_cpr]: own copies + contributing remote nodes <= topk per token
    int64_t n_i64 = 0;
  };

  // The device arena (dm), the device row count and the routing ids as the expert of every copy (e_of_copy).
  // Outputs land in the persistent buffers.
  std::pair<torch::Tensor, torch::Tensor>
  build_a2av_combine_indices(
      torch::Tensor const &routing_idx,
      int64_t m_full,
      int world_size,
      int rank,
      int64_t total_num_experts,
      int64_t ep_nexperts,   // experts per owner rank (tp == 1)
      CombineDeriveBuffers &bufs,
      CombineDevMeta const &dm,
      int32_t const *e_of_copy,
      int64_t const *verdict = nullptr) {  // deferred verdict block (degenerate step: empty indices)
    const int W = world_size;
    const int64_t nex = total_num_experts;
    const int64_t E_loc = ep_nexperts;  // experts per owner rank (tp == 1)
    const int64_t nexG = E_loc * W;
    const int64_t cpr = m_full / W;
    auto stream = c10::cuda::getCurrentCUDAStream();
    int64_t const *t64 = dm.i64;  // device-derived tables
    int64_t const *r64 = dm.rtab();
    // the kernel path covers every row count (no pack launch at zero rows, the reduce side is
    // independent of this rank's rows)
    CombinePlanArguments cargs{
        .routing_idx = routing_idx.data_ptr<int32_t>(),
        .cumA = t64,
        .offA = t64 + nexG,
        .offR_of_A = t64 + 2 * nexG,
        .expert_cum = r64 + 3 * nex,
        .my_cum = r64,
        .h_base = r64 + 2 * nex,
        .pack_index = bufs.pack_index.data_ptr<int32_t>(),
        .reduce_index = bufs.reduce_index.data_ptr<int32_t>(),
        .m_this_ep = cpr,  // launch-size hint only
        .cpr = cpr,
        .row0 = (int64_t)rank * cpr,
        .nexG = nexG,
        .nex = nex,
        .m_this_ep_dev = dm.totals(),
        .e_of_copy = e_of_copy,
        .verdict = verdict};
    a2av_combine_plan(cargs, stream);
    return {bufs.pack_index, bufs.reduce_index.narrow(0, 0, cpr)};
  }

  // Sort-free compress-plan builder: every ordering is arithmetic on the dispatch's stable scatter_index
  // + cnt/U prefix tables -- 4 kernels (a2av_compress_plan), no radix sorts, deterministic. The prefix
  // tables are the device arena (dm). e_of = the expert of every copy (the routing ids).
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  build_a2av_compress_indices_fast(
      torch::Tensor const &routing_idx,
      torch::Tensor const &e_of,
      int64_t m_full,
      int world_size,
      int nnodes,
      int local_world_size,
      int rank,
      int64_t total_num_experts,
      int64_t ep_nexperts,
      int topk,
      CombineDeriveBuffers &bufs,
      CombineDevMeta const &dm,
      int64_t const *verdict = nullptr) {  // deferred verdict block (degenerate step: empty CSRs)
    const int W = world_size;
    const int NN = nnodes;
    const int L = local_world_size;
    const int64_t nex = total_num_experts;
    const int64_t E_loc = ep_nexperts;
    const int64_t cpr = m_full / W;
    const int64_t ntok_local = cpr / topk;
    const int64_t tpr = ntok_local;  // tokens per rank (uniform homing)
    auto stream = c10::cuda::getCurrentCUDAStream();
    int64_t *d64 = const_cast<int64_t *>(dm.t64());  // device-derived tables
    int32_t const *home_base_dev = dm.i32;

    // ---- device scratch / outputs (persistent) ----
    const int64_t seg_tokens = (int64_t)(NN - 1) * tpr;
    int32_t *scr = bufs.scratch.data_ptr<int32_t>();
    A2AVCompressPlanArguments args{
        .scatter_index = routing_idx.data_ptr<int32_t>(),
        .e_of_copy = e_of.data_ptr<int32_t>(),
        .home_base = home_base_dev,
        .expert_base = d64,
        .conv_base = d64 + 2 * nex,
        .my_cnt_cum = d64 + nex,
        .recv_off_C = d64 + 2 * nex + (int64_t)(NN - 1) * L * E_loc,
        .recv_off_Cp = d64 + 2 * nex + (int64_t)(NN - 1) * L * E_loc + W,
        .rem_base = d64 + 2 * nex + (int64_t)(NN - 1) * L * E_loc + 2 * W,
        .conv_count = scr,
        .wire_row_of = scr + seg_tokens,
        .red_flags = scr + 2 * seg_tokens,
        .rem_pos = scr + 2 * seg_tokens + tpr * NN,
        .wire_ptr = bufs.wire_ptr.data_ptr<int32_t>(),
        .wire_copy = bufs.wire_copy.data_ptr<int32_t>(),
        .red_ptr = bufs.red_ptr.data_ptr<int32_t>(),
        .red_row = bufs.red_row.data_ptr<int32_t>(),
        .piece_of_e = nullptr,
        .n_pieces = 0,
        .wire_piece = nullptr,
        .wire_piece_start = nullptr,
        .m_full = m_full,
        .topk = topk,
        .world_size = W,
        .nnodes = NN,
        .local_world_size = L,
        .rank = rank,
        .nexperts = nex,
        .ep_nexperts = E_loc,
        .verdict = verdict};
    a2av_compress_plan(args, stream);
    auto red_ptr = bufs.red_ptr.narrow(0, 0, ntok_local + 1);
    return {bufs.wire_ptr, bufs.wire_copy, red_ptr, bufs.red_row};
  }
}  // namespace

class CombineWire::CombineWireImpl {
 private:
  std::shared_ptr<Group> tp_group;
  int32_t rank;
  int32_t world_size;  // the total world size
  const int nnodes;
  const int node_idx;
  const int local_rank;
  const int local_world_size;
  int32_t max_m;
  int32_t n_dim;
  int32_t topk;
  at::ScalarType output_dtype;
  const int ep_nexperts;
  const int ep_world_size;  // the world size of expert parallel
  const int n_split;

  // intra-node buffers: tensor lists / pointer arrays are [local_world_size],
  // indexed by local rank (== global rank when nnodes == 1)
  torch::Tensor reduce_buffer;
  std::vector<torch::Tensor> reduce_buffers;
  torch::Tensor reduce_buffer_dptrs;
  torch::Tensor tile_barrier;
  std::vector<torch::Tensor> tile_barriers;
  torch::Tensor tile_barrier_dptrs;
  torch::Tensor barrier;
  std::vector<torch::Tensor> barriers;
  int **barrier_dev_ptrs = nullptr;

  // cuStreamWriteValue/WaitValue64 need real device addresses, so use
  // cutlass::DeviceAllocation, not torch tensors (expandable_segments VA issue).
  // Chunk-ready flags: the pack kernel writes the step's run_id_ (epoch words, zeroed at allocation
  // only, never reset per step); the counters return to zero at the end of every pack launch.
  cutlass::DeviceAllocation<uint64_t> group_flags;  // [nnodes * n_split]
  cutlass::DeviceAllocation<int> group_counters;    // [nnodes * n_split]
  c10::cuda::CUDAStream internode_stream;
  // the compress wire ladder's blocking puts for (sid, tn) cells are pairwise independent
  // (distinct wire-panel segments, destinations and recv signals); they round-robin over
  // rs_wire_streams_ internode streams so the per-target blocking puts overlap
  int rs_wire_streams_ = 1;
  std::vector<c10::cuda::CUDAStream> internode_streams2_;   // rs_wire_streams_ - 1 extras
  cudaEvent_t a2av_inter2_done_ = nullptr;
  cudaEvent_t staging_reset_event;  // start of the step on the caller's stream
  // the gather-rs stream joins the intra / conv / inter-node lanes through this device word (written after the
  // lanes join)
  uint64_t *wire_done_ = nullptr;
  // start of the step for the lanes: the caller's stream writes run_id_ here where it records
  // staging_reset_event
  uint64_t *staging_word_ = nullptr;
  // peer-mapped bases of the symmetric buffers the lanes write into, rebuilt at init and resize
  PeerTable peers_;
  void
  build_peer_table() {
    this->peers_.clear();
    for (torch::Tensor *t : {&this->a2av_recv_panel_, &this->a2av_recv_signals_,
                             &this->a2av_conv_panel_, &this->a2av_conv_signals_}) {
      if (t->defined()) {
        this->peers_.add(t->data_ptr(), t->nbytes(), this->world_size);
      }
    }
    // device-issued wire: per local rank, its receive panel, convergence panel, receive and convergence signals
    const int L = this->local_world_size, my_node = this->rank / L;
    std::vector<int64_t> tab(4 * (size_t)L, 0);
    int i = 0;
    for (torch::Tensor *t : {&this->a2av_recv_panel_, &this->a2av_conv_panel_, &this->a2av_recv_signals_,
                             &this->a2av_conv_signals_}) {
      for (int dl = 0; dl < L && t->defined(); dl++) {
        const int pe = my_node * L + dl;
        tab[(size_t)i * L + dl] =
            reinterpret_cast<int64_t>(pe == this->rank ? t->data_ptr() : nvshmem_ptr(t->data_ptr(), pe));
      }
      i++;
    }
    this->dwire_tables_ = torch::from_blob(tab.data(), {4, (int64_t)L}, torch::kLong).to(torch::kCUDA);
    if (!this->dwire_push_counters_.defined()) {
      this->dwire_push_counters_ = torch::zeros(
          {(int64_t)this->world_size * this->n_split}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
    }
  }
  torch::Tensor dwire_tables_;          // int64 [4][L]: recv panel, conv panel, recv signals, conv signals
  torch::Tensor dwire_push_counters_;   // int32 [W * n_split]
  // front-end wait of a lane stream on a word this rank's caller stream writes (epoch semantics)
  void
  wait_word(cudaStream_t s, uint64_t *word, uint64_t run) {
    CU_CHECK(CUStreamWaitValue64(s, reinterpret_cast<CUdeviceptr>(word), run, CU_STREAM_WAIT_VALUE_GEQ));
  }
  uint64_t run_id_ = 0;

 public:
  uint64_t
  run_id() const {
    return run_id_;
  }
  void
  advance_run_id(int64_t n) {
    FLUX_CHECK_GE(n, 0) << "run ids never go back";
    run_id_ += (uint64_t)n;
  }

  // the combine GEMM's end event, set_tail_reduce() before each run(): the remote-lane receivers wait for it
  cudaEvent_t tail_gemm_end_ = nullptr;
  void
  set_tail_reduce(cudaEvent_t gemm_end) {
    this->tail_gemm_end_ = gemm_end;
  }

  // M-split waves: per-iteration pack-gate state, armed by
  // set_msplit_waves() before each run(); 0 = single-split gate
  int msplit_n_waves_ = 0;
  // pieces: per-expert-chunk cascade flag count the pack must wait at
  // entry (no-split build); 0 = per-wave gating
  int msplit_chunk_flags_ = 0;
  int msplit_wave_of_node_[kA2AVMaxNodes] = {};
  int msplit_node_order_[kA2AVMaxNodes] = {};
  // Combine plan block of the next run(), armed by set_plan_block(): the count-derived tables come from
  // the device plan block (host copy, already waited on by the caller; nullptr on a deferred-verdict
  // step of the device wire) and the bucket lane table from its device twin.
  int64_t const *plan_host_ = nullptr;      // [layout.total] int64, then [2 W + 1] int32 lanes
  int32_t const *plan_lanes_dev_ = nullptr;  // device lane table: lane_off [W + 1] | chain_pos [W]
  // deferred verdict (set_deferred): device plan block, wave-adapt decision, verdict block of the next run()
  int64_t const *dfr_plan_ = nullptr;
  int32_t const *dfr_dec_ = nullptr;
  int64_t const *dfr_verdict_ = nullptr;

  // a2av_hier combine state. Layouts mirror the a2av dispatch exactly:
  // the send panel is (home_rank, expert, copy)-ordered (== the dispatch's recv
  // layout), the recv panel is (owner_rank, expert, copy)-ordered (== the
  // dispatch's send layout), so every copy lands back at its dispatch pack
  // position and the pack/reduce gather indices are the inverses of the
  // dispatch's index math.
  // a2av_hier_compress: one partial per (token, source node) crosses the wire.
  // Source rank (n, lr) owns all wire rows to rank (tn, lr): the node's copies
  // converge on it (conv panel, NVLink), a persistent pre-reduce kernel merges
  // them per token into the wire panel, and the inter ladder puts straight into
  // the destination's recv panel (C' image) -- no destination gateway hop.
  // False when nnodes == 1 (degrades to plain a2av_hier: zero wire savings).
  const bool a2av_compress_;
  // Epilogue-fused pack: the GEMM scatters the send panel
  // directly; the pack kernel runs as a flag relay and applies NO vec_scale
  // (pre-folded into the intermediate by the caller)
  const bool a2av_fused_pack_;
  CombineOptions options_{};
  int64_t a2av_send_rows_ = 0;   // send panel row capacity per split (routing-dependent load)
  int64_t a2av_recv_rows_ = 0;   // recv panel rows per split: exactly max_m / world_size
  int64_t a2av_conv_rows_ = 0;   // compress: convergence panel row capacity per split
  int64_t a2av_wire_rows_ = 0;   // compress: wire panel row capacity per split
  torch::Tensor a2av_send_panel_;       // [n_split, a2av_send_rows_, n_per] symmetric
  torch::Tensor a2av_recv_panel_;       // [n_split, a2av_recv_rows_, n_per] symmetric
  torch::Tensor a2av_conv_panel_;       // compress: [n_split, conv_rows, n_per] symmetric
  torch::Tensor a2av_wire_panel_;       // compress: [n_split, wire_rows, n_per] symmetric
  torch::Tensor a2av_recv_signals_;     // uint64 [world_size * n_split], epoch, never reset
  torch::Tensor a2av_conv_signals_;     // compress: uint64 [L * NN * n_split], epoch, never reset
  // compress: pre-reduce kernel -> wire-ready flags, run_id-valued like group_flags (no reset)
  cutlass::DeviceAllocation<uint64_t> wire_flags_;  // [nnodes * n_split]
  cutlass::DeviceAllocation<int> wire_counters_;    // [nnodes * n_split]
  // lane-chain receiver buffers (plain device memory)
  torch::Tensor a2av_scratch_fp32_;   // [max ntokens_local, n] fp32 accumulator
  torch::Tensor a2av_token_of_row_;   // [a2av_recv_rows_] int32 recv row -> token
  // bucket receiver buffers (compress only): plain device memory for the
  // per-iteration completion map
  torch::Tensor a2av_bucket_comp_;    // [ntok_max] int32 completion position
  torch::Tensor a2av_bucket_tok_;     // [ntok_max] int32 tokens by bucket
  torch::Tensor a2av_bucket_meta_;    // [3*kA2AVMaxWorld + 2] int32: cnt | cur | ptr
  std::optional<c10::cuda::CUDAStream> a2av_intra_stream_;    // intra-node put ladder (CEs)
  std::optional<c10::cuda::CUDAStream> a2av_reduce_stream_;   // signal waits + per-split reduce
  std::optional<c10::cuda::CUDAStream> a2av_conv_stream_;     // compress: convergence put ladder
  std::optional<c10::cuda::CUDAStream> a2av_prered_stream_;   // compress: resident pre-reduce kernel
  cudaEvent_t a2av_inter_done_ = nullptr;
  cudaEvent_t a2av_reduce_done_ = nullptr;
  cudaEvent_t a2av_conv_done_ = nullptr;
  cudaEvent_t a2av_prered_done_ = nullptr;

  bool buffer_initialized = false;

 private:
  void
  init_buffer_once(at::ScalarType dtype) {
    if (this->buffer_initialized)
      return;
    {
      // a2av mode skips every dense-only buffer (ring reduce buffers, tile
      // barriers, dense staging, internode signals): peers never write partials,
      // only whole copies into the recv panel. The ctor flag is uniform across
      // ranks, so skipping the collective allocations is collectively consistent.
      const int64_t n_per = this->n_dim / this->n_split;
      this->a2av_recv_rows_ = this->max_m / this->world_size;  // exact: topk copies per token
      this->a2av_send_rows_ = this->options_.max_send_rows;
      this->a2av_send_panel_ =
          nvshmem_create_tensor({this->n_split, this->a2av_send_rows_, n_per}, dtype);
      this->a2av_recv_panel_ =
          nvshmem_create_tensor({this->n_split, this->a2av_recv_rows_, n_per}, dtype);
      this->a2av_recv_signals_ = nvshmem_create_tensor(
          {(int64_t)this->world_size * this->n_split}, at::ScalarType::Long, true);
      if (this->a2av_compress_) {
        // compress replaces the destination-side staging/gateway machinery with
        // source-side convergence + wire panels; the flag is a uniform ctor
        // input so the collective allocation swap is consistent across ranks
        this->a2av_conv_rows_ = this->options_.max_conv_rows;
        this->a2av_wire_rows_ = this->options_.max_wire_rows;
        this->a2av_conv_panel_ =
            nvshmem_create_tensor({this->n_split, this->a2av_conv_rows_, n_per}, dtype);
        this->a2av_wire_panel_ =
            nvshmem_create_tensor({this->n_split, this->a2av_wire_rows_, n_per}, dtype);
        this->a2av_conv_signals_ = nvshmem_create_tensor(
            {(int64_t)this->local_world_size * this->nnodes * this->n_split},
            at::ScalarType::Long,
            true);
        this->wire_flags_.reset(this->nnodes * this->n_split);
        this->wire_counters_.reset(this->nnodes * this->n_split);
        CUDA_CHECK(cudaMemset(
            this->wire_flags_.get(), 0, sizeof(uint64_t) * this->nnodes * this->n_split));
        CUDA_CHECK(cudaMemset(
            this->wire_counters_.get(), 0, sizeof(int) * this->nnodes * this->n_split));
        {
          const int64_t ntok_max = (int64_t)this->max_m / this->topk / this->world_size;
          auto opt_i32 = at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int);
          this->a2av_bucket_comp_ =
              empty_with_uninitialized_data(std::vector<int64_t>{ntok_max}, opt_i32);
          this->a2av_bucket_tok_ =
              empty_with_uninitialized_data(std::vector<int64_t>{ntok_max}, opt_i32);
          this->a2av_bucket_meta_ = empty_with_uninitialized_data(
              std::vector<int64_t>{3 * kA2AVMaxWorld + 2}, opt_i32);
        }
      }
      // chunk-ready flags per (dest_node, sid) -- allocated for nnodes == 1 too:
      // the intra-node ladder gates on the own-node flag
      this->group_flags.reset(this->nnodes * this->n_split);
      this->group_counters.reset(this->nnodes * this->n_split);
      CUDA_CHECK(cudaMemset(
          this->group_flags.get(), 0, sizeof(uint64_t) * this->nnodes * this->n_split));
      CUDA_CHECK(
          cudaMemset(this->group_counters.get(), 0, sizeof(int) * this->nnodes * this->n_split));
      // Preload every kernel this data path launches: the combine's own
      // kernels (attribute queries force the module loads) plus NVSHMEM's
      // on-stream transfer/signal kernels (primed by issuing one real op per
      // transport path). Under CUDA_MODULE_LOADING=LAZY a kernel's module is
      // loaded at its FIRST launch, and the compress schedule puts a persistent
      // spin kernel (the pre-reduce) on the device BEFORE the epoch's first
      // NVSHMEM on-stream call: that first-launch load never completes behind
      // the never-exiting resident kernel and the epoch deadlocks (without
      // compress the only spin kernel, the pack, drains once the GEMM
      // finishes). The ctor runs with an idle device, so every load below is
      // trivial. The priming ops write
      // SET 0 over zero-initialized signal slots (a no-op value), and
      // nvshmem_barrier_all() orders their remote delivery before any epoch.
      // The primitive set (flux_shm.cc) covers every variant the combine
      // issues: the BARE signal_op to self, to a P2P peer and to a remote node
      // (the ladders' zero-row lanes, always-signal invariant; the P2P and the
      // inter-node bare signal are distinct transport kernels and each needs
      // its own priming), the nbi put to a P2P peer and the blocking put to a remote
      // node (flux_rs_put_signal), plus the variants the dispatch issues.
      a2av_combine_preload(from_torch_dtype(dtype));
      preload_all_kernels();
      nvshmem_prime_primitives(
          (uint64_t *)this->a2av_recv_signals_.data_ptr(),
          this->rank,
          this->local_world_size,
          this->nnodes,
          c10::cuda::getCurrentCUDAStream());
      torch::cuda::synchronize();
      this->build_peer_table();
      this->buffer_initialized = true;
      return;
    }
    std::vector<void *> hptrs(this->local_world_size, nullptr);
    const int ptr_bytes = sizeof(void *) * this->local_world_size;
    // initialize the output buffer
    this->reduce_buffers = flux_create_tensor_list(
        {this->max_m / this->topk, this->n_dim}, dtype, this->tp_group.get());
    FLUX_CHECK_EQ((int)this->reduce_buffers.size(), this->local_world_size);
    this->reduce_buffer = this->reduce_buffers[this->local_rank];
    for (int i = 0; i < this->local_world_size; ++i) {
      hptrs[i] = reduce_buffers[i].data_ptr();
    }
    CHECK(!reduce_buffer_dptrs.defined());
    this->reduce_buffer_dptrs =
        torch::empty({ptr_bytes}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Byte));
    CUDA_CHECK(cudaMemcpy(
        this->reduce_buffer_dptrs.data_ptr(), hptrs.data(), ptr_bytes, cudaMemcpyHostToDevice));
    if (this->nnodes > 1) {
      this->group_flags.reset(this->nnodes * this->n_split);
      this->group_counters.reset(this->nnodes * this->n_split);
      CUDA_CHECK(cudaMemset(
          this->group_flags.get(), 0, sizeof(uint64_t) * this->nnodes * this->n_split));
      CUDA_CHECK(
          cudaMemset(this->group_counters.get(), 0, sizeof(int) * this->nnodes * this->n_split));
    }
    torch::cuda::synchronize();
    this->buffer_initialized = true;
  }
  int
  get_tile_barrier_size(int num_tiles) const {
    return num_tiles;
  }

  void
  create_rs_barrier() {
    int m_tiles_at_most = (this->max_m + kTileSizeM - 1) / kTileSizeM + this->ep_nexperts;
    const int tile_n = combine_tile_n(this->n_dim, this->n_split);
    int n_tiles = (this->n_dim + tile_n - 1) / tile_n;
    int num_tiles = m_tiles_at_most * n_tiles;

    int tile_barrier_size = get_tile_barrier_size(num_tiles);
    if (!this->tile_barrier.defined() || this->tile_barrier.numel() < tile_barrier_size) {
      // initialize the tile_barrier
      this->tile_barriers =
          flux_create_tensor_list({tile_barrier_size}, at::ScalarType::Int, this->tp_group.get());
      FLUX_CHECK_EQ((int)this->tile_barriers.size(), this->local_world_size);
      this->tile_barrier = this->tile_barriers[this->local_rank];
      std::vector<int *> hptrs(this->local_world_size, nullptr);
      const int ptr_bytes = sizeof(int *) * this->local_world_size;
      for (int i = 0; i < this->local_world_size; ++i) {
        hptrs[i] = (int *)this->tile_barriers[i].data_ptr();
      }
      CHECK(!tile_barrier_dptrs.defined());
      this->tile_barrier_dptrs =
          torch::empty({ptr_bytes}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Byte));
      CUDA_CHECK(cudaMemcpy(
          this->tile_barrier_dptrs.data_ptr(), hptrs.data(), ptr_bytes, cudaMemcpyHostToDevice));
    }
  }

  c10::cuda::CUDAStream
  create_internode_stream() const {
    at::cuda::CUDAGuard guard(at::cuda::current_device());
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return at::cuda::getStreamFromExternal(stream, at::cuda::current_device());
  }

 public:
  void *
  send_panel_ptr() {
    this->init_buffer_once(this->output_dtype);  // idempotent
    return this->a2av_send_panel_.data_ptr();
  }
  int64_t
  send_panel_rows() {
    return this->a2av_send_rows_;
  }
  void
  ensure_buffers(at::ScalarType dtype) {
    this->init_buffer_once(dtype);
  }

  void
  set_msplit_waves(
      std::vector<int> const &wave_of_node,
      std::vector<int> const &node_order,
      int n_waves,
      int n_chunk_flags = 0) {
    this->msplit_chunk_flags_ = n_chunk_flags > 0 ? n_chunk_flags : 0;
    if (n_waves <= 0) {
      this->msplit_n_waves_ = 0;
      return;
    }
    FLUX_CHECK_EQ(this->n_split, 1) << "msplit requires n_split == 1";
    FLUX_CHECK_EQ((int)wave_of_node.size(), this->nnodes);
    FLUX_CHECK_EQ((int)node_order.size(), this->nnodes);
    FLUX_CHECK_LE(this->nnodes, kA2AVMaxNodes);
    const int my_node2 = this->rank / this->local_world_size;
    FLUX_CHECK(
        node_order[this->nnodes - 1] == my_node2 || node_order[0] == my_node2)
        << "own node must be the first or final schedule step";
    this->msplit_n_waves_ = n_waves;
    for (int i = 0; i < this->nnodes; i++) {
      FLUX_CHECK_LT(wave_of_node[i], n_waves);
      this->msplit_wave_of_node_[i] = wave_of_node[i];
      this->msplit_node_order_[i] = node_order[i];
    }
  }

  // capacity rows of the send / convergence / wire panels
  void
  panel_rows(int64_t *send, int64_t *conv, int64_t *wire) const {
    *send = this->a2av_send_rows_;
    *conv = this->a2av_conv_rows_;
    *wire = this->a2av_wire_rows_;
  }

  // deferred verdict (core/verdict.h) for the next run(), one-shot: outside the wire, the combine reads the
  // device plan block, the device wave-adapt decision and the verdict block instead of host counts
  void
  set_deferred(int64_t const *plan_dev, int32_t const *msplit_dec, int64_t const *verdict) {
    this->dfr_plan_ = plan_dev;
    this->dfr_dec_ = msplit_dec;
    this->dfr_verdict_ = verdict;
  }

  // arm the plan block of the next run() (see plan_host_)
  void
  set_plan_block(int64_t const *host_block, int32_t const *lanes_dev) {
    FLUX_CHECK(lanes_dev != nullptr) << "combine plan block armed without its device lane table";
    this->plan_host_ = host_block;
    this->plan_lanes_dev_ = lanes_dev;
  }

  CombineWireImpl(
      std::shared_ptr<Group> tp_group_,
      int max_m,
      int n_dim,
      int topk,
      at::ScalarType output_dtype,
      int ep_nexperts,
      int ep_world_size,
      const std::vector<torch::Tensor> &barriers,
      int n_split_,
      CombineOptions const &options,
      int nnodes_ = 1,
      bool a2av_compress = false)
      : tp_group(tp_group_),
        rank(tp_group_->get_rank()),
        world_size(tp_group_->get_size()),
        nnodes(nnodes_),
        node_idx(DistEnv(tp_group_->get_rank(), tp_group_->get_size(), nnodes_).node_idx),
        local_rank(DistEnv(tp_group_->get_rank(), tp_group_->get_size(), nnodes_).local_rank),
        local_world_size(tp_group_->get_size() / nnodes_),
        max_m(max_m),
        n_dim(n_dim),
        topk(topk),
        output_dtype(output_dtype),
        ep_nexperts(ep_nexperts),
        ep_world_size(ep_world_size),
        n_split(n_split_),
        internode_stream(create_internode_stream()),
        a2av_compress_(a2av_compress && nnodes_ > 1),
        a2av_fused_pack_(
            nnodes_ > 1 && get_a2av_rs_fused_pack() != 0),
        barriers(barriers) {
    {
      FLUX_CHECK_EQ(n_split_, 1) << "the bucketed receiver requires n_split == 1";
    }
    this->options_ = options;
    FLUX_CHECK_GE(nnodes, 1);
    FLUX_CHECK_DIV(world_size, nnodes);
    if (nnodes > 1) {
      FLUX_CHECK_DIV(max_m / topk, world_size);
      FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == local_rank)
          << "rank layout must be node-contiguous (rank = node_idx * local_world_size + "
             "local_rank)";
    }
    if (a2av_compress && nnodes_ == 1 && this->rank == 0) {
      FLUX_LOG_FIRST_N(INFO, 1)
          << "compress mode on a single node degrades to the plain wire "
             "(node-level dedup saves zero wire bytes)\n";
    }
    {
      FLUX_CHECK_EQ(ep_world_size, world_size) << "the fused combine requires EP == world (tp == 1)";
      FLUX_CHECK_DIV(max_m, world_size);
      FLUX_CHECK(
          output_dtype == at::ScalarType::Half || output_dtype == at::ScalarType::BFloat16)
          << "the fused combine supports fp16/bf16 only";
      FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == local_rank)
          << "rank layout must be node-contiguous (rank = node_idx * local_world_size + "
             "local_rank)";
      this->a2av_intra_stream_ = create_internode_stream();
      this->a2av_reduce_stream_ = create_internode_stream();
      if (this->a2av_compress_) {
        this->a2av_conv_stream_ = create_internode_stream();
        this->a2av_prered_stream_ = create_internode_stream();
        CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_conv_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_prered_done_, cudaEventDisableTiming));
      }
      CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_inter_done_, cudaEventDisableTiming));
      CUDA_CHECK(cudaMalloc(&this->wire_done_, sizeof(uint64_t)));
      CUDA_CHECK(cudaMemset(this->wire_done_, 0, sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&this->staging_word_, sizeof(uint64_t)));
      CUDA_CHECK(cudaMemset(this->staging_word_, 0, sizeof(uint64_t)));
      wire_runtime_init();  // kill word + spin-kernel preload, on an idle device
      // wire lanes: S internode streams round-robin the (sid, tn) cells (gi % S)
      this->rs_wire_streams_ = bytedance::flux::tuning::kCombineWireStreams;
      if (this->rs_wire_streams_ < 1) {
        this->rs_wire_streams_ = 1;
      }
      if (this->rs_wire_streams_ > 32) {
        this->rs_wire_streams_ = 32;
      }
      for (int i = 1; i < this->rs_wire_streams_; i++) {
        this->internode_streams2_.push_back(create_internode_stream());
      }
      if (this->rs_wire_streams_ > 1) {
        CUDA_CHECK(
            cudaEventCreateWithFlags(&this->a2av_inter2_done_, cudaEventDisableTiming));
      }
      CUDA_CHECK(cudaEventCreateWithFlags(&this->a2av_reduce_done_, cudaEventDisableTiming));
    }
    this->init_buffer_once(output_dtype);

    std::vector<void *> barrier_ptrs(this->local_world_size, nullptr);
    FLUX_CHECK_EQ((int)this->barriers.size(), this->local_world_size);
    for (int i = 0; i < this->local_world_size; i++) {
      barrier_ptrs[i] = this->barriers[i].data_ptr();
    }
    CUDA_CHECK(cudaMalloc(&this->barrier_dev_ptrs, this->local_world_size * sizeof(void *)));
    CUDA_CHECK(cudaMemcpy(
        this->barrier_dev_ptrs,
        barrier_ptrs.data(),
        this->local_world_size * sizeof(void *),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->staging_reset_event, cudaEventDisableTiming));
    torch::cuda::synchronize();  // we don't assume create/run on the same stream so sync is safe
    this->barrier = this->barriers[this->local_rank];
  }

  ~CombineWireImpl() {
    CUDA_CHECK(cudaDeviceSynchronize());
    if (this->wire_done_ != nullptr) {
      CUDA_CHECK(cudaFree(this->wire_done_));
    }
    if (this->staging_word_ != nullptr) {
      CUDA_CHECK(cudaFree(this->staging_word_));
    }
    CUDA_CHECK(cudaEventDestroy(this->staging_reset_event));
    CUDA_CHECK(cudaStreamDestroy(this->internode_stream));
    for (auto &s : {this->a2av_intra_stream_, this->a2av_reduce_stream_,
                    this->a2av_conv_stream_, this->a2av_prered_stream_}) {
      if (s.has_value()) {
        CUDA_CHECK(cudaStreamDestroy(s.value()));
      }
    }
    for (auto &sq : this->internode_streams2_) {
      CUDA_CHECK(cudaStreamDestroy(sq));
    }
    if (this->a2av_inter2_done_ != nullptr) {
      CUDA_CHECK(cudaEventDestroy(this->a2av_inter2_done_));
    }
    for (auto e : {this->a2av_inter_done_,
                   this->a2av_reduce_done_, this->a2av_conv_done_, this->a2av_prered_done_}) {
      if (e != nullptr) {
        CUDA_CHECK(cudaEventDestroy(e));
      }
    }
    if (this->barrier_dev_ptrs != nullptr) {
      CUDA_CHECK(cudaFree(this->barrier_dev_ptrs));
    }
  }

  // a2av_hier combine: pack (persistent kernel, split-major, behind the GEMM
  // cascade flags) -> host put ladders on copy engines / NIC (intra direct,
  // inter-node aggregated via same-local-rank gateways, gateway forwards paced
  // by zero-SM cuStreamWaitValue64) -> per-split destination topk reduce once
  // that split's W per-source recv signals have fired. All host waits are
  // enqueued AFTER the pack kernel launch and in dependency order (intra/inter
  // ladders, then gateway, then reduce): under CUDA_DEVICE_MAX_CONNECTIONS=1
  // every enqueued front-end wait can block later ops in the shared channel, so
  // enqueue order must be an executable schedule.
  torch::Tensor
  combine(
      torch::Tensor gemm_out,
      torch::Tensor output,
      torch::Tensor const &splits_per_source,
      torch::Tensor const &pack_index,
      torch::Tensor const &reduce_index,
      c10::optional<torch::Tensor> const &unique_counts,
      c10::optional<std::vector<torch::Tensor>> const &wire_csr,
      c10::optional<std::vector<torch::Tensor>> const &reduce_csr,
      c10::optional<std::vector<torch::Tensor>> const &output_vec_scales,
      int m_full,
      int num_thread_blocks,
      cudaStream_t stream_raw) {
    const int W = this->world_size;
    const int L = this->local_world_size;
    const int NN = this->nnodes;
    const int my_node = this->node_idx;
    const int my_lr = this->local_rank;
    DistEnv dist_env(this->rank, W, NN);
    const int nex_total = this->ep_nexperts * this->ep_world_size;
    const int E_loc = nex_total / W;  // experts per owner rank (EP == world)
    const int64_t cpr = (int64_t)m_full / W;  // copies homed on each rank
    const int64_t M_this_ep = gemm_out.size(0);
    const int64_t n_per = this->n_dim / this->n_split;
    auto dtype = gemm_out.scalar_type();
    const int64_t row_bytes = n_per * c10::elementSize(dtype);

    FLUX_CHECK(splits_per_source.device().is_cpu()) << "splits_per_source must be a CPU tensor";
    CHECK_2D(splits_per_source, W, nex_total);
    FLUX_CHECK(splits_per_source.is_contiguous());
    FLUX_CHECK(splits_per_source.scalar_type() == at::ScalarType::Int);
    CHECK_INPUT(pack_index, at::ScalarType::Int);
    CHECK_INPUT(reduce_index, at::ScalarType::Int);
    FLUX_CHECK_GE(pack_index.numel(), M_this_ep) << "pack_index shorter than the gemm rows";
    CHECK_1D(reduce_index, cpr);
    FLUX_CHECK_LE(cpr, this->a2av_recv_rows_);

    // Plan block: every count-derived size and offset below is an indexed read of the device plan block
    // (host copy); without a host copy (no_host) nothing below reads a count on the host.
    // device-issued wire on a deferred-verdict step: no host copy (the kernels read the device block)
    FLUX_CHECK(this->plan_host_ != nullptr || this->dfr_plan_ != nullptr)
        << "combine: device plan block not armed";
    int64_t const *pb1 = this->plan_host_;
    const bool no_host = pb1 == nullptr;
    const CombinePlanLayout lay = combine_plan_layout(W, NN, E_loc);
    // Deferred verdict (set_deferred, one-shot): outside the wire nothing below reads a count on the host;
    // the pack, the pre-reduce and the receivers take their sizes from the device plan block, the forward's
    // checks of the block are device asserts (a2av_combine_plan_block), gemm_out is capacity-sized
    int64_t const *dplan = this->dfr_plan_;
    int32_t const *ddec = this->dfr_dec_;
    int64_t const *dverdict = this->dfr_verdict_;
    this->dfr_plan_ = nullptr;
    this->dfr_dec_ = nullptr;
    this->dfr_verdict_ = nullptr;
    const bool deferred = dplan != nullptr;

    if (!no_host && !deferred) {
      // the consistency checks on the device-computed words (identical on every rank)
      FLUX_CHECK_EQ(pb1[CombinePlanLayout::kM], M_this_ep)
          << "splits_per_source disagrees with gemm rows";
      FLUX_CHECK(pb1[CombinePlanLayout::kColMin] == cpr && pb1[CombinePlanLayout::kColMax] == cpr)
          << "chunk matrix column sums in [" << pb1[CombinePlanLayout::kColMin] << ", "
          << pb1[CombinePlanLayout::kColMax] << "] != ntokens_local * topk " << cpr;
      FLUX_CHECK_LE(pb1[CombinePlanLayout::kMaxSend], this->a2av_send_rows_)
          << "combine send panel overflow; raise CombineOptions.max_send_rows";
    }

    if (this->a2av_compress_) {
      FLUX_CHECK(unique_counts.has_value())
          << "a2av_hier_compress requires unique_counts ([W, nnodes] int32 CPU)";
      FLUX_CHECK(wire_csr.has_value() && reduce_csr.has_value())
          << "a2av_hier_compress requires the wire/reduce CSRs (built by the gather-rs "
             "op or passed as precomputed routing-plan inputs)";
    }
    if (this->a2av_compress_ && !deferred && pb1 != nullptr) {
      FLUX_CHECK_LE(pb1[CombinePlanLayout::kMaxConv], this->a2av_conv_rows_)
          << "a2av_hier_compress conv panel overflow; raise CombineOptions.max_conv_rows";
      FLUX_CHECK_LE(pb1[CombinePlanLayout::kMaxWire], this->a2av_wire_rows_)
          << "a2av_hier_compress wire panel overflow; raise CombineOptions.max_wire_rows";
      FLUX_CHECK_LE(pb1[lay.lane_off + W], cpr) << "C' image exceeds recv panel";
      if (pb1[CombinePlanLayout::kConvTotal] == 0) {
        FLUX_CHECK_EQ(pb1[CombinePlanLayout::kWireTotal], 0)
            << "compress: conv_total == 0 but unique_counts claims "
            << pb1[CombinePlanLayout::kWireTotal] << " wire rows (inconsistent transposed U)";
      }
    }
    // the count-derived values every consumer below reads, from the device plan block
    auto rows_to = [&](int d) -> int64_t {  // C[rank][d]
      return pb1[lay.send_rows + d];
    };
    auto send_off_at = [&](int d) -> int64_t {
      return pb1[lay.send_off + d];
    };
    auto dst_off_at = [&](int d) -> int64_t {  // C' recv offset of my rows at d
      return pb1[lay.dst_off + d];
    };
    auto conv_dst_at = [&](int dl, int tn) -> int64_t {  // conv panel offset at gateway (my_node, dl)
      return pb1[lay.conv_dst + tn * L + dl];
    };
    auto wire_seg_at = [&](int tn) -> int64_t {  // my wire panel segment of tn != my_node
      return pb1[lay.wire_seg + (tn < my_node ? tn : tn - 1)];
    };
    auto wire_rows_at = [&](int tn) -> int64_t {  // U[(tn, my_lr)][my_node]
      return pb1[lay.wire_rows + tn];
    };
    auto lane_rows = [&](int sq) -> int64_t {  // C'[sq][rank]
      return pb1[lay.lane_off + sq + 1] - pb1[lay.lane_off + sq];
    };

    // per-run epoch. The chunk / wire flags carry it (the kernels write run_id_, the lanes wait for
    // >= run_id_), so there is no per-step reset to race a pending wait of the previous layer. run_id_
    // only grows over the op's lifetime (resize keeps it), and the flags are zeroed at allocation only.
    this->run_id_ += 1;
    // device-issued wire (dwire/dwire.h): multi-node compress path of a deferred-verdict step
    const bool dw = deferred && NN > 1 && this->a2av_compress_ && this->n_split == 1;
    const int step_slot = step_current_slot();
    uint64_t const *run_dev = step_slot >= 0 ? step_slot_word(step_slot, StepSlot::kCombineRun) : nullptr;
    // start of the step on the caller's stream: an event for the receivers and the pre-reduce, an epoch word
    // for the lanes (written before their waits are enqueued, so the writer is enqueued first)
    if (!dw) CU_CHECK(CUStreamWriteValue64(
        (CUstream)stream_raw, reinterpret_cast<CUdeviceptr>(this->staging_word_), this->run_id_,
        CU_STREAM_WRITE_VALUE_DEFAULT));
    CUDA_CHECK(cudaEventRecord(this->staging_reset_event, stream_raw));
    cudaStream_t intra_stream = this->a2av_intra_stream_.value();
    cudaStream_t reduce_stream = this->a2av_reduce_stream_.value();
    if (!dw) {
      this->wait_word(intra_stream, this->staging_word_, this->run_id_);
    }
    // the receiver's plan-time kernels (bucket map / scan / scatter, reduce) read the reduce index and
    // CSR produced by the combine-metadata derive on the caller's stream (or on a side stream the caller
    // joined into it); without this edge they could read a fresh allocation before it is written
    CUDA_CHECK(cudaStreamWaitEvent(reduce_stream, this->staging_reset_event));
    if (NN > 1 && !dw) {
      this->wait_word(this->internode_stream, this->staging_word_, this->run_id_);
      for (auto &sq : this->internode_streams2_) {
        this->wait_word(sq, this->staging_word_, this->run_id_);
      }
    }
    if (this->a2av_compress_) {
      if (!dw) {
        this->wait_word(this->a2av_conv_stream_.value(), this->staging_word_, this->run_id_);
      }
      CUDA_CHECK(
          cudaStreamWaitEvent(this->a2av_prered_stream_.value(), this->staging_reset_event));
    }

    // pack kernel FIRST -- every host wait below is enqueued after it, so the
    // shared front-end channel always has the flag producer ahead of its consumers
    // fused pack: the GEMM already scattered the panel with the
    // coefficients pre-folded — the pack is a flag relay, scale must be null
    const bool fused_relay = this->a2av_fused_pack_ && this->msplit_n_waves_ > 0;
    CombinePackArguments pack_args{
        .gemm_out = gemm_out.data_ptr(),
        // deferred: the relay-or-pack choice is the device's (msplit_dec), so the row scale is always passed
        .vec_scale = ((deferred || !fused_relay) && output_vec_scales.has_value())
                         ? (float const *)output_vec_scales->at(0).data_ptr()
                         : nullptr,
        .pack_index = pack_index.data_ptr<int32_t>(),
        .send_panel = this->a2av_send_panel_.data_ptr(),
        .barrier = this->barrier.data_ptr<int>(),
        .group_flags = this->group_flags.get(),
        .group_counters = this->group_counters.get(),
        .run_id = this->run_id_,
        .node_row_start = {},
        .panel_rows = this->a2av_send_rows_,
        .n = this->n_dim,
        .n_per = (int)n_per,
        .n_split = this->n_split,
        .nnodes = NN,
        .node_idx = my_node,
        .threadblock_count = num_thread_blocks};
    if (deferred) {
      // node n's rows start at send_off[n * L] of the device plan block; [W] = rows
      pack_args.node_row_dev = dplan + lay.send_off;
      pack_args.node_row_stride = L;
      pack_args.msplit_dec = deferred ? ddec : nullptr;
    } else {
      for (int n = 0; n < NN; n++) {
        pack_args.node_row_start[n] = send_off_at(n * L);
      }
      pack_args.node_row_start[NN] = M_this_ep;
    }
    if (dw) {
      int64_t *tab = this->dwire_tables_.data_ptr<int64_t>();
      pack_args.plan = dplan;
      pack_args.plan_send_off = lay.send_off;
      pack_args.plan_send_rows = lay.send_rows;
      pack_args.plan_dst_off = lay.dst_off;
      pack_args.plan_conv_dst = lay.conv_dst;
      pack_args.recv_peer = reinterpret_cast<char *const *>(tab + 0 * L);
      pack_args.conv_peer = reinterpret_cast<char *const *>(tab + 1 * L);
      pack_args.recvsig_peer = reinterpret_cast<uint64_t *const *>(tab + 2 * L);
      pack_args.convsig_peer = reinterpret_cast<uint64_t *const *>(tab + 3 * L);
      pack_args.recv_cap = this->a2av_recv_rows_;
      pack_args.conv_cap = this->a2av_conv_rows_;
      pack_args.local_world_size = L;
      pack_args.my_lr = my_lr;
      pack_args.rank = this->rank;
      pack_args.push_counters = reinterpret_cast<unsigned *>(this->dwire_push_counters_.data_ptr<int32_t>());
      pack_args.run_ptr = run_dev;
      pack_args.kill = kill_word_dev();
    }
    // M-split waves: gate the pack per schedule step on the wave's cascade flag
    pack_args.relay_only = fused_relay ? 1 : 0;
    pack_args.n_chunk_flags = this->msplit_chunk_flags_;
    pack_args.n_pieces = 0;
    pack_args.msplit = this->msplit_n_waves_ > 0 ? 1 : 0;
    if (pack_args.msplit) {
      FLUX_CHECK_EQ(this->n_split, 1);
      for (int gi = 0; gi < NN; gi++) {
        pack_args.wave_of_node[gi] = this->msplit_wave_of_node_[gi];
        pack_args.node_order[gi] = this->msplit_node_order_[gi];
      }
    }
    // remote-node production schedule for the conv/prered/wire consumers:
    // the msplit order when armed (own node filtered out — it may sit first
    // or last in the schedule), ring otherwise — every consumer follows this
    // ONE array
    int sched_remote[kA2AVMaxNodes];
    if (pack_args.msplit) {
      int w = 0;
      for (int gi = 0; gi < NN; gi++) {
        if (this->msplit_node_order_[gi] != my_node) {
          sched_remote[w++] = this->msplit_node_order_[gi];
        }
      }
      FLUX_CHECK_EQ(w, NN - 1);
      if (deferred && ddec != nullptr) {
        // the device may collapse the waves (ring order in the pack): the production order the wire and the
        // pre-reduce follow must be the same either way (one-node waves, own node last)
        for (int gi = 0; gi < NN - 1; gi++) {
          FLUX_CHECK_EQ(sched_remote[gi], (my_node + 1 + gi) % NN)
              << "deferred verdict: the destination-wave order must equal the ring order";
        }
      }
    } else {
      for (int gi = 0; gi < NN - 1; gi++) {
        sched_remote[gi] = (my_node + 1 + gi) % NN;
      }
    }
    auto flux_dtype = from_torch_dtype(dtype);
    a2av_combine_pack(pack_args, flux_dtype, stream_raw);

    char *send_base = (char *)this->a2av_send_panel_.data_ptr();
    char *recv_base = (char *)this->a2av_recv_panel_.data_ptr();
    uint64_t *recv_sig = (uint64_t *)this->a2av_recv_signals_.data_ptr();
    auto send_ptr = [&](int sid, int64_t row) -> char * {
      return send_base + ((int64_t)sid * this->a2av_send_rows_ + row) * row_bytes;
    };
    auto recv_ptr = [&](int sid, int64_t row) -> char * {
      return recv_base + ((int64_t)sid * this->a2av_recv_rows_ + row) * row_bytes;
    };

    // compress: launch the persistent pre-reduce kernel right after the pack
    // kernel, before any host wait reaches the conn=1 channel (a blocked wait
    // ahead of a kernel launch could park it forever). It spins on the conv
    // signals per (tn, sid) and flips the wire flags the inter ladder gates on.
    char *conv_base = nullptr, *wire_base = nullptr;
    uint64_t *conv_sig = nullptr;
    if (this->a2av_compress_) {
      conv_base = (char *)this->a2av_conv_panel_.data_ptr();
      wire_base = (char *)this->a2av_wire_panel_.data_ptr();
      conv_sig = (uint64_t *)this->a2av_conv_signals_.data_ptr();
      CombinePreReduceArguments prered_args{
          .conv_panel = conv_base,
          .wire_panel = wire_base,
          .wire_ptr = wire_csr->at(0).data_ptr<int32_t>(),
          .wire_copy = wire_csr->at(1).data_ptr<int32_t>(),
          .conv_signals = conv_sig,
          .run_id = this->run_id_,
          .wire_flags = this->wire_flags_.get(),
          .wire_counters = this->wire_counters_.get(),
          .wire_seg_start = {},
          .conv_rows = this->a2av_conv_rows_,
          .wire_rows = this->a2av_wire_rows_,
          .n_per = (int)n_per,
          .n_split = this->n_split,
          .nnodes = NN,
          .node_idx = my_node,
          .local_world_size = L,
          .threadblock_count = get_a2av_prered_blocks(),
          .spin_limit = 0,
          .kill_word = kill_word_dev()};
      if (deferred) {
        // segment starts (tn asc skip own), [NN - 1] = total, from the device plan block
        prered_args.wire_seg_dev = dplan + lay.wire_seg;
      } else {
        for (int tn = 0, seg = 0; tn < NN; tn++) {
          if (tn == my_node) {
            continue;
          }
          prered_args.wire_seg_start[seg] = wire_seg_at(tn);
          seg++;
        }
        prered_args.wire_seg_start[NN - 1] = pb1[lay.wire_seg + NN - 1];  // segment-array end
      }
      // production schedule (ring by default, msplit order when armed) — must
      // ALWAYS be filled: the kernel visits args.node_order unconditionally
      for (int gi = 0; gi < NN - 1; gi++) {
        prered_args.node_order[gi] = sched_remote[gi];
      }
      prered_args.run_ptr = run_dev;
      a2av_combine_prereduce(prered_args, flux_dtype, this->a2av_prered_stream_.value());
      if (dw) {
        // the inter-node wire: one warp issues the device puts in the pre-reduce's order (dwire/dwire.h)
        DwireCombineArgs ca{};
        ca.plan = dplan;
        ca.lay_wire_rows = lay.wire_rows;
        ca.lay_wire_seg = lay.wire_seg;
        ca.lay_dst_off = lay.dst_off;
        ca.run = run_dev;
        ca.run_host = this->run_id_;
        ca.L = L;
        ca.NN = NN;
        ca.my_node = my_node;
        ca.my_lr = my_lr;
        ca.rank = this->rank;
        ca.n_split = this->n_split;
        for (int gi = 0; gi < NN - 1; gi++) {
          ca.node_order[gi] = sched_remote[gi];
        }
        ca.wire_base = wire_base;
        ca.recv_base = (char *)this->a2av_recv_panel_.data_ptr();
        ca.row_bytes = row_bytes;
        ca.wire_cap = this->a2av_wire_rows_;
        ca.recv_cap = this->a2av_recv_rows_;
        ca.wire_flags = this->wire_flags_.get();
        ca.recv_sig = (uint64_t *)this->a2av_recv_signals_.data_ptr();
        ca.kill = kill_word_dev();
        CUDA_CHECK(cudaStreamWaitEvent(this->internode_stream, this->staging_reset_event));
        dwire_combine_wire(ca, this->internode_stream);
        CUDA_CHECK(cudaEventRecord(this->a2av_inter_done_, this->internode_stream));
      }
    }
    auto conv_ptr = [&](int sid, int64_t row) -> char * {
      return conv_base + ((int64_t)sid * this->a2av_conv_rows_ + row) * row_bytes;
    };
    auto wire_ptr_at = [&](int sid, int64_t row) -> char * {
      return wire_base + ((int64_t)sid * this->a2av_wire_rows_ + row) * row_bytes;
    };
    // The ladders are enqueued INTERLEAVED PER SPLIT, in dependency order
    // (inter -> intra -> gateway -> reduce): under CUDA_DEVICE_MAX_CONNECTIONS=1
    // all streams multiplex one front-end channel and a pending wait can block
    // later-enqueued ops, so the enqueue order must itself be an executable
    // pipelined schedule. Within a split the pack kernel flips remote-node flags
    // first (production order) and the own-node flag last, so the inter waits
    // sit ahead of the intra wait; the reduce waits depend on this rank's own
    // gateway forwards, which are enqueued just before them.
    for (int sid = 0; sid < this->n_split; sid++) {
      if (this->a2av_compress_ && !dw) {
        // conv ladder: behind the pack chunk flag per target node (produced
        // remote-first). My sub-chunk for dest (tn, dl) converges on local
        // gateway (my_node, dl): self sub-chunk is a CE memcpy into my own
        // conv panel, peers get one contiguous putmem_signal each (NVLink CE).
        // Every (peer, tn) pair signals every split, payload or not.
        cudaStream_t conv_stream = this->a2av_conv_stream_.value();
        for (int gi = 0; gi < NN - 1; gi++) {
          int tn = sched_remote[gi];  // production schedule (ring / size-sorted)
          this->wait_word(conv_stream, this->group_flags.get() + tn * this->n_split + sid, this->run_id_);
          // the L convergence copies of (tn, sid): one copy-engine batch, then their SET signals
          // as stream writes (every gateway is on my node)
          CeBatch cbatch;
          CeSignals csigs;
          for (int di = 0; di < L; di++) {
            int dl = (my_lr + di) % L;  // self first, then rotation (no incast)
            int d = tn * L + dl;
            int gw = my_node * L + dl;
            int64_t rows = rows_to(d);
            int64_t coff = conv_dst_at(dl, tn);
            uint64_t *slot =
                conv_sig + ((int64_t)my_lr * NN + tn) * this->n_split + sid;
            void *pd = this->peers_.peer(conv_ptr(sid, coff), gw);
            if (pd != nullptr) {
              cbatch.add(pd, send_ptr(sid, send_off_at(d)), rows * row_bytes);  // zero rows: signal only
              csigs.add(slot, this->peers_.peer(slot, gw), gw, this->run_id_);
              continue;
            }
            // no P2P mapping for gw: the NVSHMEM calls, in program order
            char *cdst = conv_ptr(sid, coff);
            const char *csrc = send_ptr(sid, send_off_at(d));
            const size_t cbytes = rows * row_bytes;
            if (gw == this->rank) {
              if (cbytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    cdst, csrc, cbytes, cudaMemcpyDeviceToDevice, conv_stream));
              }
              nvshmemx_signal_op_on_stream(slot, this->run_id_, NVSHMEM_SIGNAL_SET, gw, conv_stream);
            } else if (cbytes > 0) {
              flux_rs_put_signal(
                  cdst, csrc, cbytes, slot, this->run_id_, NVSHMEM_SIGNAL_SET, gw, conv_stream,
                  this->local_world_size, this->node_idx);
            } else {
              nvshmemx_signal_op_on_stream(slot, this->run_id_, NVSHMEM_SIGNAL_SET, gw, conv_stream);
            }
          }
          cbatch.issue(conv_stream);
          csigs.issue(conv_stream);
        }
        // wire ladder: behind the pre-reduce kernel's (tn, sid) wire flag, one
        // direct putmem_signal per remote node into the same-lr destination's
        // recv panel (C' image) -- no destination gateway hop. The (rank, sid)
        // slot at the destination keeps exactly one writer: me.
        for (int gi = 0; gi < NN - 1; gi++) {
          int tn = sched_remote[gi];  // production schedule (ring / size-sorted)
          int d = tn * L + my_lr;
          // S <= NN-1: per-split round-robin (gi % S).
          // S > NN-1: the split loop only has NN-1 puts, so extra lanes are
          // dead under gi % S; spread by GLOBAL cell index (sid*(NN-1)+gi)
          // instead, so puts from different splits round-robin over all S
          // lanes.
          int wire_lane = 0;
          if (this->rs_wire_streams_ > 1) {
            wire_lane =
                (this->rs_wire_streams_ <= NN - 1)
                    ? (gi % this->rs_wire_streams_)
                    : (int)(((int64_t)sid * (NN - 1) + gi) % this->rs_wire_streams_);
          }
          cudaStream_t wstream =
              (wire_lane > 0) ? (cudaStream_t)this->internode_streams2_[wire_lane - 1]
                              : (cudaStream_t)this->internode_stream;
          this->wait_word(wstream, this->wire_flags_.get() + tn * this->n_split + sid, this->run_id_);
          int64_t rows = wire_rows_at(tn);
          {
            // BLOCKING inter-node put: a non-blocking put-with-signal can expose the signal before the data
            // on CXI
            char *wdst = recv_ptr(sid, dst_off_at(d));
            const char *wsrc = wire_ptr_at(sid, wire_seg_at(tn));
            const size_t wbytes = rows * row_bytes;
            uint64_t *slot = recv_sig + this->rank * this->n_split + sid;
            if (wbytes > 0) {
              flux_rs_put_signal(
                  wdst, wsrc, wbytes, slot, this->run_id_, NVSHMEM_SIGNAL_SET, d, wstream,
                  this->local_world_size, this->node_idx);
            } else {
              nvshmemx_signal_op_on_stream(slot, this->run_id_, NVSHMEM_SIGNAL_SET, d, wstream);
            }
          }
        }
      }
      if (!dw) {  // device wire: the pack kernel pushed the intra rows
      // intra-node: behind the own-node chunk flag; self chunk is a local CE
      // copy, peers get one contiguous putmem_signal each (CE over NVLink for
      // same-node PEs). Every pair signals every split, payload or not.
      this->wait_word(intra_stream, this->group_flags.get() + my_node * this->n_split + sid, this->run_id_);
      // intra-node ladder of the split: the self copy and the L-1 peer copies as one copy-engine
      // batch, then the L SET signals as stream writes
      {
        CeBatch ibatch;
        CeSignals isigs;
        uint64_t *islot = recv_sig + this->rank * this->n_split + sid;
        for (int dl = 0; dl < L; dl++) {
          int d = dl == 0 ? this->rank
                          : dist_env.local_rank_to_global_rank((my_lr - dl + L) % L, my_node);
          int64_t rows = rows_to(d);
          char *dst = recv_ptr(sid, dst_off_at(d));
          void *pd = this->peers_.peer(dst, d);
          if (pd != nullptr) {
            ibatch.add(pd, send_ptr(sid, send_off_at(d)), rows * row_bytes);  // zero rows: signal only
            isigs.add(islot, this->peers_.peer(islot, d), d, this->run_id_);
            continue;
          }
          // no P2P mapping for d: the NVSHMEM calls, in program order
          const char *isrc = send_ptr(sid, send_off_at(d));
          const size_t ibytes = rows * row_bytes;
          if (d == this->rank) {
            if (ibytes > 0) {
              CUDA_CHECK(cudaMemcpyAsync(dst, isrc, ibytes, cudaMemcpyDeviceToDevice, intra_stream));
            }
            nvshmemx_signal_op_on_stream(islot, this->run_id_, NVSHMEM_SIGNAL_SET, d, intra_stream);
          } else if (ibytes > 0) {
            flux_rs_put_signal(
                dst, isrc, ibytes, islot, this->run_id_, NVSHMEM_SIGNAL_SET, d, intra_stream,
                this->local_world_size, this->node_idx);
          } else {
            nvshmemx_signal_op_on_stream(islot, this->run_id_, NVSHMEM_SIGNAL_SET, d, intra_stream);
          }
        }
        ibatch.issue(intra_stream);
        isigs.issue(intra_stream);
      }
      // the receivers below enqueue FRONT-END waits on the recv signals, whose writers include this rank's
      // own lanes (the self slot) and, through the peers' waits on theirs, every lane of the split: a wait
      // enqueued ahead of a lane in a shared hardware queue would block it, so this split's lanes are
      // enqueued first
      }  // !dw
      if (!this->a2av_compress_) {
        // single node: wait for every source's recv signal of the split, then one
        // memory-bound kernel folds the top-k rows of every local token
        for (int s = 0; s < W; s++) {
          CU_CHECK(CUStreamWaitValue64(
              reduce_stream,
              (CUdeviceptr)(recv_sig + s * this->n_split + sid),
              this->run_id_,
              CU_STREAM_WAIT_VALUE_GEQ));
        }
        CombineReduceArguments reduce_args{
            .recv_panel = this->a2av_recv_panel_.data_ptr(),
            .reduce_index = reduce_index.data_ptr<int32_t>(),
            .output = output.data_ptr(),
            .panel_rows = this->a2av_recv_rows_,
            .ntokens_local = cpr / this->topk,
            .n = this->n_dim,
            .n_per = (int)n_per,
            .topk = this->topk,
            .sid = sid,
            .threadblock_count = get_a2av_reduce_blocks(),
            .verdict = dverdict};
        a2av_combine_reduce(reduce_args, flux_dtype, reduce_stream);
      } else {
        // COMPLETION-BUCKETED receiver: a ~us plan-time
        // bucket sort of the reduce CSR by completion chain position (on the
        // reduce stream, inside the timed bracket), then per-lane front-end
        // waits each releasing a register-CSR fold of exactly the tokens
        // that lane completes. Own-node lanes chain FIRST -- consumption
        // order only, the wire keeps the own-last production order: own rows
        // are ready at GEMM end, which precedes the remote drain, so tokens
        // complete at their last REMOTE arrival and the fold spreads over
        // the window at wait-all's 1x byte budget. Sequential waits on one
        // stream give the bucket-prefix guarantee; skipping zero-row lanes
        // is safe (no CSR entry, so no token can complete there).
        FLUX_CHECK_EQ(this->n_split, 1);
        const int64_t ntok_local = cpr / this->topk;
        int chain_lane_of[kA2AVMaxWorld];
        int S = 0;
        chain_lane_of[S++] = this->rank;
        for (int dl = 1; dl < L; dl++) {
          chain_lane_of[S++] = my_node * L + (my_lr - dl + L) % L;
        }
        for (int gi = 1; gi < NN; gi++) {
          const int n2 = (my_node - gi + NN) % NN;
          chain_lane_of[S++] = n2 * L + my_lr;
        }
        FLUX_CHECK_LE(S, kA2AVMaxWorld);
        // per-iteration lane tables: the device plan's twin (written by the derive, which the reduce stream
        // is ordered after)
        int32_t const *lanes_d = this->plan_lanes_dev_;
        int32_t *bucket_meta = this->a2av_bucket_meta_.data_ptr<int32_t>();
        int32_t *bucket_cnt = bucket_meta;
        int32_t *bucket_cur = bucket_meta + kA2AVMaxWorld;
        int32_t *bucket_ptr_d = bucket_meta + 2 * kA2AVMaxWorld;
        CUDA_CHECK(
            cudaMemsetAsync(bucket_cnt, 0, sizeof(int32_t) * (size_t)S, reduce_stream));
        a2av_bucket_map(
            A2AVBucketMapArguments{
                reduce_csr->at(0).data_ptr<int32_t>(),
                reduce_csr->at(1).data_ptr<int32_t>(),
                lanes_d,
                lanes_d + (W + 1),
                W,
                S,
                ntok_local,
                this->a2av_bucket_comp_.data_ptr<int32_t>(),
                bucket_cnt},
            reduce_stream);
        a2av_bucket_scan(
            A2AVBucketScanArguments{bucket_cnt, S, bucket_ptr_d, bucket_cur},
            reduce_stream);
        a2av_bucket_scatter(
            A2AVBucketScatterArguments{
                this->a2av_bucket_comp_.data_ptr<int32_t>(),
                bucket_ptr_d,
                ntok_local,
                bucket_cur,
                this->a2av_bucket_tok_.data_ptr<int32_t>()},
            reduce_stream);
        bool tail_joined = false;
        for (int k = 0; k < S; k++) {
          const int sq = chain_lane_of[k];
          if (!deferred && !no_host && lane_rows(sq) <= 0) {
            continue;  // zero-row lane: still signals, but its bucket is empty (deferred / no host copy: waited)
          }
          // remote lanes (chain positions >= L) wide, behind GEMM 2's end
          const bool wide = k >= L;
          if (wide && !tail_joined) {
            CUDA_CHECK(cudaStreamWaitEvent(reduce_stream, this->tail_gemm_end_));
            tail_joined = true;
          }
          if (!dw) {
            CU_CHECK(CUStreamWaitValue64(
                reduce_stream,
                (CUdeviceptr)(recv_sig + sq * this->n_split + sid),
                this->run_id_,
                CU_STREAM_WAIT_VALUE_GEQ));
          }
          A2AVBucketReduceArguments bargs{
                  this->a2av_recv_panel_.data_ptr(),
                  reduce_csr->at(0).data_ptr<int32_t>(),
                  reduce_csr->at(1).data_ptr<int32_t>(),
                  bucket_ptr_d,
                  this->a2av_bucket_tok_.data_ptr<int32_t>(),
                  k,
                  output.data_ptr(),
                  this->a2av_recv_rows_,
                  (int)this->n_dim,
                  (int)n_per,
                  sid,
                  wide ? get_a2av_tail_reduce_blocks() : get_a2av_reduce_blocks()};
          if (dw) {
            // the lane's arrival is waited inside the kernel (no front-end wait in the layer)
            bargs.wait_sig = recv_sig + sq * this->n_split + sid;
            bargs.run_ptr = run_dev;
            bargs.run_host = this->run_id_;
            bargs.kill = kill_word_dev();
            if (wide) {
              // a one-thread wait ahead of the fold: the wide grid is resident only once the lane has landed (a
              // wide grid spinning from GEMM 2's end would hold the SMs every rank's returns still need)
              a2av_lane_wait(bargs.wait_sig, bargs.run_ptr, bargs.run_host, bargs.kill, reduce_stream);
              bargs.wait_sig = nullptr;
            }
          }
          a2av_combine_bucket_reduce(bargs, flux_dtype, reduce_stream);
        }
      }
    }

    // tail joins: everything the epoch produced must reach the gather-rs stream
    // before the caller's gather_rs_done_event / closing barrier, covering
    // panel + staging reuse in the next iteration
    {
      // the lanes (intra, inter-node, conv) join into the intra stream, which then writes the done word;
      // the caller's stream waits on the word
      if (dw) {
        // device wire: the wire warp's stream joins here (the pre-reduce and the receivers below)
        CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_inter_done_));
      } else {
      const uint64_t run = this->run_id_;
      if (NN > 1) {
        CUDA_CHECK(cudaEventRecord(this->a2av_inter_done_, this->internode_stream));
        CUDA_CHECK(cudaStreamWaitEvent(intra_stream, this->a2av_inter_done_));
        for (auto &sq : this->internode_streams2_) {
          CUDA_CHECK(cudaEventRecord(this->a2av_inter2_done_, sq));
          CUDA_CHECK(cudaStreamWaitEvent(intra_stream, this->a2av_inter2_done_));
        }
      }
      if (this->a2av_compress_) {
        CUDA_CHECK(cudaEventRecord(this->a2av_conv_done_, this->a2av_conv_stream_.value()));
        CUDA_CHECK(cudaStreamWaitEvent(intra_stream, this->a2av_conv_done_));
      }
      CU_CHECK(CUStreamWriteValue64(
          (CUstream)intra_stream, reinterpret_cast<CUdeviceptr>(this->wire_done_), run,
          CU_STREAM_WRITE_VALUE_DEFAULT));
      // the writer is already enqueued, a front-end wait is safe
      CU_CHECK(CUStreamWaitValue64(
          (CUstream)stream_raw, reinterpret_cast<CUdeviceptr>(this->wire_done_), run,
          CU_STREAM_WAIT_VALUE_GEQ));
      }  // !dw
    }
    if (this->a2av_compress_) {
      CUDA_CHECK(cudaEventRecord(this->a2av_prered_done_, this->a2av_prered_stream_.value()));
      CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_prered_done_));
    }
    CUDA_CHECK(cudaEventRecord(this->a2av_reduce_done_, reduce_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream_raw, this->a2av_reduce_done_));
    return output;
  }

  torch::Tensor
  run(std::vector<torch::Tensor> gemm_outs,  // of group_size
      c10::optional<torch::Tensor> output_,
      int ep_start,
      int ep_nexperts,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> output_vec_scales,
      int num_thread_blocks,
      intptr_t cp_stream,
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> pack_index = c10::nullopt,
      c10::optional<torch::Tensor> reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> reduce_csr = c10::nullopt) {
    at::cuda::CUDAStream stream =
        at::cuda::getStreamFromExternal((cudaStream_t)cp_stream, at::cuda::current_device());
    at::cuda::CUDAStreamGuard _(stream);
    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    CHECK_INPUT(splits, at::ScalarType::Int);
    int N = this->n_dim;
    int m_full = routing_idx.size(0);
    int ntokens = m_full / this->topk;
    int ntokens_per_rank = ntokens / this->world_size;
    int ntokens_out = ntokens_per_rank;
    FLUX_CHECK_GE(gemm_outs.size(), 1);
    FLUX_CHECK_LE(gemm_outs.size(), kMaxNumGroups);
    auto dtype = gemm_outs[0].scalar_type();

    auto output = output_.value_or(empty_with_uninitialized_data(
        std::vector<int64_t>{ntokens_out, N}, gemm_outs[0].options()));
    CHECK_TYPE(output, dtype);
    CHECK_2D(output, ntokens_out, N);

    {
      FLUX_CHECK_EQ((int)gemm_outs.size(), 1) << "the fused combine supports a single weight group";
      FLUX_CHECK(splits_per_source.has_value())
          << "the fused combine requires splits_per_source ([W, nexperts] int32 CPU)";
      FLUX_CHECK(!this->a2av_compress_ || wire_csr.has_value())
          << "compress mode needs the combine metadata from derive_combine_meta";
      FLUX_CHECK(pack_index.has_value() && reduce_index.has_value())
          << "the combine needs pack_index and reduce_index (GemmCombineOp builds them)";
      return combine(
          gemm_outs[0],
          output,
          splits_per_source.value(),
          pack_index.value(),
          reduce_index.value(),
          unique_counts,
          wire_csr,
          reduce_csr,
          output_vec_scales,
          m_full,
          num_thread_blocks,
          (cudaStream_t)cp_stream);
    }

    return output;
  }

  // Grow the capacity-sized symmetric panels in place (collective; idle device on every rank).
  // Signals and the epoch scheme are untouched.
  void
  resize_capacities(CombineOptions const &options) {
    FLUX_CHECK(this->buffer_initialized);
    const int64_t n_per = this->n_dim / this->n_split;
    const at::ScalarType dtype = this->a2av_send_panel_.scalar_type();
    if (options.max_send_rows > this->a2av_send_rows_) {
      this->a2av_send_panel_ = torch::Tensor();
      this->a2av_send_rows_ = options.max_send_rows;
      this->a2av_send_panel_ =
          nvshmem_create_tensor({this->n_split, this->a2av_send_rows_, n_per}, dtype);
    }
    if (this->a2av_compress_) {
      if (options.max_conv_rows > this->a2av_conv_rows_) {
        this->a2av_conv_panel_ = torch::Tensor();
        this->a2av_conv_rows_ = options.max_conv_rows;
        this->a2av_conv_panel_ =
            nvshmem_create_tensor({this->n_split, this->a2av_conv_rows_, n_per}, dtype);
      }
      if (options.max_wire_rows > this->a2av_wire_rows_) {
        this->a2av_wire_panel_ = torch::Tensor();
        this->a2av_wire_rows_ = options.max_wire_rows;
        this->a2av_wire_panel_ =
            nvshmem_create_tensor({this->n_split, this->a2av_wire_rows_, n_per}, dtype);
      }
    }
    this->options_ = options;
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    this->build_peer_table();  // the conv panel may have moved
  }

  void
  reset_buffer() {
    if (this->tile_barrier.defined()) {
      this->tile_barrier.zero_();
    }
  }
};

/// This class only runs the basic grouped_gemm, it is mainly used for testing
class GemmCombineOp::GemmCombineOpImpl {
 private:
  std::shared_ptr<Group> tp_group;
  int32_t ep_nexperts;
  int32_t ep_start;
  const int32_t total_num_experts;
  int32_t max_m;
  int32_t n_dim;
  int32_t topk;
  at::ScalarType output_dtype;
  int32_t max_input_groups;
  int32_t rank;
  int32_t world_size;     // the total world size
  int32_t tp_world_size;  // the world size of tensor parallel
  int32_t ep_world_size;  // the world size of expert parallel
  int32_t nnodes;
  int32_t local_rank;        // == rank when nnodes == 1
  int32_t local_world_size;  // == world_size when nnodes == 1
  int n_split;
  bool a2av_compress;  // a2av_hier_compress ctor flag; false when nnodes == 1
  torch::Tensor barrier;
  std::vector<torch::Tensor> barriers;  // [local_world_size], indexed by local rank
  std::unique_ptr<CombineWire> combine_wire = nullptr;
  // M-split waves: per-iteration wave
  // tables (pinned host arena -> device, one async H2D per forward)
  bool msplit_ = false;
  bool fused_pack_ = false;    // epilogue-fused pack: GEMM writes the send panel
  int msplit_wave_nodes_ = 1;
  torch::Tensor msplit_dev_;   // device int32 [4 * NN * E + NN] (chunked combine tables)
  torch::Tensor msplit_iota_;      // int32 [max_m] shared identity indices
  torch::Tensor msplit_inv_pack_;  // int32 [max_m] gemm row -> panel row
  // swap: combine-side weight gate, armed per forward
  // by set_weight_gate() (one-shot). gate_of_expert[e] = index into the
  // signal array (-1 ungated); the forward orders gated experts LAST inside
  // every combine wave and ships a per-problem gate map to the GEMM.
  bool wgate_armed_ = false;
  torch::Tensor wgate_override_;  // swap "gate on staging": per-expert B base override (one-shot)
  torch::Tensor devmeta_i64_, devmeta_i32_;  // device-derived combine tables
  CombineDeriveBuffers derive_bufs_;  // persistent derive outputs
  // combine plan block: two parities by derive sequence number, device and pinned host; each holds
  // plan_lay_.total int64 words followed by the int32 bucket lane table [2 W + 1]
  CombinePlanLayout plan_lay_{};
  int64_t plan_bytes_ = 0;
  torch::Tensor plan_dev_, plan_host_;
  cudaEvent_t plan_event_ = nullptr;  // the host copy of the latest block has landed
  uint64_t plan_seq_ = 0;
  bool plan_armed_ = false;           // a derive ran whose block the next forward has not read yet
  // side stream of the overlapped combine derive (owned by the op, so no other component's work
  // can be queued on it)
  cudaStream_t meta_stream_ = nullptr;
  // the msplit tables and the pack inverse on a side stream forked from the combine metadata's completion event
  // (set_prep_fork, one-shot per forward): a layer graph runs them while the dispatch GEMM computes instead of after
  // the activation; the combine stream joins before the scale fold. Without a fork they run in line.
  cudaStream_t prep_stream_ = nullptr;
  cudaEvent_t prep_fork_ev_ = nullptr;
  cudaEvent_t prep_join_ev_ = nullptr;
  bool prep_fork_armed_ = false;
  torch::Tensor wgate_signal_;
  uint64_t wgate_epoch_ = 0;
  bool wgate_epoch_device_ = false;
  std::vector<int32_t> wgate_of_expert_;
  // deferred verdict: the per-expert gate map on the device (the swap lane's arm kernel writes it; the host
  // does not know which slots arrive), replacing wgate_of_expert_ in the device tables
  torch::Tensor wgate_map_dev_;
  // deferred verdict: the device wave-adapt decision of the step (int32 [1], a2av_msplit_tables writes it)
  torch::Tensor msplit_dec_;
  torch::Tensor wgate_dev_;   // device int32 [cap]

  // swap: one-shot GEMM-START MARK. set_gemm_start_mark(e)
  // arms the next forward to write e into a device int64 flag on the forward
  // stream right before the GEMM kernel launches; the swap lane's movement
  // streams wait on it (cuStreamWaitValue64 GEQ) so the NVLink expert copies
  // start with the GEMM, not with the op's metadata/staging prologue.
  torch::Tensor gemm_mark_;  // device int64[1]
  int64_t gemm_mark_epoch_ = 0;
  bool gemm_mark_armed_ = false;
  uint64_t *gemm_mark_raw_ = nullptr;  // the mark's memory (owned by gemm_mark_)
  std::vector<int> msplit_wave_of_node_;  // schedule position -> cascade flag
  std::vector<int> msplit_node_order_;    // schedule position -> dest node

  torch::Tensor workspace;
  cudaEvent_t gemm_start_event;
  cudaEvent_t gemm_end_event_ = nullptr;  // the combine GEMM's end (the remote-lane receivers wait for it)
  cudaEvent_t gather_rs_done_event;
  cudaStream_t gather_rs_stream;

  int
  get_barrier_size(int problem_count) const {
    return pad_to(this->n_split, 128) * 2  // 1st: ready flag per tile, 2nd: counter per split
           + pad_to(problem_count, 128);   // counter for each problem gemm done tiles
  }

  void
  create_barriers() {
    // M-split waves need tile counters for up to nnodes * ep_nexperts problems.
    // The wave-flag/problem-counter regions reuse the 128-padded per-split
    // slots (n_waves <= nnodes <= 128, checked in the ctor), so the per-split
    // layout is unchanged for everything it addresses; this is a superset
    // sizing.
    const int problem_count = std::max<int>(
        this->n_split * this->ep_nexperts * this->max_input_groups,
        this->msplit_ ? this->nnodes * this->ep_nexperts * this->max_input_groups : 0);
    const int barrier_size = get_barrier_size(problem_count);
    if (this->barriers.empty()) {
      this->barriers = flux_create_tensor_list(
          std::vector<int64_t>{barrier_size}, at::ScalarType::Int, this->tp_group.get(), true);
      FLUX_CHECK_EQ((int)this->barriers.size(), this->local_world_size);
      this->barrier = this->barriers[this->local_rank];
    }
  }

 public:
  //
  // index build as ONE host call: the device tables from the device counts (a2av_combine_tables), the
  // combine plan block on the device with its copy to pinned host memory (the forward waits for that
  // copy before its first count-derived read), then the arithmetic-identity index builders; the chain
  // launches back-to-back with no interpreter gaps and no host loop over the counts. Returns
  // {pack_index, reduce_index} for a2av_hier, plus {wire_ptr, wire_copy, red_ptr, red_row} when
  // compress: the whole persistent buffers (the true counts live on the device).
  std::vector<torch::Tensor>
  derive_combine_meta(
      torch::Tensor splits_gpu,
      torch::Tensor routing_idx,
      torch::Tensor splits_per_source,
      c10::optional<torch::Tensor> unique_counts,
      c10::optional<torch::Tensor> sps_dev,
      c10::optional<torch::Tensor> uc_dev,
      c10::optional<torch::Tensor> routing_ids) {
    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    CHECK_INPUT(splits_gpu, at::ScalarType::Int);
    CHECK_1D(splits_gpu, this->total_num_experts);
    FLUX_CHECK(splits_per_source.device().is_cpu()) << "splits_per_source must be CPU";
    CHECK_2D(splits_per_source, this->world_size, this->total_num_experts);
    FLUX_CHECK(splits_per_source.scalar_type() == at::ScalarType::Int);
    FLUX_CHECK(splits_per_source.is_contiguous());
    const int64_t m_full = routing_idx.numel();
    FLUX_CHECK_DIV(m_full, (int64_t)this->world_size * this->topk);
    FLUX_CHECK_LE(m_full, (int64_t)this->max_m) << "derive_combine_meta: more copies than max_m";
    ensure_derive_buffers();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // deferred-verdict step (core/verdict.h; a forward is open): the derive's per-copy kernels read the
    // verdict block (a degenerate step builds empty indices) and the plan block kernel turns the forward's
    // host checks into device asserts
    int64_t *verdict = verdict_armed();
    // derive the tables on device from the counts
    FLUX_CHECK(sps_dev.has_value() && uc_dev.has_value())
        << "derive_combine_meta needs the device sps/uc counts";
    FLUX_CHECK(routing_ids.has_value())
        << "derive_combine_meta needs the routing ids (expert of every copy)";
    CHECK_INPUT(routing_ids.value(), at::ScalarType::Int);
    FLUX_CHECK_EQ(routing_ids->numel(), m_full) << "routing_ids must have one id per copy";
    CHECK_INPUT(sps_dev.value(), at::ScalarType::Int);
    CHECK_INPUT(uc_dev.value(), at::ScalarType::Int);
    CHECK_2D(sps_dev.value(), this->world_size, this->total_num_experts);
    CHECK_2D(uc_dev.value(), this->world_size, this->world_size + this->nnodes);
    CombineDevMeta dm;
    {
      const int W = this->world_size, NN = this->nnodes, L = this->local_world_size;
      const int64_t E = this->ep_nexperts, nex = this->total_num_experts, nexG = nex;
      const int64_t n_i64 = 2 * nex + (int64_t)(NN - 1) * L * E + 2 * W + NN;
      int64_t *base = this->devmeta_i64_.data_ptr<int64_t>();
      int64_t *rt = base + 3 * nexG;
      int64_t *t64 = rt + 4 * nex;
      A2AVCombineTablesArguments targs{
          .sps = sps_dev->data_ptr<int32_t>(),
          .uc = uc_dev->data_ptr<int32_t>(),
          .W = (int32_t)W,
          .ep_nexperts = (int32_t)E,
          .L = (int32_t)L,
          .nnodes = (int32_t)NN,
          .rank = (int32_t)this->rank,
          .cumA = base,
          .offA = base + nexG,
          .offR_of_A = base + 2 * nexG,
          .my_cum = rt,
          .e_base = rt + nex,
          .h_base = rt + 2 * nex,
          .e_cum = rt + 3 * nex,
          .expert_base = t64,
          .my_cnt_cum = t64 + nex,
          .conv_base = t64 + 2 * nex,
          .recv_off_C = t64 + 2 * nex + (int64_t)(NN - 1) * L * E,
          .recv_off_Cp = t64 + 2 * nex + (int64_t)(NN - 1) * L * E + W,
          .rem_base = t64 + 2 * nex + (int64_t)(NN - 1) * L * E + 2 * W,
          .totals = t64 + n_i64,
          .home_base = this->devmeta_i32_.data_ptr<int32_t>()};
      a2av_combine_tables(targs, stream);
      // the plan block, then its host copy (the forward waits for this event, not for the rest of
      // the derive chain below)
      launch_plan_block(sps_dev.value(), uc_dev.value(), stream, verdict, m_full / this->world_size);
      dm.i64 = base;
      dm.i32 = this->devmeta_i32_.data_ptr<int32_t>();
      dm.nexG = nexG;
      dm.nex = nex;
      dm.n_i64 = n_i64;
    }
    CombineDeriveBuffers &bufs = this->derive_bufs_;
    // the expert of every copy: the routing ids
    torch::Tensor e_of = routing_ids.value();
    auto [pack_idx, reduce_idx] = build_a2av_combine_indices(
        routing_idx,
        m_full,
        this->world_size,
        this->rank,
        this->total_num_experts,
        this->ep_nexperts,
        bufs,
        dm,
        routing_ids->data_ptr<int32_t>(),
        verdict);
    std::vector<torch::Tensor> out{pack_idx, reduce_idx};
    if (this->a2av_compress) {
      // sort-free path: scd-arithmetic kernels, no radix sorts
      auto [wp, wc, rp, rr] = build_a2av_compress_indices_fast(
          routing_idx,
          e_of,
          m_full,
          this->world_size,
          this->nnodes,
          this->local_world_size,
          this->rank,
          this->total_num_experts,
          this->ep_nexperts,
          this->topk,
          bufs,
          dm,
          verdict);
      out.push_back(wp);
      out.push_back(wc);
      out.push_back(rp);
      out.push_back(rr);
    }
    return out;
  }

  int64_t
  meta_stream() {
    return reinterpret_cast<int64_t>(this->meta_stream_);
  }
  uint64_t
  run_id() const {
    return this->combine_wire != nullptr ? this->combine_wire->run_id() : 0;
  }
  void
  advance_run_id(int64_t n) {
    FLUX_CHECK(this->combine_wire != nullptr);
    this->combine_wire->advance_run_id(n);
  }

 private:
  // Persistent derive state, allocated once by the constructor (sizes depend only on its arguments).
  void
  ensure_derive_buffers() {
    CombineDeriveBuffers &b = this->derive_bufs_;
    if (b.pack_index.defined()) {
      return;
    }
    const int W = this->world_size, NN = this->nnodes, L = this->local_world_size;
    const int64_t E = this->ep_nexperts, nex = this->total_num_experts, nexG = E * W;
    const int64_t max_cpr = (int64_t)this->max_m / W;
    const int64_t max_tpr = max_cpr / this->topk;
    b.n_i64 = 2 * nex + (int64_t)(NN - 1) * L * E + 2 * W + NN;
    auto dev64 = torch::TensorOptions(torch::kCUDA).dtype(torch::kLong);
    auto dev32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
    b.pack_index = torch::zeros({(int64_t)this->max_m}, dev32);
    b.reduce_index = torch::zeros({max_cpr}, dev32);
    b.scratch = torch::zeros({2 * (int64_t)(NN - 1) * max_tpr + 2 * max_tpr * NN + 1}, dev32);
    b.wire_ptr = torch::zeros({(int64_t)(NN - 1) * max_tpr + 1}, dev32);
    b.wire_copy = torch::zeros({(int64_t)(NN - 1) * max_cpr + 1}, dev32);
    b.red_ptr = torch::zeros({max_tpr + 1}, dev32);
    b.red_row = torch::zeros({max_cpr + 1}, dev32);
    // the tables arena and the plan block (two parities, device and pinned host; each = layout.total
    // int64 words + the int32 bucket lane table)
    const int64_t n_all = 3 * nexG + 4 * nex + b.n_i64 + 8;
    this->devmeta_i64_ = torch::zeros({n_all}, dev64);
    this->devmeta_i32_ = torch::zeros({nex * W}, dev32);
    this->plan_lay_ = combine_plan_layout(W, NN, E);
    this->plan_bytes_ =
        pad_to((int64_t)(this->plan_lay_.total * sizeof(int64_t) + (2 * W + 1) * sizeof(int32_t)), 256);
    this->plan_dev_ = torch::zeros(
        {2 * this->plan_bytes_}, torch::TensorOptions(torch::kCUDA).dtype(torch::kByte));
    this->plan_host_ =
        torch::zeros({2 * this->plan_bytes_}, torch::TensorOptions().dtype(torch::kByte).pinned_memory(true));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->plan_event_, cudaEventDisableTiming));
  }

  char *
  plan_dev_bytes(uint64_t seq) {
    return (char *)this->plan_dev_.data_ptr() + (int64_t)(seq & 1) * this->plan_bytes_;
  }
  char *
  plan_host_bytes(uint64_t seq) {
    return (char *)this->plan_host_.data_ptr() + (int64_t)(seq & 1) * this->plan_bytes_;
  }

  // one block on the derive stream, then one D2H of the block (+ lane table) into pinned memory and
  // an event; parity double-buffered by the derive sequence number
  void
  launch_plan_block(
      torch::Tensor const &sps_dev, torch::Tensor const &uc_dev, cudaStream_t stream,
      int64_t *verdict = nullptr, int64_t cpr = 0) {
    const uint64_t seq = ++this->plan_seq_;
    char *dev = plan_dev_bytes(seq);
    A2AVCombinePlanBlockArguments args{
        .sps = sps_dev.data_ptr<int32_t>(),
        .uc = uc_dev.data_ptr<int32_t>(),
        .W = (int32_t)this->world_size,
        .ep_nexperts = (int32_t)this->ep_nexperts,
        .L = (int32_t)this->local_world_size,
        .nnodes = (int32_t)this->nnodes,
        .rank = (int32_t)this->rank,
        .seq = (int64_t)seq,
        .layout = this->plan_lay_,
        .block = reinterpret_cast<int64_t *>(dev),
        .lanes = reinterpret_cast<int32_t *>(dev + this->plan_lay_.total * sizeof(int64_t))};
    if (verdict != nullptr) {
      args.verdict = verdict;
      this->combine_wire->panel_rows(&args.lim_send, &args.lim_conv, &args.lim_wire);
      args.cpr = cpr;
      args.compress = this->a2av_compress ? 1 : 0;
    }
    a2av_combine_plan_block(args, stream);
    const size_t bytes =
        this->plan_lay_.total * sizeof(int64_t) + (2 * this->world_size + 1) * sizeof(int32_t);
    if (verdict == nullptr) {
      // (deferred verdict: no host reader of the block, as the dispatch's mirrors in dispatch_gemm.cc)
      CUDA_CHECK(cudaMemcpyAsync(plan_host_bytes(seq), dev, bytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaEventRecord(this->plan_event_, stream));
    this->plan_armed_ = true;
  }

  // host copy of this step's plan block: waits for its D2H (once per forward)
  int64_t const *
  wait_plan_block() {
    FLUX_CHECK(this->plan_armed_) << "combine forward without a fresh derive_combine_meta";
    {
      const auto t0 = std::chrono::steady_clock::now();
      for (;;) {
        const cudaError_t q = cudaEventQuery(this->plan_event_);
        if (q == cudaSuccess) {
          break;
        }
        FLUX_CHECK(q == cudaErrorNotReady) << "combine plan event: " << cudaGetErrorString(q);
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) {
          fprintf(stderr, "[zepp] combine: plan block %llu not on the host after 60 s\n",
                  (unsigned long long)this->plan_seq_);
          std::abort();
        }
        std::this_thread::yield();
      }
    }
    this->plan_armed_ = false;
    auto const *pb = reinterpret_cast<int64_t const *>(plan_host_bytes(this->plan_seq_));
    FLUX_CHECK_EQ(pb[CombinePlanLayout::kSeq], (int64_t)this->plan_seq_)
        << "combine plan block of another derive";
    return pb;
  }
  int64_t const *
  plan_dev_block() {
    return reinterpret_cast<int64_t const *>(plan_dev_bytes(this->plan_seq_));
  }
  int32_t const *
  plan_dev_lanes() {
    return reinterpret_cast<int32_t const *>(
        plan_dev_bytes(this->plan_seq_) + this->plan_lay_.total * sizeof(int64_t));
  }

  // the msplit problem tables (out, n_waves > 0) and / or the weight-gate map (wgate) of this forward,
  // built on the device from the plan block's node_base (one small launch; waves and the per-expert
  // gate indices travel as kernel arguments)
  void
  launch_dev_tables(
      int32_t *out,
      int32_t *wgate,
      std::vector<std::pair<int, int>> const &waves,
      int n_waves,
      bool reorder,
      int n_wgate,
      cudaStream_t stream,
      int32_t *dec = nullptr,  // deferred verdict: the device wave-adapt decision (out)
      int64_t adapt_reread = 0,
      int64_t adapt_row_bytes = 0) {
    const int E = this->ep_nexperts;
    FLUX_CHECK_LE(E, kMsplitMaxExperts) << "device msplit tables: too many experts per rank";
    FLUX_CHECK_LE(n_waves, (int)waves.size());
    A2AVMsplitTablesArguments a{};
    a.node_base = plan_dev_block() + this->plan_lay_.node_base;
    a.out = out;
    a.wgate = wgate;
    a.E = E;
    a.NN = this->nnodes;
    a.n_waves = n_waves;
    a.reorder = reorder ? 1 : 0;
    a.n_wgate = n_wgate;
    for (int w = 0; w < n_waves; w++) {
      a.wave_lo[w] = waves[w].first;
      a.wave_hi[w] = waves[w].second;
    }
    for (int e = 0; e < E; e++) {
      const int32_t g = this->wgate_armed_ ? this->wgate_of_expert_[e] : -1;
      FLUX_CHECK_LT(g, 32768) << "weight-gate index beyond the device table range";
      a.gate[e] = (int16_t)g;
    }
    // deferred verdict: the gate map written by the swap lane on the device (set_weight_gate gate_map)
    if (this->wgate_armed_ && this->wgate_map_dev_.defined()) {
      a.gate_dev = this->wgate_map_dev_.data_ptr<int32_t>();
    }
    if (dec != nullptr) {
      a.dec = dec;
      a.remote_rows = plan_dev_block() + CombinePlanLayout::kRemoteRows;
      a.adapt_reread = adapt_reread;
      a.adapt_ratio = get_a2av_rs_wave_adapt();
      a.adapt_row_bytes = adapt_row_bytes;
    }
    a2av_msplit_tables(a, stream);
  }
 private:

  void
  create_workspace_or_expand(int64_t workspace_size) {
    if (workspace_size <= 0)
      return;
    workspace_size = pad_to(workspace_size, 128);
    if (!this->workspace.defined() || workspace_size > this->workspace.numel()) {
      // captured layer graphs keep the address of the workspace they ran with: a superseded workspace stays
      // allocated (freed, the caching allocator would hand its memory to another owner under the graphs)
      if (this->workspace.defined()) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &st);
        fprintf(stderr, "[zepp] combine workspace grows %ld -> %ld bytes (capturing %d); the old one is kept\n",
                (long)this->workspace.numel(), (long)workspace_size, (int)(st != cudaStreamCaptureStatusNone));
        this->workspace_superseded_.push_back(this->workspace);
      }
      this->workspace = torch::empty(
          {workspace_size}, at::TensorOptions().dtype(at::ScalarType::Byte).device(at::kCUDA));
    }
  }
  std::vector<torch::Tensor> workspace_superseded_;

  c10::cuda::CUDAStream
  CreateReduceScatterStream() {
    at::cuda::CUDAGuard guard(at::cuda::current_device());
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &stream, cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
    return at::cuda::getStreamFromExternal(stream, at::cuda::current_device());
  }

  // tile-aware + a2av-aware. A split whose n_per is a multiple of the
  // 1024-wide tile is accepted first, dense and a2av alike.
  static int
  n_split_fixed(int n_split, int n_dim, bool a2av) {
    const int n_per = n_dim / n_split;
    if (n_per % kTileSizeN == 0) {
      return n_split;  // 1024-tile aligned: accept
    }
    if (a2av) {
      // the a2av combine only needs 8-elem pack alignment, so honor the requested n_split
      FLUX_CHECK(n_dim % n_split == 0 && (n_dim / n_split) % 8 == 0)
          << "a2av: n (" << n_dim << ") / n_split (" << n_split
          << ") must be a multiple of 8";
      return n_split;
    }
    if (n_dim % n_split == 0 && n_per % kTileSizeNMin == 0) {
      // the 512-tile dense lane accepts this split, so honor it
      return n_split;
    }
    if (n_dim % kTileSizeN == 0) {
      return n_dim / kTileSizeN;  // demote to 1024-wide splits (dense lanes only)
    }
    // 512-tile dense lane (e.g. H = 3584 = 7*512)
    if (n_dim % n_split == 0 && n_per % kTileSizeNMin == 0) {
      return n_split;
    }
    FLUX_CHECK_DIV(n_dim, kTileSizeNMin);
    return n_dim / kTileSizeNMin;
  }

 public:
  GemmCombineOpImpl(
      std::shared_ptr<Group> tp_group_,
      int64_t total_num_experts,
      int64_t max_m,
      int64_t n_dim,
      int64_t topk,
      at::ScalarType output_dtype,
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t max_input_groups,
      int64_t n_split_,
      CombineOptions const &options,
      int64_t nnodes_ = 1)
      : tp_group(tp_group_),
        total_num_experts(total_num_experts),
        max_m(max_m),
        n_dim(n_dim),
        topk(topk),
        output_dtype(output_dtype),
        max_input_groups(max_input_groups),
        rank(tp_group_->get_rank()),
        world_size(tp_group_->get_size()),
        tp_world_size(tp_world_size),
        ep_world_size(ep_world_size),
        nnodes(nnodes_),
        local_rank(DistEnv(tp_group_->get_rank(), tp_group_->get_size(), nnodes_).local_rank),
        local_world_size(tp_group_->get_size() / nnodes_),
        n_split(n_split_fixed(n_split_, n_dim, true)),
        a2av_compress(nnodes_ > 1) {
    if (this->n_split != n_split_) {
      FLUX_LOG_FIRST_N(WARN, 1) << "warning: (n / split_n) not aligned to the combine tile ("
                                << combine_tile_n(n_dim, this->n_split)
                                << "), set split_n=" << this->n_split << "\n";
    }
    FLUX_CHECK_EQ(this->tp_world_size * this->ep_world_size, this->world_size);
    FLUX_CHECK_DIV(this->total_num_experts, this->ep_world_size);
    FLUX_CHECK_LE(max_input_groups, kMaxNumGroups);
    FLUX_CHECK_GE(this->nnodes, 1);
    FLUX_CHECK_DIV(this->world_size, this->nnodes);
    {
      // scope: complete [1, hidden] GEMM rows (with tp > 1 each copy would be
      // a K-partial and a2av-of-copies would not apply), single weight group
      FLUX_CHECK_EQ(this->tp_world_size, 1) << "the fused combine requires tp_world_size == 1";
      FLUX_CHECK_EQ(this->max_input_groups, 1) << "the fused combine requires max_input_groups == 1";
      FLUX_CHECK_DIV(this->max_m, this->world_size);
    }
    this->ep_nexperts = this->total_num_experts / this->ep_world_size;
    int ep_rank = this->rank / this->tp_world_size;
    this->ep_start = this->ep_nexperts * ep_rank;
    this->gather_rs_stream = CreateReduceScatterStream();
    {
      at::cuda::CUDAGuard guard(at::cuda::current_device());
      CUDA_CHECK(cudaStreamCreateWithFlags(&this->meta_stream_, cudaStreamNonBlocking));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&this->gemm_start_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->gather_rs_done_event, cudaEventDisableTiming));
    // M-split waves: ctor-scoped mode resolution (allocation
    // implications: barrier sizing below + the wave-table arena)
    this->msplit_ = this->nnodes > 1 &&
                    (this->n_split == 1);
    if (this->msplit_) {
      FLUX_CHECK_EQ(this->n_split, 1) << "M-split waves require n_split == 1";
      FLUX_CHECK_LE(this->nnodes, kA2AVMaxNodes);
      FLUX_CHECK_LE(this->nnodes, 128)
          << "msplit wave flags must fit the 128-padded barrier flag region";
      this->msplit_wave_nodes_ = get_a2av_rs_wave_nodes();
      // wave_M + wave_off + prob_eid + prob_group (chunked) + ne_wave
      const int64_t tbl = (int64_t)4 * this->nnodes * this->ep_nexperts + this->nnodes;
      this->msplit_dev_ = empty_with_uninitialized_data(
          std::vector<int64_t>{tbl}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int));
      this->msplit_dec_ = torch::ones({1}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int));
      this->msplit_wave_of_node_.assign(this->nnodes, 0);
      this->msplit_node_order_.assign(this->nnodes, 0);
      this->fused_pack_ = get_a2av_rs_fused_pack() != 0;
      if (this->fused_pack_) {
        this->msplit_inv_pack_ = empty_with_uninitialized_data(
            std::vector<int64_t>{this->max_m},
            at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int));
      }
    }
    // shared identity indices: the ScatterD iterator always reads an index
    // array, so every non-fused problem points at this relative iota
    this->msplit_iota_ = torch::arange(
        this->max_m, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int));
    create_barriers();
    combine_wire = std::make_unique<CombineWire>(
        tp_group_,
        max_m,
        n_dim,
        topk,
        output_dtype,
        total_num_experts / ep_world_size,
        ep_world_size,
        this->barriers,
        this->n_split,
        options,
        nnodes_,
        this->a2av_compress);
    // the derive's persistent buffers: allocated here, on an idle device (the first derive may run
    // beside the spinning dispatch GEMM, where an allocation or a first kernel launch must not happen)
    ensure_derive_buffers();
  }



  void
  ensure_gemm_mark() {
    if (this->gemm_mark_.defined()) {
      return;
    }
    // the mark is written by a stream memop of the epoch (no pinned upload, no host event wait), so it lives
    // in cudaMalloc memory: with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True a caching-allocator
    // pointer is a virtual address that stream memops reject
    void *raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(raw, 0, sizeof(int64_t)));
    this->gemm_mark_raw_ = reinterpret_cast<uint64_t *>(raw);
    this->gemm_mark_ = torch::from_blob(
        raw, {1}, [](void *q) { cudaFree(q); },
        at::TensorOptions(at::kCUDA, at::cuda::current_device()).dtype(at::ScalarType::Long));
  }

  void
  resize_capacities(CombineOptions const &options) {
    FLUX_CHECK(this->combine_wire != nullptr);
    this->combine_wire->resize_capacities(options);
  }

  torch::Tensor
  gemm_start_mark() {
    ensure_gemm_mark();
    return this->gemm_mark_;
  }

  void
  set_prep_fork(int64_t event) {
    this->prep_fork_ev_ = reinterpret_cast<cudaEvent_t>(event);
    this->prep_fork_armed_ = event != 0;
  }

  void
  set_gemm_start_mark(int64_t epoch) {
    ensure_gemm_mark();
    this->gemm_mark_epoch_ = epoch;
    this->gemm_mark_armed_ = true;
  }

  void
  write_gemm_mark(cudaStream_t stream) {
    if (!this->gemm_mark_armed_) {
      return;
    }
    CU_CHECK(CUStreamWriteValue64(
        (CUstream)stream, reinterpret_cast<CUdeviceptr>(this->gemm_mark_raw_),
        static_cast<uint64_t>(this->gemm_mark_epoch_), CU_STREAM_WRITE_VALUE_DEFAULT));
    this->gemm_mark_armed_ = false;
  }

  void
  set_weight_gate(
      c10::optional<torch::Tensor> weight_signal,
      int64_t weight_signal_epoch,
      std::vector<int64_t> const &gate_of_expert,
      c10::optional<torch::Tensor> weight_ptr_override,
      c10::optional<torch::Tensor> gate_map) {
    // disarm
    this->wgate_override_ = torch::Tensor();
    this->wgate_map_dev_ = torch::Tensor();
    if (!weight_signal.has_value()) {
      this->wgate_armed_ = false;
      return;
    }
    if (weight_ptr_override.has_value()) {
      FLUX_CHECK(weight_ptr_override->is_cuda());
      FLUX_CHECK(weight_ptr_override->scalar_type() == at::ScalarType::Long)
          << "weight_ptr_override must be int64 device addresses";
      FLUX_CHECK(weight_ptr_override->is_contiguous());
      FLUX_CHECK_GE(weight_ptr_override->numel(), (int64_t)this->ep_nexperts)
          << "weight_ptr_override too small for ep_nexperts";
      this->wgate_override_ = *weight_ptr_override;
    }
    FLUX_CHECK(weight_signal->is_cuda());
    FLUX_CHECK(weight_signal->scalar_type() == at::ScalarType::Long)
        << "weight_signal must be int64 (u64 epoch signals)";
    FLUX_CHECK(weight_signal->is_contiguous());
    bool any = false;
    this->wgate_of_expert_.assign(this->ep_nexperts, -1);
    if (gate_map.has_value()) {
      // deferred verdict: the gate of every expert lives on the device (-1 ungated), so the gate is armed
      // every step; only the device table builders read it
      FLUX_CHECK(gate_map->is_cuda() && gate_map->scalar_type() == at::ScalarType::Int &&
                 gate_map->is_contiguous() && gate_map->numel() >= (int64_t)this->ep_nexperts)
          << "gate_map must be int32 CUDA with one entry per local expert";
      FLUX_CHECK_LE((int64_t)this->ep_nexperts - 1, weight_signal->numel())
          << "gate_map: weight_signal shorter than the gated slots";
      this->wgate_map_dev_ = *gate_map;
      any = true;
    } else {
      FLUX_CHECK_EQ((int64_t)gate_of_expert.size(), (int64_t)this->ep_nexperts)
          << "gate_of_expert must have one entry per local expert";
      for (int e = 0; e < this->ep_nexperts; e++) {
        const int64_t g = gate_of_expert[e];
        if (g >= 0) {
          FLUX_CHECK_LT(g, weight_signal->numel()) << "gate index beyond weight_signal";
          this->wgate_of_expert_[e] = (int32_t)g;
          any = true;
        }
      }
    }
    if (!any) {
      this->wgate_armed_ = false;
      return;
    }
    if (!this->wgate_dev_.defined()) {
      const int64_t cap =
          (int64_t)std::max<int64_t>(std::max<int64_t>(this->nnodes, 1), (int64_t)this->n_split * this->max_input_groups) *
          this->ep_nexperts + 128;
      this->wgate_dev_ = empty_with_uninitialized_data(
          std::vector<int64_t>{cap}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int));
    }
    this->wgate_signal_ = *weight_signal;
    this->wgate_epoch_ = (uint64_t)weight_signal_epoch;
    this->wgate_epoch_device_ = weight_signal_epoch < 0;  // < 0: read from the step slot
    this->wgate_armed_ = true;
  }

  torch::Tensor
  forward_impl(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> bias,
      c10::optional<std::vector<torch::Tensor>> input_scales,
      c10::optional<std::vector<torch::Tensor>> weight_scales,
      c10::optional<std::vector<torch::Tensor>> output_vec_scales,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::optional<UnifiedGemmHParams> const &hparams,
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> pack_index = c10::nullopt,
      c10::optional<torch::Tensor> reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> reduce_csr = c10::nullopt) {
    /*
      Note: When expert parallel is enabled, the inputs/weights tensor should be
      the partial the current expert parallel rank. But the splits_cpu and routing
      idx should be global no matter whether expert parallel is enabled, which means the
      splits_cpu/routing_idx should contains all the experts / tokens no matter whether expert
      parallel is enabled.
    */
    FLUX_CHECK(!bias.has_value());
    FLUX_CHECK_LE(inputs.size(), this->max_input_groups);
    int num_groups = inputs.size();
    FLUX_CHECK_LE(num_groups, this->max_input_groups);
    FLUX_CHECK_EQ(num_groups, weights.size());

    at::ScalarType input_torch_type = weights[0].scalar_type();
    FLUX_CHECK(input_torch_type != at::ScalarType::Char)
        << "Moe AG+Scatter INT8 not supported yet";
    bool is_fp8 = is_fp8_torch_dtype(input_torch_type);
    // if the dtype of input is fp8, use bfloat16 as the output dtype
    at::ScalarType output_torch_type = is_fp8 ? at::ScalarType::BFloat16 : input_torch_type;
    DataTypeEnum output_type = from_torch_dtype(output_torch_type);
    int m_full = routing_idx.size(0);
    int ntokens = m_full / this->topk;
    int n_tokens_per_rank = ntokens / this->world_size;
    int M_this_ep = inputs[0].size(0);
    int K = inputs[0].size(1);
    int E = weights[0].size(0);
    int N = weights[0].size(1);
    // check input/weight
    for (int i = 0; i < num_groups; i++) {
      CHECK_3D(weights[i], this->ep_nexperts, N, K);  // only RCR layout supported
      CHECK_INPUT(weights[i], input_torch_type);
      CHECK_2D(inputs[i], M_this_ep, K);
      CHECK_INPUT(inputs[i], input_torch_type);
    }
    // check input_scale/weight_scale/output_vec_scale
    if (input_scales.has_value()) {
      FLUX_CHECK_EQ(input_scales->size(), num_groups);
      for (auto &input_scale : input_scales.value()) {
        CHECK_1D(input_scale, 1);
        CHECK_INPUT(input_scale, at::ScalarType::Float);
      }
    }
    if (weight_scales.has_value()) {
      FLUX_CHECK_EQ(weight_scales->size(), num_groups);
      for (auto &weight_scale : weight_scales.value()) {
        CHECK_1D(weight_scale, E);
        CHECK_INPUT(weight_scale, at::ScalarType::Float);
      }
    }
    if (output_vec_scales.has_value()) {
      FLUX_CHECK_EQ(output_vec_scales->size(), num_groups);
      for (auto &output_vec_scale : output_vec_scales.value()) {
        CHECK_1D(output_vec_scale, M_this_ep);
        CHECK_INPUT(output_vec_scale, at::ScalarType::Float);
      }
    }

    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    if (this->ep_world_size == 1) {
      FLUX_CHECK_EQ(M_this_ep, m_full);
    } else {
      FLUX_CHECK_LE(M_this_ep, m_full) << "input.size(0) larger than routing_idx.size(0)";
    }
    FLUX_CHECK_DIV(m_full, this->world_size * this->topk);
    FLUX_CHECK_LE(m_full, this->max_m) << "input.size(0) " << M_this_ep << " larger than max_m\n";
    FLUX_CHECK_EQ(N, this->n_dim);

    FLUX_CHECK_GE(N, 8) << "N must be greater than or equal 8 for cutlass grouped gemm.";
    FLUX_CHECK_GE(K, 8) << "K must be greater than or equal 8 for cutlass grouped gemm.";
    torch::Tensor splits_gpu;
    if (!splits.is_cuda()) {
      splits_gpu = empty_with_uninitialized_data(
          splits.sizes(), at::TensorOptions(c10::kCUDA).dtype(at::ScalarType::Int));
      splits_gpu.copy_(splits, true);
    } else {
      splits_gpu = splits;
    }
    CHECK_INPUT(splits_gpu, at::ScalarType::Int);
    CHECK_1D(splits_gpu, this->total_num_experts);

    torch::Tensor a2av_pack_idx_t, a2av_reduce_idx_t;
    {
      FLUX_CHECK_EQ(num_groups, 1) << "the fused combine supports a single weight group";
      FLUX_CHECK(!is_fp8) << "the fused combine supports fp16/bf16 only";
      FLUX_CHECK(splits_per_source.has_value())
          << "the fused combine requires splits_per_source ([W, nexperts] int32 CPU)";
      auto const &cnt_t = splits_per_source.value();
      FLUX_CHECK(cnt_t.device().is_cpu()) << "splits_per_source must be a CPU tensor";
      CHECK_2D(cnt_t, this->world_size, this->total_num_experts);
      FLUX_CHECK(cnt_t.scalar_type() == at::ScalarType::Int);
      FLUX_CHECK(cnt_t.is_contiguous());
      // compress (dedup) routing-plan tensors: shape-validated here.
      // All-or-none per pair; the U matrix is host metadata, like
      // splits_per_source.
      FLUX_CHECK(wire_csr.has_value() == reduce_csr.has_value())
          << "pass both wire_csr and reduce_csr or neither";
      if (wire_csr.has_value()) {
        FLUX_CHECK(unique_counts.has_value())
            << "compress CSRs require unique_counts ([W, nnodes] int32 CPU)";
        FLUX_CHECK_EQ((int)wire_csr->size(), 2) << "wire_csr = [wire_ptr, wire_copy]";
        FLUX_CHECK_EQ((int)reduce_csr->size(), 2)
            << "reduce_csr = [red_ptr, red_row]";
        for (auto const &t : *wire_csr) {
          CHECK_INPUT(t, at::ScalarType::Int);
        }
        for (auto const &t : *reduce_csr) {
          CHECK_INPUT(t, at::ScalarType::Int);
        }
      }
      if (unique_counts.has_value()) {
        auto const &u_t = unique_counts.value();
        FLUX_CHECK(u_t.device().is_cpu()) << "unique_counts must be a CPU tensor";
        CHECK_2D(u_t, this->world_size, this->nnodes);
        FLUX_CHECK(u_t.scalar_type() == at::ScalarType::Int);
        FLUX_CHECK(u_t.is_contiguous());
      }
      if (this->a2av_compress) {
        FLUX_CHECK(unique_counts.has_value())
            << "a2av_hier_compress requires unique_counts ([W, nnodes] int32 CPU) -- "
               "host metadata, like splits_per_source";
        FLUX_CHECK(wire_csr.has_value())
            << "compress mode needs the combine metadata from derive_combine_meta";
      } else {
        FLUX_CHECK(!unique_counts.has_value() && !wire_csr.has_value())
            << "compress plan tensors require the a2av_hier_compress ctor flag";
      }
      FLUX_CHECK(pack_index.has_value() && reduce_index.has_value())
          << "the combine needs the pack_index and reduce_index of derive_combine_meta";
      a2av_pack_idx_t = pack_index.value();
      a2av_reduce_idx_t = reduce_index.value();
    }

    auto stream = c10::cuda::getCurrentCUDAStream();
    // set_prep_fork (one-shot): this forward's fork point for the side-stream table build
    const bool prep_fork = this->prep_fork_armed_;
    this->prep_fork_armed_ = false;
    bool invert_early = false;
    // the host copy of this step's combine plan block is waited for right before the first count-derived
    // read (the msplit wave decision, else the combine's checks).
    // deferred-verdict step (core/verdict.h; a forward is open): no count is read on the host outside the
    // wire. The input is capacity-sized (its rows bound this rank's rows), the GEMM is always launched, the
    // destination-wave decision is made on the device with both variants built from one table kernel, the
    // K-side scale fold and the pack inverse read the device row count.
    const bool deferred = verdict_armed() != nullptr;
    int64_t *verdict = deferred ? verdict_armed() : nullptr;
    int64_t adapt_reread = 0, adapt_row_bytes = 0;  // deferred: the wave-adapt rule's host constants
    int64_t const *plan_blk = nullptr;
    auto plan_block = [&]() -> int64_t const * {
      if (plan_blk == nullptr) {
        plan_blk = this->wait_plan_block();
      }
      return plan_blk;
    };

    ArchEnum arch = get_arch();
    SMCoreEnum sm_core = get_sm_core();
    auto input_type = from_torch_dtype(input_torch_type);
    auto dt_conf = to_gemm_dtype_config(
        make_gemm_dtype_config(input_type, input_type, output_type, output_type));
    auto impl_spec = make_gemm_v2_meta(fast_accum and dt_conf.is_input_fp8());
    // always use topk=1 impl: to save some compile time
    auto comm_spec = make_gather_rs_meta(1);
    auto meta = make_gemm_meta(
        dt_conf, arch, sm_core, _GatherRS{}, _RCR{}, _GemmGroupedV2{}(), impl_spec, comm_spec);
    // GEMM hparams are looked up per runtime shape (OpRegistry::get_hparams). Key the lookup by the
    // covering token bucket of the serving path (K, 2K, 4K, ... capped at the per-rank maximum,
    // python/zepp/serving.py token_buckets), not by this step's exact size: an exact bucket then
    // resolves to the kernel of its power-of-two bucket, which the warm-up has loaded, and no new
    // kernel can be selected mid-serving.
    const int64_t m_key = covering_bucket_rows(m_full, this->world_size, this->topk, this->max_m);
    auto rt_conf = make_runtime_config(N, (int)cute::ceil_div(m_key, (int64_t)this->ep_nexperts), K);
    OpRegistry::OpPtr gemm_op;
    if (hparams.has_value()) {
      gemm_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    std::vector<torch::Tensor> gemm_outs;
    for (int i = 0; i < num_groups; i++) {
      gemm_outs.push_back(empty_with_uninitialized_data(
          std::vector<int64_t>{M_this_ep, N},
          at::TensorOptions(at::kCUDA).dtype(output_torch_type)));
    }
    torch::Tensor output = empty_with_uninitialized_data(
        std::vector<int64_t>{n_tokens_per_rank, N},
        at::TensorOptions(at::kCUDA).dtype(output_torch_type));

    // ---- M-split waves: destination-wave GEMM sub-problems --
    // Built per iteration on the device from the plan block (a2av_msplit_tables).
    int msplit_n_waves = 0;
    int msplit_chunk_E = 0;  // chunked combine: 0 = wave-outer problem order
    // swap: per-wave moved-last problem order (gated experts drained last)
    bool wgate_reorder = false;
    const int32_t *msplit_wave_M_dev = nullptr;
    const int32_t *msplit_wave_off_dev = nullptr;
    const int32_t *msplit_ne_wave_dev = nullptr;
    const int32_t *msplit_eid_dev = nullptr;
    const int32_t *msplit_grp_dev = nullptr;
    // Byte-adaptive wave collapse (see get_a2av_rs_wave_adapt): when the
    // (n_waves-1) weight re-read passes outweigh the wire bytes the waves
    // could overlap, run this iteration on the single-gate problem
    // structure instead (msplit_run false => ws_args.msplit 0, per-split
    // problem_count, gemm n_split, fused pack off, set_msplit_waves(0)).
    bool msplit_run = this->msplit_;
    std::vector<std::pair<int, int>> waves;  // ascending home-node range [a, b) per wave
    bool dev_tables_done = false;            // msplit tables / weight-gate map launched
    if (this->msplit_ && get_a2av_rs_wave_adapt() > 0) {
      FLUX_CHECK(splits_per_source.has_value());
      const int NN = this->nnodes;
      const int E = this->ep_nexperts;
      const int L = this->local_world_size;
      const int my_node = this->rank / L;
      const int NG = this->msplit_wave_nodes_;
      int64_t remote_rows = 0;
      if (deferred) {
        // the rows stay on the device
      } else {
        remote_rows = plan_block()[CombinePlanLayout::kRemoteRows];
      }
      const int r1 = NN - 1 - my_node;
      const int n_waves_planned =
          1 + (r1 + NG - 1) / NG + (my_node + NG - 1) / NG;  // own + ring runs
      const int64_t elt_w = c10::elementSize(input_torch_type);
      const int64_t elt_o = c10::elementSize(output_torch_type);
      const int64_t reread_bytes =
          (int64_t)(n_waves_planned - 1) * E * N * K * elt_w;
      const int64_t wire_bytes = remote_rows * N * elt_o;
      if (deferred) {
        adapt_reread = reread_bytes;  // decided on the device (a2av_msplit_tables)
        adapt_row_bytes = (int64_t)N * elt_o;
      } else if (reread_bytes > (int64_t)get_a2av_rs_wave_adapt() * wire_bytes) {
        msplit_run = false;
      }
    }
    if (msplit_run) {
      FLUX_CHECK(splits_per_source.has_value());
      const int NN = this->nnodes;
      const int E = this->ep_nexperts;
      const int L = this->local_world_size;
      const int my_node = this->rank / L;
      const int NG = this->msplit_wave_nodes_;
      // Ring order own-node-last; waves = NG ring-consecutive nodes, never
      // crossing the my_node wrap (the intermediate's within-expert rows are
      // home-node ASCENDING — the dispatch's stable scatter order — so a wave must
      // be one contiguous ascending node range).
      {
        const bool own_first = get_a2av_rs_own_wave_first() != 0;
        if (own_first) {
          waves.emplace_back(my_node, my_node + 1);  // own node = FIRST wave
        }
        const int r1 = NN - 1 - my_node;  // ring run 1: nodes my+1 .. NN-1
        for (int s = 0; s < r1; s += NG) {
          waves.emplace_back(my_node + 1 + s, std::min(my_node + 1 + s + NG, NN));
        }
        for (int s = 0; s < my_node; s += NG) {  // ring run 2: nodes 0 .. my-1
          waves.emplace_back(s, std::min(s + NG, my_node));
        }
        if (!own_first) {
          waves.emplace_back(my_node, my_node + 1);  // own node = final wave
        }
      }
      msplit_n_waves = (int)waves.size();
      // schedule arrays, ONE source of truth for every consumer:
      // node_order[i] = i-th node in production order; wave_of_order[i] = its
      // cascade flag (== position of its wave). Ring mode reproduces the
      // (my_node+1+gi)%NN ring sequence exactly.
      {
        int pos = 0;
        for (int w = 0; w < msplit_n_waves; w++) {
          for (int node = waves[w].first; node < waves[w].second; node++, pos++) {
            this->msplit_node_order_[pos] = node;
            this->msplit_wave_of_node_[pos] = w;
          }
        }
        FLUX_CHECK_EQ(pos, NN);
        const int own_pos = get_a2av_rs_own_wave_first() != 0 ? 0 : NN - 1;
        FLUX_CHECK_EQ(this->msplit_node_order_[own_pos], my_node);
      }
      // chunked combine: problem-order permutation (chunk-outer, wave-
      // inner); chunk_E == 0 keeps the wave-outer linear order (maps not
      // shipped; the ws kernel uses i % ep_nexperts and the cascade the
      // uniform division).
      {
        int ce = 0;
        if (ce < 0) {
          const int64_t panel_bytes =
              (int64_t)N * K * c10::elementSize(input_torch_type);
          const int64_t l2 = (int64_t)get_a2av_rs_chunk_l2_mib() << 20;
          ce = (int)std::max<int64_t>(
              1, l2 / std::max<int64_t>(panel_bytes, 1));
        }
        msplit_chunk_E = ce > 0 ? std::min(ce, E) : 0;
      }
      const int n_probs = msplit_n_waves * E;
      wgate_reorder = this->wgate_armed_ && msplit_chunk_E == 0;
      // the tables (and the weight-gate map of the same problem order) built on the device from the plan
      // block's node_base: no host loop, no pinned arena, no host wait
      FLUX_CHECK_EQ(msplit_chunk_E, 0) << "device msplit tables: chunked problem order unsupported";
      if (this->wgate_armed_) {
        FLUX_CHECK_LE(n_probs, this->wgate_dev_.numel()) << "wgate map buffer too small";
      }
      if (!deferred) {
        plan_block();
      }
      cudaStream_t tstream = stream;
      if (prep_fork && deferred) {
        if (this->prep_stream_ == nullptr) {
          CUDA_CHECK(cudaStreamCreateWithFlags(&this->prep_stream_, cudaStreamNonBlocking));
          CUDA_CHECK(cudaEventCreateWithFlags(&this->prep_join_ev_, cudaEventDisableTiming));
        }
        // everything the tables and the pack inverse read is written by the combine metadata derive (plan
        // block, pack index) or the swap lane's arm (gate map), all ordered before the fork event
        CUDA_CHECK(cudaStreamWaitEvent(this->prep_stream_, this->prep_fork_ev_, 0));
        tstream = this->prep_stream_;
      }
      launch_dev_tables(
          this->msplit_dev_.data_ptr<int32_t>(),
          this->wgate_armed_ ? this->wgate_dev_.data_ptr<int32_t>() : nullptr,
          waves,
          msplit_n_waves,
          wgate_reorder,
          n_probs,
          tstream,
          deferred ? this->msplit_dec_.data_ptr<int32_t>() : nullptr,
          adapt_reread,
          adapt_row_bytes);
      dev_tables_done = true;
      if (tstream != stream) {
        if (msplit_run && this->fused_pack_) {
          a2av_invert_index(
              A2AVInvertIndexArguments{
                  a2av_pack_idx_t.data_ptr<int32_t>(),
                  this->msplit_inv_pack_.data_ptr<int32_t>(),
                  (int64_t)M_this_ep,
                  plan_dev_block() + CombinePlanLayout::kM},
              tstream);
          invert_early = true;
        }
        CUDA_CHECK(cudaEventRecord(this->prep_join_ev_, tstream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, this->prep_join_ev_, 0));
      }
      const int32_t *dp = this->msplit_dev_.data_ptr<int32_t>();
      msplit_wave_M_dev = dp;
      msplit_wave_off_dev = dp + (int64_t)n_probs;
      msplit_eid_dev = dp + (int64_t)2 * n_probs;
      msplit_grp_dev = dp + (int64_t)3 * n_probs;
      msplit_ne_wave_dev = dp + (int64_t)4 * n_probs;
    }

    MoeGatherRSWorkspaceArgs ws_args{
        .num_groups = num_groups,
        .N_split = this->n_split,
        .ep_start = this->ep_start,
        .ep_nexperts = this->ep_nexperts,
        .N = N,
        .K = K,
        .splits_gpu = splits_gpu.data_ptr<int>()};
    for (int i = 0; i < num_groups; i++) {
      ws_args.input[i] = inputs[i].data_ptr();
      ws_args.weights[i] = weights[i].data_ptr();
      ws_args.output[i] = gemm_outs[i].data_ptr();
      ws_args.input_scales[i] =
          input_scales.has_value() ? input_scales->at(i).data_ptr<float>() : nullptr;
      ws_args.weight_scales[i] =
          weight_scales.has_value() ? weight_scales->at(i).data_ptr<float>() : nullptr;
    }

    if (msplit_run) {
      ws_args.msplit = 1;
      ws_args.n_waves = msplit_n_waves;
      ws_args.wave_M = msplit_wave_M_dev;
      ws_args.wave_off = msplit_wave_off_dev;
      ws_args.non_empty_per_wave = msplit_ne_wave_dev;
      ws_args.n_flags = 0;  // per-wave flags
      ws_args.prob_eid = (msplit_chunk_E > 0 || wgate_reorder) ? msplit_eid_dev : nullptr;
      ws_args.barrier = this->barrier.data_ptr<int>();
    }
    ws_args.iota = this->msplit_iota_.data_ptr<int32_t>();
    ws_args.weight_ptr_override =
        this->wgate_override_.defined() ? this->wgate_override_.data_ptr<int64_t>() : nullptr;
    int64_t const *m_dev = deferred ? plan_dev_block() + CombinePlanLayout::kM : nullptr;
    if (deferred) {
      // the K-side fold below on the device row count, only when the device keeps the waves; and the
      // rows-bound assert (this rank's rows within the capacity-sized input) in every case
      const bool fold = msplit_run && this->fused_pack_ && output_vec_scales.has_value();
      a2av_fold_scales(
          A2AVFoldScalesArguments{
              .input = inputs[0].data_ptr(),
              .scales = fold ? output_vec_scales->at(0).data_ptr<float>() : nullptr,
              .rows_dev = m_dev,
              .dec = msplit_run ? this->msplit_dec_.data_ptr<int32_t>() : nullptr,
              .fold = fold ? 1 : 0,
              .k = (int64_t)K,
              .rows_bound = (int64_t)M_this_ep,
              .verdict = verdict},
          from_torch_dtype(input_torch_type),
          stream);
      if (msplit_run) {
        ws_args.msplit_dec = this->msplit_dec_.data_ptr<int32_t>();
      } else {
        // single-gate GEMM: a step without rows fires no cascade flag, the workspace kernel presets them
        ws_args.barrier = this->barrier.data_ptr<int>();
        ws_args.preset_empty = 1;
      }
    }
    if (msplit_run && this->fused_pack_) {
      // epilogue-fused pack: fold the gate coefficients into the
      // intermediate on the K side (mathematically identical: sum_j w_j (A_j B)
      // == sum_j (w_j A_j) B), build the pack inverse, and point D at the
      // dest-major send panel
      if (output_vec_scales.has_value() && !deferred) {
        inputs[0].mul_(output_vec_scales->at(0).unsqueeze(1));
      }
      if (!invert_early) {
        a2av_invert_index(
            A2AVInvertIndexArguments{
                a2av_pack_idx_t.data_ptr<int32_t>(),
                this->msplit_inv_pack_.data_ptr<int32_t>(),
                (int64_t)M_this_ep,
                m_dev},
            stream);
      }
      if (!deferred) {
        FLUX_CHECK_LE(
            (int64_t)M_this_ep, this->combine_wire->send_panel_rows())
            << "fused pack: send panel capacity (raise CombineOptions.max_send_rows)";
      }
      ws_args.fused_pack = 1;
      ws_args.inv_pack = this->msplit_inv_pack_.data_ptr<int32_t>();
      ws_args.send_panel = this->combine_wire->send_panel_ptr();
    }
    int problem_count = msplit_run
                            ? msplit_n_waves * ws_args.ep_nexperts
                            : ws_args.N_split * ws_args.num_groups * ws_args.ep_nexperts;
    torch::Tensor workspace_gpu = empty_with_uninitialized_data(
        std::vector<int64_t>{get_args_workspace_size(problem_count)},
        at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Char));
    void *workspace = workspace_gpu.data_ptr();
    make_workspace(
        ws_args,
        GemmLayoutEnum::RCR,
        c10::elementSize(input_torch_type),
        c10::elementSize(output_torch_type),
        workspace,
        stream);

    constexpr int kAlignment = 128;

    // the offsets
    int offset_problem_sizes = 0;
    int offset_ptr_A = pad_to(
        offset_problem_sizes + problem_count * sizeof(cutlass::gemm::GemmCoord), kAlignment);
    int offset_ptr_B = pad_to(offset_ptr_A + problem_count * sizeof(void *), kAlignment);
    int offset_ptr_C = pad_to(offset_ptr_B + problem_count * sizeof(void *), kAlignment);
    int offset_ptr_D = pad_to(offset_ptr_C + problem_count * sizeof(void *), kAlignment);
    int offset_lda = pad_to(offset_ptr_D + problem_count * sizeof(void *), kAlignment);
    int offset_ldb = pad_to(offset_lda + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldc = pad_to(offset_ldb + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldd = pad_to(offset_ldc + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldr = pad_to(offset_ldd + problem_count * sizeof(int64_t), kAlignment);
    int offset_scale_A = pad_to(offset_ldr + problem_count * sizeof(int64_t), kAlignment);
    int offset_scale_B = pad_to(offset_scale_A + problem_count * sizeof(float *), kAlignment);
    int offset_scatter_D = pad_to(offset_scale_B + problem_count * sizeof(float *), kAlignment);
    int offset_non_empty_problem_count =
        pad_to(offset_scatter_D + problem_count * sizeof(int *), kAlignment);
    // the ptrs
    cutlass::gemm::GemmCoord *problem_sizes =
        (cutlass::gemm::GemmCoord *)((char *)workspace + offset_problem_sizes);
    void **ptr_A = (void **)((char *)workspace + offset_ptr_A);
    void **ptr_B = (void **)((char *)workspace + offset_ptr_B);
    void **ptr_C = (void **)((char *)workspace + offset_ptr_C);
    void **ptr_D = (void **)((char *)workspace + offset_ptr_D);
    int64_t *lda = (int64_t *)((char *)workspace + offset_lda);
    int64_t *ldb = (int64_t *)((char *)workspace + offset_ldb);
    int64_t *ldc = (int64_t *)((char *)workspace + offset_ldc);
    int64_t *ldd = (int64_t *)((char *)workspace + offset_ldd);
    int64_t *ldr = (int64_t *)((char *)workspace + offset_ldr);
    float **scale_A = (float **)((char *)workspace + offset_scale_A);
    float **scale_B = (float **)((char *)workspace + offset_scale_B);
    int **scatter_D_ptr_ws = (int **)((char *)workspace + offset_scatter_D);
    int *non_empty_problem_count = (int *)((char *)workspace + offset_non_empty_problem_count);

    float alpha = 1.0, beta = 0.0;

    // swap: per-problem combine-side weight-gate map (device-built on
    // the FINAL problem order; -1 = ungated)
    const int32_t *wgate_dev = nullptr;
    const int n_probs_all = msplit_run ? msplit_n_waves * this->ep_nexperts
                                       : num_groups * this->ep_nexperts * this->n_split;
    if (this->wgate_armed_) {
      FLUX_CHECK_LE(n_probs_all, this->wgate_dev_.numel()) << "wgate map buffer too small";
      if (!dev_tables_done) {
        // single-gate problem list: wgate[i] = gate[i % E] (one group, n_split == 1), on the device
        FLUX_CHECK_EQ(num_groups, 1);
        if (!deferred) {
          plan_block();
        }
        launch_dev_tables(
            nullptr, this->wgate_dev_.data_ptr<int32_t>(), waves, 0, false, n_probs_all, stream);
      }
      wgate_dev = this->wgate_dev_.data_ptr<int32_t>();
    }

    GemmCombineArguments args{
        .problem_sizes = problem_sizes,
        .problem_count = problem_count,
        .non_empty_problem_count = non_empty_problem_count,
        .alpha = alpha,
        .beta = beta,
        .ptr_A = ptr_A,
        .ptr_B = ptr_B,
        .ptr_C = ptr_C,
        .ptr_D = ptr_D,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .ldd = ldd,
        .ldr = ldr,
        .scaleA = (float const **)scale_A,
        .scaleB = (float const **)scale_B,
        .topk = this->topk,
        .barrier = this->barrier.data_ptr<int>(),
        .routing_idx = routing_idx.data_ptr<int32_t>(),
        // msplit: the cascade's group axis is the WAVE (n_split stays 1 for
        // panels/columns — full-N sub-problems)
        .n_split = msplit_run ? msplit_n_waves : n_split,
        .sm_margin = sm_margin + (true
                                      ? get_a2av_pack_blocks() + get_a2av_reduce_blocks() +
                                            (this->a2av_compress ? get_a2av_prered_blocks() : 0)
                                      : get_rs_threadblock_count()),
        .non_empty_per_group = msplit_run ? msplit_ne_wave_dev : nullptr,
        .prob_group_map =
            (msplit_run && (msplit_chunk_E > 0 || wgate_reorder))
                ? msplit_grp_dev
                : nullptr,
        .scatter_D_ptr = scatter_D_ptr_ws,
        .prob_wgate_map = wgate_dev,
        .weight_signal_ptr = wgate_dev != nullptr
                                 ? reinterpret_cast<uint64_t const *>(this->wgate_signal_.data_ptr())
                                 : nullptr,
        .weight_signal_expected = wgate_dev != nullptr ? this->wgate_epoch_ : 0};
    if (wgate_dev != nullptr && this->wgate_epoch_device_) {
      // the swap lane's epoch of this step lives in the step slot (device step state)
      const int step_slot = step_current_slot();
      FLUX_CHECK(step_slot >= 0) << "weight_signal_epoch < 0 needs the device step state of the step";
      args.weight_signal_expected_ptr = step_slot_word(step_slot, StepSlot::kSwapEpoch);
    }
    this->wgate_armed_ = false;  // one-shot
    this->wgate_override_ = torch::Tensor();
    this->wgate_map_dev_ = torch::Tensor();

    int64_t workspace_size = gemm_op->get_workspace_size(args);
    this->create_workspace_or_expand(workspace_size);

    // ensure barrier initialized correctly
    CUDA_CHECK(cudaEventRecord(this->gemm_start_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(gather_rs_stream, this->gemm_start_event));

    this->write_gemm_mark(stream);  // swap: GEMM-start mark (one-shot)
    // deferred: always launched (a step without rows schedules no tile; the workspace kernel preset the
    // cascade flags the pack waits on)
    if (M_this_ep > 0 || deferred) {
      gemm_op->run(args, this->workspace.defined() ? this->workspace.data_ptr() : nullptr, stream);
    } else {
      this->barrier.fill_(1);
    }
    // the remote-lane receivers wait for the combine GEMM's end (no wide grid beside the resident GEMM)
    if (this->gemm_end_event_ == nullptr) {
      CUDA_CHECK(cudaEventCreateWithFlags(&this->gemm_end_event_, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(this->gemm_end_event_, stream));
    combine_wire->set_tail_reduce(this->gemm_end_event_);
    if (deferred) {
      // the wire section gets no host copy: nothing waits for the block on the host at all
      combine_wire->set_plan_block(nullptr, plan_dev_lanes());
    } else {
      combine_wire->set_plan_block(plan_block(), plan_dev_lanes());
    }
    if (deferred) {
      combine_wire->set_deferred(
          plan_dev_block(), msplit_run ? this->msplit_dec_.data_ptr<int32_t>() : nullptr, verdict);
    }
    combine_wire->set_msplit_waves(
        this->msplit_wave_of_node_,
        this->msplit_node_order_,
        msplit_run ? msplit_n_waves : 0,
        0);
    output = combine_wire->run(
        gemm_outs,
        output,
        this->ep_start,
        this->ep_nexperts,
        splits_gpu,
        routing_idx,
        output_vec_scales,
        true ? get_a2av_pack_blocks() : get_rs_threadblock_count(),
        (intptr_t)gather_rs_stream,
        splits_per_source,
        true ? c10::optional<torch::Tensor>(a2av_pack_idx_t) : c10::nullopt,
        true ? c10::optional<torch::Tensor>(a2av_reduce_idx_t) : c10::nullopt,
        this->a2av_compress ? unique_counts : c10::nullopt,
        this->a2av_compress ? wire_csr : c10::nullopt,
        this->a2av_compress ? reduce_csr : c10::nullopt);
    CUDA_CHECK(cudaEventRecord(this->gather_rs_done_event, gather_rs_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->gather_rs_done_event));

    // no end-of-layer synchronization orders the cross-rank buffer reuse of the next layer: every put of the
    // layer has a consumer that waited for its signal (signal after data), every layer-step starts with
    // the NCCL exchanges (loads, routing), collectives a rank enters only after its whole previous layer-step
    // (stream / graph order), and no cross-rank write of a layer-step precedes its first exchange; so a rank's
    // writes of layer k+1 follow every rank's layer k.
    this->barrier.zero_();
    this->combine_wire->reset_buffer();
    return output;
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> pack_index = c10::nullopt,
      c10::optional<torch::Tensor> reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> reduce_csr = c10::nullopt) {
    return forward_impl(
        {std::move(input)},
        {std::move(weight)},
        std::move(splits_cpu),
        std::move(routing_idx),
        as_optional_vec(bias),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_vec_scale),
        fast_accum,
        sm_margin,
        with_stream_sync,
        c10::nullopt,
        std::move(splits_per_source),
        std::move(pack_index),
        std::move(reduce_index),
        std::move(unique_counts),
        std::move(wire_csr),
        std::move(reduce_csr));
  }



  std::tuple<int64_t, int64_t, int64_t>
  get_pickle_info() const {
    return std::make_tuple(this->max_m, this->n_dim, this->ep_nexperts);
  }
};

CombineWire::CombineWire(
    std::shared_ptr<Group> tp_group,
    int max_m,
    int n_dim,
    int topk,
    at::ScalarType output_dtype,
    int ep_nexperts,
    int ep_world_size,
    std::vector<torch::Tensor> barriers,
    int n_split,
    CombineOptions const &options,
    int nnodes,
    bool a2av_compress)
    : impl_(new CombineWireImpl(
          tp_group,
          max_m,
          n_dim,
          topk,
          output_dtype,
          ep_nexperts,
          ep_world_size,
          barriers,
          n_split,
          options,
          nnodes,
          a2av_compress)) {}
CombineWire::~CombineWire() { delete impl_; }
void
CombineWire::set_tail_reduce(cudaEvent_t gemm_end) {
  FLUX_CHECK(impl_ != nullptr) << "CombineWire is not initialized";
  impl_->set_tail_reduce(gemm_end);
}
void
CombineWire::set_msplit_waves(
    std::vector<int> const &wave_of_node,
    std::vector<int> const &node_order,
    int n_waves,
    int n_chunk_flags) {
  FLUX_CHECK(impl_ != nullptr) << "CombineWire is not initialized";
  impl_->set_msplit_waves(wave_of_node, node_order, n_waves, n_chunk_flags);
}
void
CombineWire::set_plan_block(int64_t const *host_block, int32_t const *lanes_dev) {
  FLUX_CHECK(impl_ != nullptr) << "CombineWire is not initialized";
  impl_->set_plan_block(host_block, lanes_dev);
}
void
CombineWire::set_deferred(int64_t const *plan_dev, int32_t const *msplit_dec, int64_t const *verdict) {
  FLUX_CHECK(impl_ != nullptr) << "CombineWire is not initialized";
  impl_->set_deferred(plan_dev, msplit_dec, verdict);
}
void
CombineWire::panel_rows(int64_t *send, int64_t *conv, int64_t *wire) {
  FLUX_CHECK(impl_ != nullptr) << "CombineWire is not initialized";
  impl_->panel_rows(send, conv, wire);
}
void *
CombineWire::send_panel_ptr() {
  FLUX_CHECK(impl_ != nullptr) << "CombineWire is not initialized";
  return impl_->send_panel_ptr();
}
int64_t
CombineWire::send_panel_rows() {
  FLUX_CHECK(impl_ != nullptr) << "CombineWire is not initialized";
  return impl_->send_panel_rows();
}
void
CombineWire::resize_capacities(CombineOptions const &options) {
  impl_->resize_capacities(options);
}
uint64_t
CombineWire::run_id() const {
  return impl_->run_id();
}
void
CombineWire::advance_run_id(int64_t n) {
  impl_->advance_run_id(n);
}

void
CombineWire::reset_buffer() {
  FLUX_CHECK(impl_ != nullptr) << "CombineWire not initialized";
  impl_->reset_buffer();
}
torch::Tensor
CombineWire::run(
    std::vector<torch::Tensor> gemm_outs,  // of group_size
    c10::optional<torch::Tensor> output,
    int ep_start,
    int ep_nexperts,
    torch::Tensor splits,
    torch::Tensor routing_idx,
    c10::optional<std::vector<torch::Tensor>> output_vec_scales,
    int num_thread_blocks,
    intptr_t cp_stream,
    c10::optional<torch::Tensor> splits_per_source,
    c10::optional<torch::Tensor> pack_index,
    c10::optional<torch::Tensor> reduce_index,
    c10::optional<torch::Tensor> unique_counts,
    c10::optional<std::vector<torch::Tensor>> wire_csr,
    c10::optional<std::vector<torch::Tensor>> reduce_csr) {
  FLUX_CHECK(impl_ != nullptr) << "CombineWire not initialized";
  return impl_->run(
      std::move(gemm_outs),
      std::move(output),
      ep_start,
      ep_nexperts,
      std::move(splits),
      std::move(routing_idx),
      std::move(output_vec_scales),
      num_thread_blocks,
      cp_stream,
      std::move(splits_per_source),
      std::move(pack_index),
      std::move(reduce_index),
      std::move(unique_counts),
      std::move(wire_csr),
      std::move(reduce_csr));
}

GemmCombineOp::GemmCombineOp(
    std::shared_ptr<Group> tp_group_,
    int64_t total_num_experts,
    int64_t max_m,
    int64_t n_dim,
    int64_t topk,
    at::ScalarType output_dtype,
    int64_t tp_world_size,
    int64_t ep_world_size,
    int64_t max_input_groups,
    int64_t n_split_,
    CombineOptions const &options,
    int64_t nnodes)
    : impl_(new GemmCombineOpImpl(
          tp_group_,
          total_num_experts,
          max_m,
          n_dim,
          topk,
          output_dtype,
          tp_world_size,
          ep_world_size,
          max_input_groups,
          n_split_,
          options,
          nnodes)) {}

GemmCombineOp::~GemmCombineOp() { delete impl_; }
torch::Tensor
GemmCombineOp::forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync,
    c10::optional<torch::Tensor> splits_per_source,
    c10::optional<torch::Tensor> pack_index,
    c10::optional<torch::Tensor> reduce_index,
    c10::optional<torch::Tensor> unique_counts,
    c10::optional<std::vector<torch::Tensor>> wire_csr,
    c10::optional<std::vector<torch::Tensor>> reduce_csr) {
  FLUX_CHECK(impl_ != nullptr) << "GemmCombineOp not initialized";
  return impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(splits_cpu),
      std::move(routing_idx),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_vec_scale),
      fast_accum,
      sm_margin,
      with_stream_sync,
      std::move(splits_per_source),
      std::move(pack_index),
      std::move(reduce_index),
      std::move(unique_counts),
      std::move(wire_csr),
      std::move(reduce_csr));
}
torch::Tensor
GemmCombineOp::gemm_start_mark() {
  FLUX_CHECK(impl_ != nullptr) << "GemmCombineOp not initialized";
  return impl_->gemm_start_mark();
}
void
GemmCombineOp::resize_capacities(CombineOptions const &options) {
  impl_->resize_capacities(options);
}

void
GemmCombineOp::set_gemm_start_mark(int64_t epoch) {
  FLUX_CHECK(impl_ != nullptr) << "GemmCombineOp not initialized";
  impl_->set_gemm_start_mark(epoch);
}
void
GemmCombineOp::set_prep_fork(int64_t event) {
  FLUX_CHECK(impl_ != nullptr) << "GemmCombineOp not initialized";
  impl_->set_prep_fork(event);
}
void
GemmCombineOp::set_weight_gate(
    c10::optional<torch::Tensor> weight_signal,
    int64_t weight_signal_epoch,
    std::vector<int64_t> gate_of_expert,
    c10::optional<torch::Tensor> weight_ptr_override,
    c10::optional<torch::Tensor> gate_map) {
  FLUX_CHECK(impl_ != nullptr) << "GemmCombineOp not initialized";
  impl_->set_weight_gate(
      std::move(weight_signal), weight_signal_epoch, gate_of_expert, std::move(weight_ptr_override),
      std::move(gate_map));
}
std::vector<torch::Tensor>
GemmCombineOp::derive_combine_meta(
    torch::Tensor splits_gpu,
    torch::Tensor routing_idx,
    torch::Tensor splits_per_source,
    c10::optional<torch::Tensor> unique_counts,
    c10::optional<torch::Tensor> sps_dev,
    c10::optional<torch::Tensor> uc_dev,
    c10::optional<torch::Tensor> routing_ids) {
  FLUX_CHECK(impl_ != nullptr) << "GemmCombineOp not initialized";
  return impl_->derive_combine_meta(
      std::move(splits_gpu),
      std::move(routing_idx),
      std::move(splits_per_source),
      std::move(unique_counts),
      std::move(sps_dev),
      std::move(uc_dev),
      std::move(routing_ids));
}
int64_t
GemmCombineOp::meta_stream() {
  FLUX_CHECK(impl_ != nullptr) << "GemmCombineOp not initialized";
  return impl_->meta_stream();
}
void
GemmCombineOp::advance_run_id(int64_t n) {
  FLUX_CHECK(impl_ != nullptr) << "GemmCombineOp not initialized";
  impl_->advance_run_id(n);
}
uint64_t
GemmCombineOp::run_id() const {
  FLUX_CHECK(impl_ != nullptr) << "GemmCombineOp not initialized";
  return impl_->run_id();
}

}  // namespace bytedance::flux::ths_op
