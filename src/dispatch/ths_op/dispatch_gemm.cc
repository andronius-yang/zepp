//===- dispatch_gemm.cc ------------------------------ C++ ---===//
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
#include "dispatch/ths_op/dispatch_gemm.h"
#include <cstdlib>
#include "cute/tensor.hpp"
#include "cutlass/util/device_memory.h"
#include "flux/args/dispatch_gemm.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/cuda/kernel_registry.h"
#include "core/wire_runtime.h"
#include "core/verdict.h"
#include "core/step_state.h"
#include "dwire/dwire.h"
#include "flux/flux.h"
#include "flux/utils.h"
#include "core/tuning.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include "dispatch/sort_util.h"
#include "dispatch/workspace_util.h"
#include <nvshmemx.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>
#include <torch/cuda.h>
#include <torch/types.h>

namespace {
c10::optional<std::vector<torch::Tensor>>
as_optional_vec(c10::optional<torch::Tensor> &t) {
  if (t.has_value()) {
    return c10::optional<std::vector<torch::Tensor>>{{t.value()}};
  }
  return {};
}
}  // namespace

namespace bytedance::flux::ths_op {

/**
 * @return M_this_ep, M_this_ep_pad, gather_A_index, scatter_D_index, expert_idx, rank_start_idx,
 * rank_end_idx
 */

class DispatchGemmOp::DispatchGemmOpImpl {
 private:
  std::shared_ptr<Group> tp_group;
  const int rank;
  const int world_size;
  const int ep_size;
  const int nnodes;
  const DistEnv dist_env;
  const int ffn_tp_size;
  const int ep_rank;
  const int ffn_tp_rank;
  const int max_ntokens;
  const int N;
  const int hidden;
  const int nexperts;
  const int topk;
  at::ScalarType input_dtype;
  at::ScalarType output_dtype;
  const int32_t ep_nexperts;
  const int32_t ep_start;

  torch::Tensor workspace_buffer;

  // Idle streams and events (nothing is issued on them), created in a fixed order before the device wire's
  // streams: CUDA maps streams to the device's hardware work queues in creation order.
  c10::cuda::CUDAStream cp_stream;
  c10::cuda::CUDAStream cp_stream_inter_node;
  c10::cuda::CUDAStream pack_stream_;
  cudaEvent_t fetch_remote_event;
  cudaEvent_t all_gather_event;

  // Dispatch wire (all-to-all-v over NVSHMEM, issued by the device wire kernels, dwire/dwire.h). Each
  // (token, top-k slot) copy travels from the producer rank to the expert-owner rank, but a token
  // crosses a link at most once per destination rank (intra-node) and at most once per destination
  // node (inter-node): the node's copies are aggregated into one union per (source node, target
  // node), cut into L near-equal chunks by the minimal-move water-fill (lb_minmove_), and each local
  // relay rank gathers its chunk from its peers' send buffers over NVLink, stages it per round
  // (relay_slots_ slots) and puts it to the same-local-rank gateway on the target node. The gateway
  // forwards the whole staged union to every local rank and each receiver aliases its subset out of
  // the union through the consumer index, so GEMM problem sizes and schedule stay logical (several A
  // rows may alias one recv row). Every GEMM tile spins on the arrival signals of the sources it reads.
  std::vector<c10::cuda::CUDAStream> pull_streams_;    // idle (see cp_stream); one stream, nnodes > 1
  std::vector<cudaEvent_t> relay_pull_events_;         // idle, NN-1
  // Consumer metadata is built by fused kernels (sort_util): A rows are assigned per
  // (expert, source) group from the host offA table plus an atomic in-group rank, the gather
  // index is the dedup recv row from the mine-token cumsum, and the gating lanes are
  // histogrammed sort-free.
  // Minimal-move relay partition: each (source node -> target node) union is cut into L chunks
  // by the water-fill with minimal intra-node movement Sum_k (V_k - cap_k)^+ — relay k keeps
  // the first min(V_k, cap_k) rows of its OWN send segment and imports only the excess of
  // over-cap ranks. The remote union regions on the receiver are chunk-major and the consumer
  // build remaps the canonical dedup row through per-node piece tables (sort_util mm_*).
  const bool lb_minmove_;
  // Per-round relay staging: the staging holds relay_slots_ (S) round slots instead of all
  // NN-1 rounds' chunks; round dn is gathered into slot (dn - 1) % S and put from it, and the
  // gather of round dn waits for the put of round dn - S to return.
  const int relay_slots_;
  std::vector<cudaEvent_t> relay_put_events_;   // idle, NN-1
  static int64_t mm_words(int64_t NN, int64_t L) {
    // receiver tables (arena i64 words): off[NN+1] base[NN] lo/hi/dst[NN*2L]
    return (NN + 1) + NN + 3 * NN * 2 * L;
  }
  uint64_t run_id_ = 0;              // epoch value carried by the NVSHMEM signals
  int64_t max_recv_ntokens_ = 0;     // rows of the symmetric recv buffer
  int64_t max_stage_ntokens_ = 0;    // rows of the symmetric gateway staging buffer
  torch::Tensor a2av_send_buffer;    // symmetric [tokens_per_rank_max * topk, hidden]
  torch::Tensor a2av_recv_buffer;    // symmetric [max_recv_ntokens_, hidden]
  torch::Tensor a2av_signal_buffer;  // symmetric uint64[world_size], never memset
  // a2av_hier only (nnodes > 1): staging area for inbound node-aggregated
  // payloads, plus per-source-node arrival signals (epoch discipline, never memset)
  torch::Tensor a2av_stage_buffer_;            // symmetric [max_stage_ntokens_, hidden]
  torch::Tensor a2av_node_signal_buffer_;      // symmetric uint64[nnodes]
  // one-shot dispatch scratch: allocated once (setup), contents rebuilt every
  // iteration — routing is never cached across forwards
  torch::Tensor a2av_e_all_;        // i64 [n_copies_max] fused-kernel outputs...
  torch::Tensor a2av_s_all_buf_;    // i64 [n_copies_max]
  torch::Tensor a2av_flat_dst_;     // i64 [n_copies_max]
  torch::Tensor a2av_not_mine_;     // bool [n_copies_max]
  torch::Tensor a2av_pack_key_;     // i64 [copies_per_rank_max]
  // splits_per_source (metadata) path: the group tables of the step in a device
  // arena, built on the device from the routed counts (a2av_meta_arena_impl).
  // Layout: cumA/offA/offR_of_A i64 [G], expert_base i64 [nexperts],
  // sorted_splits_cumsum i32 [G], with G = ep_nexperts * world_size.
  torch::Tensor a2av_meta_dev_;     // device bytes
  int64_t meta_stride_ = 0;         // bytes of one meta arena parity slice
  int64_t send_half_rows_ = 0;      // rows of one send-buffer parity half
  // fixed-shape index scratch for the sync-free pack / consumer index builds (garbage-slot idiom,
  // no nonzero/masked_select)
  torch::Tensor a2av_mine_token_;   // i32 [max_ntokens + 1] (+1 = garbage slot)
  torch::Tensor a2av_pack_flag_;    // i32 [tokens_per_rank_max * (L + NN - 1)]
  torch::Tensor a2av_pack_gather_;  // i64 [tokens_per_rank_max * topk + 1]
  int64_t compress_meta_off_ = 0;   // 8-aligned offset of the compress fields in the meta arena
  // balanced inter-node relay (compress, nnodes > 1): each
  // round's canonical stream (the node's L union segments, ascending source
  // local rank) is cut into L near-equal chunks; local relay rank k stages and
  // wire-puts chunk k to the same-local-rank gateway on the target node. All
  // chunk boundaries derive from the replicated U matrix, so sender / relay /
  // gateway / destination agree with zero extra metadata.
  torch::Tensor a2av_relay_stage_;           // symmetric [max_relay_ntokens_, hidden]
  // pack announces (nnodes > 1): epoch word [source local rank * NN + target node], written at the node
  // peers by the pack-push kernel once that remote segment is packed, waited by their relay kernels
  torch::Tensor a2av_pack_seg_sig_;          // symmetric u64 [L * NN], epoch
  // Pack-done epoch words [seg], one per send segment (+ one spare): written by the pack-push kernel,
  // waited by the relay and wire kernels. Device memory, zeroed once; run_id_ only grows, so no reset.
  uint64_t *pack_words_ = nullptr;
  // per-(expert, gating-lane) inclusive cumsum [E, W]
  // where the remote node's lanes are WINDOW boundaries (data-dependent under
  // dedup, hence device-computed) and local lanes keep source boundaries.
  // Fed to args.accum_per_rank_ptr so the per-tile spin is window-keyed.
  torch::Tensor a2av_gating_cumsum_;
  torch::Tensor a2av_offA_lane_;  // i64 [E*W] lane-keyed A-order starts
  // persistent consumer-build outputs and the per-group
  // atomic rank counters (+ gating histogram in the second half)
  torch::Tensor a2av_sorted_gather_;   // i32 [n_copies_max]
  torch::Tensor a2av_sorted_scatter_;  // i32 [n_copies_max]
  torch::Tensor a2av_blk_cnt_;         // i32 [2 * E * W]: blk_cnt | gate_hist
  int64_t max_relay_ntokens_ = 0;            // rows of the symmetric relay staging buffer
  cudaEvent_t signal_done_event_ = nullptr;  // idle (see cp_stream)

  // Device-issued wire (dwire/dwire.h): the pack-push kernel (before GEMM 1) and the relay,
  // wire-warp and forward kernels (forked onto their own streams before GEMM 1, joined at the end of the step).
  bool dwire_forked_ = false;    // the relay / wire / forward kernels were launched (multi-node)
  torch::Tensor dwire_tables_;   // int64 [4][L]: per local rank: recv buffer, arrival signals, pack announces, send buffer
  torch::Tensor dwire_words_;    // u64 [2 * NN]: slot ready [NN], slot free [NN]
  torch::Tensor dwire_counters_; // u32: pack [nseg] | relay [NN] | forward [NN * L]
  std::vector<c10::cuda::CUDAStream> dwire_streams_;
  cudaEvent_t dwire_fork_ = nullptr;
  cudaEvent_t dwire_done_[3] = {nullptr, nullptr, nullptr};
  void
  dwire_init() {
    const int L = dist_env.local_world_size, NN = dist_env.nnodes;
    const int64_t nseg = (int64_t)L + NN - 1;
    auto i64 = torch::TensorOptions(torch::kCUDA).dtype(torch::kLong);
    auto i32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
    this->dwire_words_ = torch::zeros({2 * (int64_t)NN}, i64);
    this->dwire_counters_ = torch::zeros({nseg + NN + (int64_t)NN * L}, i32);
    for (int i = 0; i < 3; i++) {
      this->dwire_streams_.push_back(this->create_cp_stream());
      CUDA_CHECK(cudaEventCreateWithFlags(&this->dwire_done_[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&this->dwire_fork_, cudaEventDisableTiming));
  }
  // device peer table of the device wire (init and resize: buffers may move)
  void
  dwire_build_tables() {
    const int L = dist_env.local_world_size;
    std::vector<int64_t> t(4 * (size_t)L, 0);
    auto peer = [&](const torch::Tensor &buf, int dl) -> int64_t {
      if (!buf.defined()) {
        return 0;
      }
      const int pe = dist_env.local_rank_to_global_rank(dl, dist_env.node_idx);
      return reinterpret_cast<int64_t>(pe == this->rank ? buf.data_ptr() : nvshmem_ptr(buf.data_ptr(), pe));
    };
    for (int dl = 0; dl < L; dl++) {
      t[0 * L + dl] = peer(this->a2av_recv_buffer, dl);
      t[1 * L + dl] = peer(this->a2av_signal_buffer, dl);
      t[2 * L + dl] = peer(this->a2av_pack_seg_sig_, dl);
      t[3 * L + dl] = peer(this->a2av_send_buffer, dl);
      FLUX_CHECK(t[0 * L + dl] != 0 && t[1 * L + dl] != 0) << "device wire: node peer " << dl << " not P2P-mapped";
    }
    this->dwire_tables_ = torch::from_blob(t.data(), {4, (int64_t)L}, torch::kLong).to(torch::kCUDA);
  }
  // the relay, wire-warp and forward kernels are producers of GEMM 1's tiles: enqueued (on their own streams, after
  // everything enqueued on `stream` so far) right after the pack-push of the remote segments (they need only the
  // plan block and the pack words), joined into the caller's stream at the end of the step (finish_tail).
  void
  dwire_fork_wire(torch::Tensor const &inputs_shard, int par, cudaStream_t stream) {
    const DwireDispatchArgs a = this->dwire_args(inputs_shard, par);
    CUDA_CHECK(cudaEventRecord(this->dwire_fork_, stream));
    for (int i = 0; i < 3; i++) {
      CUDA_CHECK(cudaStreamWaitEvent(this->dwire_streams_[i], this->dwire_fork_, 0));
    }
    dwire_dispatch_relay(a, tuning::kDwireRelayBlocks, this->dwire_streams_[0]);
    dwire_dispatch_wire(a, this->dwire_streams_[1]);
    dwire_dispatch_forward(a, tuning::kDwireForwardBlocks, this->dwire_streams_[2]);
    for (int i = 0; i < 3; i++) {
      CUDA_CHECK(cudaEventRecord(this->dwire_done_[i], this->dwire_streams_[i]));
    }
    this->dwire_forked_ = true;
  }
  DwireDispatchArgs
  dwire_args(torch::Tensor const &inputs_shard, int par) {
    const int L = dist_env.local_world_size, NN = dist_env.nnodes;
    const int64_t nseg = (int64_t)L + NN - 1;
    const int64_t row_bytes = (int64_t)hidden * c10::elementSize(input_dtype);
    const int slot = step_current_slot();
    int64_t *tab = this->dwire_tables_.data_ptr<int64_t>();
    uint64_t *words = reinterpret_cast<uint64_t *>(this->dwire_words_.data_ptr<int64_t>());
    unsigned *ctr = reinterpret_cast<unsigned *>(this->dwire_counters_.data_ptr<int32_t>());
    DwireDispatchArgs a{};
    a.plan = this->rt_plan_dev_.data_ptr<int64_t>();
    a.run = slot >= 0 ? step_slot_word(slot, StepSlot::kDispatchRun) : nullptr;
    a.run_host = this->run_id_;
    a.L = L;
    a.NN = NN;
    a.my_node = dist_env.node_idx;
    a.my_lr = dist_env.local_rank;
    a.rank = this->rank;
    a.relay_slots = std::max(1, this->relay_slots_);
    a.row_bytes = row_bytes;
    a.relay_slot_rows = NN > 1 ? this->max_relay_ntokens_ / a.relay_slots : 0;
    a.src = static_cast<const char *>(inputs_shard.data_ptr());
    a.pack_gather = this->a2av_pack_gather_.data_ptr<int64_t>();
    a.send_base = reinterpret_cast<char *>(this->a2av_send_buffer.data_ptr()) +
                  (int64_t)par * this->send_half_rows_ * row_bytes;
    a.recv_peer = reinterpret_cast<char *const *>(tab + 0 * L);
    a.sig_peer = reinterpret_cast<uint64_t *const *>(tab + 1 * L);
    a.segsig_peer = reinterpret_cast<uint64_t *const *>(tab + 2 * L);
    a.peer_send = reinterpret_cast<const char *const *>(tab + 3 * L);
    a.pack_words = this->pack_words_;
    a.pack_counters = ctr;
    a.relay_counters = ctr + nseg;
    a.fwd_counters = ctr + nseg + NN;
    a.slot_ready = words;
    a.slot_free = words + NN;
    a.relay_base = NN > 1 ? reinterpret_cast<char *>(this->a2av_relay_stage_.data_ptr()) : nullptr;
    a.stage_base = NN > 1 ? reinterpret_cast<char *>(this->a2av_stage_buffer_.data_ptr()) : nullptr;
    a.node_sig = NN > 1 ? reinterpret_cast<uint64_t *>(this->a2av_node_signal_buffer_.data_ptr()) : nullptr;
    a.kill = kill_word_dev();
    return a;
  }

 private:
  c10::cuda::CUDAStream
  create_cp_stream() const {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, CU_STREAM_NON_BLOCKING));
    return at::cuda::getStreamFromExternal(stream, at::cuda::current_device());
  }


 public:
  DispatchGemmOpImpl(
      std::shared_ptr<Group> tp_group,
      int ep_size,
      int nnodes,
      int max_ntokens,
      int ffn_hidden,  // before TP shard
      int hidden,
      int nexperts,
      int topk,
      at::ScalarType input_dtype,
      at::ScalarType output_dtype,
      DispatchOptions const &options)
      : tp_group(tp_group),
        world_size(tp_group->get_size()),
        ep_size(ep_size),
        nnodes(nnodes),
        dist_env(tp_group->get_rank(), tp_group->get_size(), nnodes),
        ffn_tp_size(world_size / ep_size),
        rank(tp_group->get_rank()),
        ffn_tp_rank(rank % ffn_tp_size),
        ep_rank(rank / ffn_tp_size),
        max_ntokens(max_ntokens),
        N(ffn_hidden / ffn_tp_size),
        hidden(hidden),
        nexperts(nexperts),
        topk(topk),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        ep_nexperts(nexperts / ep_size),
        ep_start(this->ep_nexperts * ep_rank),
        cp_stream(create_cp_stream()),
        cp_stream_inter_node(create_cp_stream()),
        pack_stream_(create_cp_stream()),
        lb_minmove_(nnodes > 1),
        relay_slots_(tuning::kRelaySlots) {
    CHECK_DIV(nexperts, ep_size);
    CHECK_DIV(ffn_hidden, ffn_tp_size);
    FLUX_CHECK_GE(nnodes, 1);
    CHECK_DIV(world_size, nnodes);
    // the predicate segment gate is W-unbounded; the sort/schedule kernels' shared-memory
    // tables cap the world size
    FLUX_CHECK_LE(world_size, 128) << "world_size is capped at 128 ranks";
    {
      if (nnodes > 1) {
        this->pull_streams_.push_back(create_cp_stream());
        for (int i = 0; i < nnodes - 1; i++) {
          cudaEvent_t ev = nullptr;
          CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
          this->relay_pull_events_.push_back(ev);
        }
        for (int i = 0; i < nnodes - 1; i++) {
          cudaEvent_t ev = nullptr;
          CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
          this->relay_put_events_.push_back(ev);
        }
      }
      FLUX_CHECK_EQ(this->ffn_tp_size, 1) << "a2av dispatch requires ep_size == world_size";
      FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == dist_env.local_rank);
      int64_t tokens_per_rank_max = (max_ntokens + world_size - 1) / world_size;
      this->max_recv_ntokens_ = (int)options.max_recv_rows;
      this->send_half_rows_ = tokens_per_rank_max * topk;
      this->a2av_send_buffer = nvshmem_create_tensor(
          {this->send_half_rows_ * (1), hidden}, input_dtype);
      this->a2av_recv_buffer =
          nvshmem_create_tensor({this->max_recv_ntokens_, hidden}, input_dtype);
      this->a2av_signal_buffer =
          nvshmem_create_tensor({world_size}, at::ScalarType::Long, /*init_zero=*/true);
      if (nnodes > 1) {
        // gateway staging: holds the node-aggregated inbound payloads from the
        // nnodes-1 same-local-rank peers; expected load ~= one rank's recv (the
        // node's inbound traffic splits across L gateways by source local rank)
        this->max_stage_ntokens_ = (int)options.max_stage_rows;
        this->a2av_stage_buffer_ =
            nvshmem_create_tensor({this->max_stage_ntokens_, hidden}, input_dtype);
        this->a2av_node_signal_buffer_ =
            nvshmem_create_tensor({nnodes}, at::ScalarType::Long, /*init_zero=*/true);
        {
          // balanced-relay staging: the relay_slots_ round slots of MY wire chunks
          // (max_relay_rows / relay_slots_ rows each)
          const int64_t L = world_size / nnodes;
          this->max_relay_ntokens_ = (int)options.max_relay_rows;
          this->a2av_relay_stage_ =
              nvshmem_create_tensor({this->max_relay_ntokens_, hidden}, input_dtype);
          this->a2av_pack_seg_sig_ = nvshmem_create_tensor(
              {(int64_t)L * nnodes}, at::ScalarType::Long, /*init_zero=*/true);
        }
      }
      const int64_t n_copies_max = tokens_per_rank_max * (int64_t)topk * world_size;
      auto opt_cuda_i64 = torch::TensorOptions(torch::kCUDA).dtype(torch::kLong);
      this->a2av_e_all_ = torch::empty({n_copies_max}, opt_cuda_i64);
      this->a2av_s_all_buf_ = torch::empty({n_copies_max}, opt_cuda_i64);
      this->a2av_flat_dst_ = torch::empty({n_copies_max}, opt_cuda_i64);
      this->a2av_not_mine_ =
          torch::empty({n_copies_max}, torch::TensorOptions(torch::kCUDA).dtype(torch::kBool));
      this->a2av_pack_key_ = torch::empty({tokens_per_rank_max * (int64_t)topk}, opt_cuda_i64);
      const int64_t meta_groups = (int64_t)this->ep_nexperts * world_size;
      const int64_t meta_bytes = 3 * meta_groups * sizeof(int64_t) + nexperts * sizeof(int64_t) +
                                 meta_groups * sizeof(int32_t);
      // compress appends (8-aligned): send-segment offsets i64[nseg + 1] with
      // nseg = L + NN - 1, then the tables below; covered by the same single
      // H2D upload.
      int64_t total_meta_bytes = meta_bytes;
      {
        const int64_t L = world_size / nnodes;
        const int64_t nseg = L + nnodes - 1;
        const int64_t R = nnodes - 1;
        // union bcast forwards whole unions: no forward-index tables; the
        // gating searchsorted queries i64[E * (W + 1)] ride the upload instead
        // minmove appends the receiver piece tables (mm_words) behind gate_q
        const int64_t extra = (int64_t)this->ep_nexperts * (world_size + 1) +
                              (this->lb_minmove_ ? mm_words(nnodes, L) : 0);
        this->compress_meta_off_ = pad_to(meta_bytes, (int64_t)8);
        total_meta_bytes =
            this->compress_meta_off_ + (nseg + 1 + extra) * (int64_t)sizeof(int64_t);
      }
      // one meta arena slice (meta_par = 1): the GEMM reads it (ssc_dev /
      // accum_per_rank_ptr) for its whole runtime
      this->meta_stride_ = total_meta_bytes;
      const int64_t meta_par = 1;
      this->a2av_meta_dev_ = torch::empty(
          {total_meta_bytes * meta_par}, torch::TensorOptions(torch::kCUDA).dtype(torch::kByte));
      {
        const int64_t L = world_size / nnodes;
        const int64_t nseg = L + nnodes - 1;
        auto opt_cuda_i32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
        this->a2av_mine_token_ = torch::empty({(int64_t)max_ntokens + 1}, opt_cuda_i32);
        this->a2av_pack_flag_ = torch::empty({tokens_per_rank_max * nseg}, opt_cuda_i32);
        {
          this->a2av_sorted_gather_ = torch::empty({n_copies_max}, opt_cuda_i32);
          this->a2av_sorted_scatter_ = torch::empty({n_copies_max}, opt_cuda_i32);
          this->a2av_blk_cnt_ = torch::empty({2 * meta_groups}, opt_cuda_i32);
        }
        this->a2av_pack_gather_ =
            torch::empty({tokens_per_rank_max * (int64_t)topk + 1}, opt_cuda_i64);
      }
      if (rank == 0) {
        double sym_mb = (this->a2av_send_buffer.nbytes() + this->a2av_recv_buffer.nbytes() +
                         this->a2av_signal_buffer.nbytes()) /
                        1024.0 / 1024.0;
        if (this->a2av_stage_buffer_.defined()) {
          sym_mb += (this->a2av_stage_buffer_.nbytes() + this->a2av_node_signal_buffer_.nbytes()) /
                    1024.0 / 1024.0;
        }
        if (this->a2av_relay_stage_.defined()) {
          sym_mb += this->a2av_relay_stage_.nbytes() / 1024.0 / 1024.0;
        }
        fprintf(
            stderr,
            "[flux a2av] recv rows %ld send rows %ld -> %.0f MiB symmetric heap per rank\n",
            (long)this->max_recv_ntokens_,
            (long)(tokens_per_rank_max * topk),
            sym_mb);
      }
      {
        // Lazy-loading guard (flux/cuda/kernel_registry.h): under CUDA_MODULE_LOADING=LAZY a
        // kernel's code loads at its FIRST launch, and while a load runs with a kernel resident the GPU
        // takes no new work from any thread; the persistent GEMM would strand that first load behind
        // resident spin kernels. Load every kernel of the library (metadata, workspace, the GEMM of every
        // registered op, the combine, the planner) and every NVSHMEM on-stream primitive variant now, idle
        // device, collective.
        preload_all_kernels();
        cudaStream_t pstream = c10::cuda::getCurrentCUDAStream();
        uint64_t *psig = reinterpret_cast<uint64_t *>(this->a2av_signal_buffer.data_ptr());
        nvshmem_prime_primitives(psig, this->rank, dist_env.local_world_size, nnodes, pstream);
      }
    }
    // the device wire indexes pack words by segment (L + NN - 1 of them, also on one node)
    const int64_t nw = (int64_t)dist_env.local_world_size + dist_env.nnodes;
    CUDA_CHECK(cudaMalloc(&this->pack_words_, sizeof(uint64_t) * nw));
    CUDA_CHECK(cudaMemset(this->pack_words_, 0, sizeof(uint64_t) * nw));
    this->dwire_init();
    this->dwire_build_tables();
    wire_runtime_init();  // kill word + spin-kernel preload, on an idle device
    verdict_dev();        // the deferred verdict block: allocated here, on an idle device
    CUDA_CHECK(cudaEventCreateWithFlags(&this->fetch_remote_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->all_gather_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->signal_done_event_, cudaEventDisableTiming));
  }

  ~DispatchGemmOpImpl() {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(this->pack_words_));
    CUDA_CHECK(cudaEventDestroy(this->signal_done_event_));
    CUDA_CHECK(cudaEventDestroy(this->all_gather_event));
    CUDA_CHECK(cudaEventDestroy(this->fetch_remote_event));
    for (auto &ev : this->relay_put_events_) {
      CUDA_CHECK(cudaEventDestroy(ev));
    }
    for (auto &ev : this->relay_pull_events_) {
      CUDA_CHECK(cudaEventDestroy(ev));
    }
    for (auto &s : this->pull_streams_) {
      CUDA_CHECK(cudaStreamDestroy(s));
    }
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream_inter_node));
    CUDA_CHECK(cudaStreamDestroy(this->pack_stream_));
  }

 protected:
  auto
  get_gemm_meta(bool fast_accum) const {
    auto arch = get_arch();
    auto sm_core = get_sm_core();
    auto gemm_layout = _RCR{};  // TODO(houqi.1993) only RCR supported
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    auto dt_conf = make_gemm_dtype_config(input_dtype, input_dtype, output_dtype, output_dtype);
    auto v2_meta = make_gemm_v2_meta(fast_accum && dt_conf.is_input_fp8());
    auto meta = make_gemm_meta(
        dt_conf, arch, sm_core, _AGScatter{}, gemm_layout, _GemmGroupedV2{}, v2_meta);
    return meta;
  }

  // the GEMM hparams lookup key does not depend on the step's token count, so every bucket (exact or
  // power-of-two) resolves to the same kernel, loaded by the kernel registry at construction
  auto
  get_rt_conf() const {
    return make_runtime_config(512, this->N, this->hidden);
  }


  struct A2AVDispatchState {
    // index tensors are allocated at full n_copies size (fixed shapes keep the
    // build sync-free); only the first M_this_ep rows are valid and the GEMM
    // reads exactly that many via data_ptr + M_this_ep
    torch::Tensor sorted_gather_index;   // int32: sorted-A row -> recv-buffer row
    torch::Tensor sorted_scatter_index;  // int32: sorted-D row -> per-expert D row
    torch::Tensor sorted_splits_cumsum;  // int32 [ep_nexperts, world_size]
    int M_this_ep = 0;
  };

  A2AVDispatchState
  a2av_dispatch(
      torch::Tensor const &inputs_shard,
      torch::Tensor const &splits_gpu,
      torch::Tensor const &scatter_index,
      const int32_t *cnt_host,  // [W, nexperts] splits_per_source (derive_routed_meta's pinned copy)
      const int32_t *uc_host,   // [W, W + nnodes] unique_counts
      cudaStream_t stream,
      // deferred-verdict step (core/verdict.h): no count is read on the host outside the wire; the GEMM rows
      // are bounded by rows_bound (the caller's output rows) and the pack bounds live on the device
      bool deferred = false,
      int64_t rows_bound = 0) {
    const int W = this->world_size;
    const int64_t E = this->ep_nexperts;
    const int tokens_per_rank = inputs_shard.size(0);
    const int64_t copies_per_rank = (int64_t)tokens_per_rank * topk;
    const int64_t n_copies = copies_per_rank * W;
    FLUX_CHECK(cnt_host != nullptr && uc_host != nullptr)
        << "compress mode requires splits_per_source and unique counts";
    torch::Tensor seg_off_dev, mm_dev;
    this->run_id_ += 1;
    // the send buffer and the meta arena hold one slice (par 0); the pack runs on the caller's stream
    const int par = 0;
    cudaStream_t pack_str = stream;

    // ---- metadata: everything the wire and the schedule need comes from the device-built plan block
    // and meta arena of this step (derive_routed_meta).
    const int64_t nexG = E * W;  // number of (expert_loc, source) groups
    int64_t M_this_ep = 0;
    torch::Tensor offA_dev, expert_base_dev, ssc_dev, gate_q_dev;
    // the dispatch plan block every later stage reads (sort_util.h DispatchPlan), device-built by
    // derive_routed_meta
    const int64_t *P = nullptr;
    FLUX_CHECK(this->rt_plan_fresh_ && cnt_host == this->rt_sps_cpu_.data_ptr<int32_t>())
        << "dispatch: the plan comes from derive_routed_meta of this step";
    this->rt_plan_fresh_ = false;
    FLUX_CHECK_EQ(this->rt_topk_ids_.numel(), scatter_index.numel());
    // stage 1 reads the expert of each copy
    const int32_t *routing_ids = this->rt_topk_ids_.data_ptr<int32_t>();
    {
      const int64_t nex = this->nexperts;
      char *dev_slice = reinterpret_cast<char *>(this->a2av_meta_dev_.data_ptr()) +
                        (int64_t)par * this->meta_stride_;
      // the arena is derived on the device from the counts landed by derive_routed_meta
      FLUX_CHECK(
          this->routed_meta_ready_ && cnt_host == this->rt_sps_cpu_.data_ptr<int32_t>())
          << "dispatch: the splits_per_source must come from derive_routed_meta";
      // planning branches: derive_routed_meta already built this step's arena (stream-ordered before this point)
      if (!this->rt_arena_early_) {
        a2av_meta_arena_impl(this->arena_args(dev_slice), pack_str);
      }
      auto opt_dev_i64 = torch::TensorOptions(torch::kCUDA).dtype(torch::kLong);
      auto opt_dev_i32 = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt);
      char *dev = reinterpret_cast<char *>(this->a2av_meta_dev_.data_ptr()) +
                  (int64_t)par * this->meta_stride_;
      offA_dev = torch::from_blob(dev + nexG * 8, {nexG}, opt_dev_i64);
      expert_base_dev = torch::from_blob(dev + 3 * nexG * 8, {nex}, opt_dev_i64);
      ssc_dev = torch::from_blob(dev + 3 * nexG * 8 + nex * 8, {E, (int64_t)W}, opt_dev_i32);
      {
        const int64_t L = dist_env.local_world_size;
        const int64_t NN = dist_env.nnodes;
        const int64_t nseg = L + NN - 1;
        seg_off_dev = torch::from_blob(dev + this->compress_meta_off_, {nseg}, opt_dev_i64);
        if (NN > 1) {
          gate_q_dev = torch::from_blob(
              dev + this->compress_meta_off_ + (nseg + 1) * 8,
              {(int64_t)this->ep_nexperts * ((int64_t)W + 1)},
              opt_dev_i64);
          if (this->lb_minmove_) {
            mm_dev = torch::from_blob(
                dev + this->compress_meta_off_ +
                    (nseg + 1 + (int64_t)this->ep_nexperts * ((int64_t)W + 1)) * 8,
                {mm_words(NN, L)},
                opt_dev_i64);
          }
        }
      }
    }
    {
      // the plan block of this step
      using DP = DispatchPlan;
      P = this->rt_plan_cpu_.data_ptr<int64_t>();
      // deferred: the rows this rank computes stay on the device (the GEMM problem sizes come from the
      // device splits); the host sizes everything by the bound
      M_this_ep = deferred ? rows_bound : P[DP::kM];
      // identical on every rank (replicated inputs), so a failure is collective. Deferred: the plan kernel
      // ORs these bits into the verdict block (capacity bits through the demands kernel), checked once per
      // forward on the host
      if (!deferred) {
        FLUX_CHECK_EQ(P[DP::kErr], 0)
            << "dispatch plan error bits " << P[DP::kErr]
            << " (1 water-fill room, 2 pieces, 4 unique counts, 8 node union, 16 recv capacity, "
               "32 send rows, 64 gateway staging, 128 relay staging)";
      }
    }

    // ---- stage 1 (pre-wire, minimal): one fused kernel decodes every copy,
    // fills all stage-2 inputs and the pack flags; then the producer pack. No
    // host sync of any kind in this stage — CUDA bincount/nonzero are banned
    // (both hide a full stream drain).
    auto e_all = this->a2av_e_all_.narrow(0, 0, n_copies);
    auto s_all = this->a2av_s_all_buf_.narrow(0, 0, n_copies);
    auto flat_dst = this->a2av_flat_dst_.narrow(0, 0, n_copies);
    auto not_mine = this->a2av_not_mine_.narrow(0, 0, n_copies);
    const int64_t nseg_c = (int64_t)dist_env.local_world_size + dist_env.nnodes - 1;
    {
      // pre-zero the seg-major [nseg, tokens] pack flags the fused stage1
      // kernel writes (the pack scan below turns them into pack_gather)
      CUDA_CHECK(cudaMemsetAsync(
          this->a2av_pack_flag_.data_ptr(),
          0,
          (size_t)(nseg_c * tokens_per_rank) * sizeof(int32_t),
          pack_str));
      {
        // fused consumer build: stage 1 writes the per-token keep flags,
        // incl. the +1 garbage slot
        CUDA_CHECK(cudaMemsetAsync(
            this->a2av_mine_token_.data_ptr(),
            0,
            (size_t)((int64_t)tokens_per_rank * W + 1) * sizeof(int32_t),
            pack_str));
      }
    }
    a2av_stage1_impl(
        A2AVStage1Arguments{
            .scatter_index = scatter_index.data_ptr<int32_t>(),
            .splits = splits_gpu.data_ptr<int32_t>(),
            .routing_ids = routing_ids,
            .nexperts = this->nexperts,
            .ep_nexperts = (int)E,
            .world_size = W,
            .rank = rank,
            .copies_per_rank = copies_per_rank,
            .n_copies = n_copies,
            .e_all = e_all.data_ptr<int64_t>(),
            .s_all = s_all.data_ptr<int64_t>(),
            .flat_dst = flat_dst.data_ptr<int64_t>(),
            .not_mine = not_mine.data_ptr<bool>(),
            // the counts and expert_base come from the device meta arena
            .expert_base = nullptr,
            .chunks = nullptr,
            .pack_key = this->a2av_pack_key_.data_ptr<int64_t>(),
            .pack_flag = this->a2av_pack_flag_.data_ptr<int32_t>(),
            .topk = (int)topk,
            .local_world_size = dist_env.local_world_size,
            .node_idx = dist_env.node_idx,
            .mine_token = this->a2av_mine_token_.data_ptr<int32_t>(),
            .union_bcast = dist_env.nnodes > 1,
            .verdict = deferred ? verdict_armed() : nullptr},
        pack_str);

    {
      // compressed producer pack, fused: the stage1 kernel already wrote the
      // seg-major [nseg, tokens] flags; ONE scan kernel assigns each flagged
      // token its exclusive rank within its segment and builds pack_gather,
      // then the pack-push kernel gathers the rows from inputs_shard. Fused
      // kernels instead of an ATen scatter / cumsum chain: on the single
      // hardware queue the cost is the launch count, not bandwidth.
      a2av_pack_scan_impl(
          A2AVPackScanArguments{
              .pack_flag = this->a2av_pack_flag_.data_ptr<int32_t>(),
              .seg_off = seg_off_dev.data_ptr<int64_t>(),
              .pack_gather = this->a2av_pack_gather_.data_ptr<int64_t>(),
              .tokens = (int64_t)tokens_per_rank,
              .nseg = (int)nseg_c},
          pack_str);
      this->dwire_forked_ = false;
      // device wire: ONE pack-push kernel (own-node rows straight into the node peers' receive buffers +
      // their arrival signals; remote segments packed locally + pack words + announces), before GEMM 1
      DwireDispatchArgs pa = this->dwire_args(inputs_shard, par);
      if (dist_env.nnodes > 1) {
        // the remote segments (the inter-node wire's inputs) first, then the wire kernels (they wait for nothing
        // the own-node push produces), then the own-node push
        const int NN = dist_env.nnodes, nseg = dist_env.local_world_size + NN - 1;
        pa.remote_first = 1;
        pa.k_lo = 0;
        pa.k_hi = NN - 1;
        dwire_dispatch_pack_push(pa, tuning::kDwirePackBlocks, pack_str);
        this->dwire_fork_wire(inputs_shard, par, stream);
        pa.k_lo = NN - 1;
        pa.k_hi = nseg;
        dwire_dispatch_pack_push(pa, tuning::kDwirePackBlocks, pack_str);
      } else {
        dwire_dispatch_pack_push(pa, tuning::kDwirePackBlocks, pack_str);
      }
    }

    // ---- stage 2 (overlaps the wire): consumer indices, enqueued while the
    // puts fly. Everything is fixed-shape at n_copies: rows past M_this_ep are
    // in-bounds garbage the GEMM never reads (it consumes data_ptr + M_this_ep).
    torch::Tensor sorted_gather_index, sorted_scatter_index, sorted_splits_cumsum;
    {
      // fused consumer build: one cumsum + a few kernels (no ATen
      // key / argsort / index_select chain, no gating searchsorted). Stage 1
      // already wrote the per-token keep flags. A-row assignment:
      //   one node: per (expert, source) group from the offA table plus an
      //     atomic in-group rank — interior order arbitrary, the per-source
      //     tile gating (ssc) compares only group boundaries;
      //   multi-node: per (expert, LANE) group — the tile gate partitions an
      //     expert's A rows by lane via gating_cumsum, and lanes (chunk_bound
      //     windows) cut through source regions, so a source-keyed order would
      //     break the gate's invariant (a tile could read a row before it
      //     lands). Two-pass: lane histogram -> cumsum + exclusive lane
      //     offsets -> lane-keyed assignment.
      const int64_t ntokens = (int64_t)tokens_per_rank * W;
      auto mine_n = this->a2av_mine_token_.narrow(0, 0, ntokens + 1);
      CUDA_CHECK(cudaMemsetAsync(
          this->a2av_blk_cnt_.data_ptr(), 0, this->a2av_blk_cnt_.nbytes(), stream));
      auto c_excl = mine_n.cumsum(0) - mine_n;  // C[t], i64 [ntokens + 1]
      const bool tier_b = dist_env.nnodes > 1;
      int32_t *blk_cnt = this->a2av_blk_cnt_.data_ptr<int32_t>();
      A2AVConsumerBuildArguments cb_args{
          .n_copies = n_copies,
          .topk = (int)topk,
          .ep_start = (int)ep_start,
          .ep_nexperts = (int)E,
          .world_size = W,
          .e_all = e_all.data_ptr<int64_t>(),
          .s_all = s_all.data_ptr<int64_t>(),
          .flat_dst = flat_dst.data_ptr<int64_t>(),
          .not_mine = not_mine.data_ptr<bool>(),
          .c_excl = c_excl.data_ptr<int64_t>(),
          .offA = offA_dev.data_ptr<int64_t>(),
          .expert_base = expert_base_dev.data_ptr<int64_t>(),
          .blk_cnt = blk_cnt,
          .gather = this->a2av_sorted_gather_.data_ptr<int32_t>(),
          .scatter = this->a2av_sorted_scatter_.data_ptr<int32_t>(),
          // gate_q row 0 is [0, end(0), ..., end(W-1)]: skip the base
          .lane_end = tier_b ? gate_q_dev.data_ptr<int64_t>() + 1 : nullptr,
          .gate_hist = tier_b ? blk_cnt + nexG : nullptr,
          .hist_only = false,
          .offA_lane = nullptr,
          .local_world_size = (int)dist_env.local_world_size,
          .node_idx = dist_env.node_idx,
          .mm_off = nullptr,
          .mm_lo = nullptr,
          .mm_hi = nullptr,
          .mm_dst = nullptr,
          .mm_base = nullptr};
      if (this->lb_minmove_) {
        FLUX_CHECK(tier_b && mm_dev.defined());
        const int64_t NNm = dist_env.nnodes, Lm = dist_env.local_world_size;
        const int64_t P = NNm * 2 * Lm;
        int64_t *mm = mm_dev.data_ptr<int64_t>();
        cb_args.mm_off = mm;
        cb_args.mm_base = mm + NNm + 1;
        cb_args.mm_lo = cb_args.mm_base + NNm;
        cb_args.mm_hi = cb_args.mm_lo + P;
        cb_args.mm_dst = cb_args.mm_hi + P;
      }
      if (tier_b) {
        if (!this->a2av_gating_cumsum_.defined()) {
          this->a2av_gating_cumsum_ = torch::empty(
              {(int64_t)E, (int64_t)W}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }
        if (!this->a2av_offA_lane_.defined()) {
          this->a2av_offA_lane_ = torch::empty(
              {(int64_t)E * W}, torch::TensorOptions(torch::kCUDA).dtype(torch::kLong));
        }
        // pass 1: per-(expert, lane) histogram
        cb_args.hist_only = true;
        a2av_consumer_build_impl(cb_args, stream);
        // pass 2: inclusive gating cumsum + exclusive lane-keyed A offsets
        a2av_gating_cumsum_impl(
            A2AVGatingCumsumArguments{
                .ep_nexperts = (int)E,
                .world_size = W,
                .gate_hist = blk_cnt + nexG,
                .gating_cumsum = this->a2av_gating_cumsum_.data_ptr<int32_t>(),
                .offA = offA_dev.data_ptr<int64_t>(),
                .offA_lane = this->a2av_offA_lane_.data_ptr<int64_t>()},
            stream);
        // pass 3: lane-keyed row assignment (blk_cnt region is still zero —
        // pass 1 touched only gate_hist)
        cb_args.hist_only = false;
        cb_args.gate_hist = nullptr;
        cb_args.offA_lane = this->a2av_offA_lane_.data_ptr<int64_t>();
        a2av_consumer_build_impl(cb_args, stream);
      } else {
        a2av_consumer_build_impl(cb_args, stream);
      }
      sorted_gather_index = this->a2av_sorted_gather_.narrow(0, 0, n_copies);
      sorted_scatter_index = this->a2av_sorted_scatter_.narrow(0, 0, n_copies);
      sorted_splits_cumsum = ssc_dev;  // meta arena, exact LOGICAL [E, W] semantics
    }

    return A2AVDispatchState{
        sorted_gather_index, sorted_scatter_index, sorted_splits_cumsum, (int)M_this_ep};
  }

 public:
  // Routed metadata: splits / scatter_index / splits_per_source / unique_counts are derived ON
  // DEVICE from the raw replicated routing inside the timed window; the two small host-consumed
  // matrices are D2H'd into pinned staging (event-synced). The stable scatter index is
  // bit-identical to argsort(stable).argsort(): replicated cross-rank data must never come from
  // the non-deterministic calc_scatter_index.

  void
  ensure_routed_meta() {
    if (routed_meta_ready_) {
      return;
    }
    const int W = this->world_size;
    const int64_t E = this->nexperts;
    // tiles hold whole tokens of one source: W x ceil(max tokens per rank / tile tokens)
    const int32_t nblocks =
        W * a2av_meta_tiles_per_src((int64_t)this->max_ntokens / W, this->topk);
    auto dev = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt32);
    auto pin = torch::TensorOptions(torch::kCPU)
                   .dtype(torch::kInt32)
                   .pinned_memory(true);
    rt_splits_dev_ = torch::zeros({E}, dev);
    rt_scatter_dev_ = torch::zeros({(int64_t)this->max_ntokens, this->topk},
                                   dev);
    rt_sps_dev_ = torch::zeros({(int64_t)W, E}, dev);
    rt_uc_dev_ = torch::zeros({(int64_t)W, W + this->nnodes}, dev);
    rt_sps_cpu_ = torch::zeros({(int64_t)W, E}, pin);
    rt_uc_cpu_ = torch::zeros({(int64_t)W, W + this->nnodes}, pin);
    rt_block_hist_ = torch::zeros({(int64_t)nblocks, E}, dev);
    rt_block_offset_ = torch::zeros({(int64_t)nblocks, E}, dev);
    rt_uc_blk_ = torch::zeros({(int64_t)nblocks, W + this->nnodes}, dev);
    rt_expert_base_ = torch::zeros({E + 1}, dev);
    rt_dem_dev_ = torch::zeros({8}, torch::TensorOptions(torch::kCUDA).dtype(torch::kLong));
    rt_dem_cpu_ = torch::zeros(
        {8}, torch::TensorOptions(torch::kCPU).dtype(torch::kLong).pinned_memory(true));
    // dispatch plan block (sort_util.h DispatchPlan): device-built, rides the same D2H + event
    const int64_t pw = DispatchPlan::words(W / this->nnodes, this->nnodes);
    rt_plan_dev_ = torch::zeros({pw}, torch::TensorOptions(torch::kCUDA).dtype(torch::kLong));
    rt_plan_cpu_ = torch::zeros(
        {pw}, torch::TensorOptions(torch::kCPU).dtype(torch::kLong).pinned_memory(true));
    CUDA_CHECK(cudaEventCreateWithFlags(&rt_meta_event_,
                                        cudaEventDisableTiming));
    routed_meta_ready_ = true;
  }

  // the device meta arena's arguments (slice `arena`): the same in the forward and in derive_routed_meta
  A2AVMetaArenaArguments
  arena_args(char *arena) const {
    return A2AVMetaArenaArguments{
        .sps = this->rt_sps_dev_.data_ptr<int32_t>(),
        .uc = this->rt_uc_dev_.data_ptr<int32_t>(),
        .W = (int32_t)this->world_size,
        .nexperts = (int32_t)this->nexperts,
        .ep_nexperts = (int32_t)this->ep_nexperts,
        .ep_start = (int32_t)this->ep_start,
        .L = (int32_t)this->dist_env.local_world_size,
        .nnodes = (int32_t)this->dist_env.nnodes,
        .my_node = (int32_t)this->dist_env.node_idx,
        .rank = (int32_t)this->rank,
        .lb_minmove = this->lb_minmove_ ? 1 : 0,
        .recv_key = this->max_recv_ntokens_,
        .arena = arena,
        .compress_off = this->compress_meta_off_};
  }

  // the planning branches: the side streams and their fork / join events (created on first use)
  void
  ensure_plan_branches() {
    if (this->br_pass2_ != nullptr) {
      return;
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&this->br_pass2_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&this->br_arena_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->br_fork_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->br_pass2_done_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->br_arena_done_, cudaEventDisableTiming));
  }

  std::vector<torch::Tensor>
  derive_routed_meta(torch::Tensor topk_ids, std::vector<int64_t> const &caps, bool direct) {
    CHECK_INPUT(topk_ids, torch::kInt32);
    CHECK_NDIM(topk_ids, 2);
    const int W = this->world_size;
    const int64_t ntok = topk_ids.size(0);
    FLUX_CHECK_EQ(topk_ids.size(1), this->topk);
    FLUX_CHECK(ntok % W == 0);
    FLUX_CHECK_LE(ntok, this->max_ntokens);
    ensure_routed_meta();
    // the routed ids of this step: stage 1 reads the expert of copy p from them (no decode search)
    rt_topk_ids_ = topk_ids;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int64_t E = this->nexperts;
    // three launches, no memsets, no global atomics (sort_util.h)
    A2AVMetaArguments margs{
        topk_ids.data_ptr<int32_t>(),
        ntok,
        (int32_t)this->topk,
        (int32_t)E,
        (int32_t)this->ep_nexperts,
        (int32_t)W,
        (int32_t)this->nnodes,
        (int32_t)(W / this->nnodes),
        ntok / W,
        a2av_meta_tiles_per_src(ntok / W, this->topk),
        rt_block_hist_.data_ptr<int32_t>(),
        rt_block_offset_.data_ptr<int32_t>(),
        rt_uc_blk_.data_ptr<int32_t>(),
        rt_expert_base_.data_ptr<int32_t>(),
        rt_splits_dev_.data_ptr<int32_t>(),
        rt_sps_dev_.data_ptr<int32_t>(),
        rt_uc_dev_.data_ptr<int32_t>(),
        rt_scatter_dev_.data_ptr<int32_t>()};
    // planning branches (deferred-verdict step): the planning chain after the routing exchange runs as three
    // branches of the layer-step. After the meta front (pass1 + scan): branch 1 = meta pass2 (scatter_index),
    // branch 2 = the meta arena from the counts (speculative), the main stream = dispatch plan -> demands; the
    // main stream then joins branch 2 and re-runs the arena only on a degenerate step (kDead: the demands kernel
    // zeroed the counts, so the arena must be the one built after the demands), and joins branch 1. The outputs
    // equal those of the single chain; the dispatch forward skips its own arena launch for this step.
    const bool branches =
        !direct && caps.size() == 11 && this->a2av_meta_dev_.defined() && verdict_armed() != nullptr;
    this->rt_arena_early_ = false;
    if (branches) {
      ensure_plan_branches();
      a2av_meta_front_impl(margs, stream);
      CUDA_CHECK(cudaEventRecord(this->br_fork_, stream));
      CUDA_CHECK(cudaStreamWaitEvent(this->br_pass2_, this->br_fork_));
      a2av_meta_pass2_impl(margs, this->br_pass2_);
      CUDA_CHECK(cudaEventRecord(this->br_pass2_done_, this->br_pass2_));
      CUDA_CHECK(cudaStreamWaitEvent(this->br_arena_, this->br_fork_));
      a2av_meta_arena_impl(this->arena_args(reinterpret_cast<char *>(this->a2av_meta_dev_.data_ptr())),
                           this->br_arena_);
      CUDA_CHECK(cudaEventRecord(this->br_arena_done_, this->br_arena_));
    } else {
      a2av_meta_impl(margs, stream);
    }
    // exact buffer demands vs capacities on device (caps = 8 capacities +
    // relay_slots); the result rides the same D2H + event sync as sps/uc.
    // Deferred verdict (core/verdict.h): caps carries 2 more words, the MoE layer ordinal within the forward
    // and the test-abort flag; the demands kernel then sets the process's verdict block instead of the host
    // checking the demands, and makes the step degenerate (zero counts) once the block is set.
    const bool with_demands = !caps.empty();
    bool deferred = false;
    if (with_demands) {
      FLUX_CHECK(caps.size() == 9 || caps.size() == 11)
          << "derive_routed_meta: caps = 8 capacities + relay_slots [+ layer ordinal, abort flag]";
      FLUX_CHECK(caps[8] == this->relay_slots_)
          << "derive_routed_meta: relay_slots " << caps[8] << " != tuning::kRelaySlots " << this->relay_slots_;
      deferred = caps.size() == 11;
      FLUX_CHECK(!deferred || verdict_armed() != nullptr)
          << "deferred verdict step outside an open forward (verdict_begin)";
      FLUX_CHECK(!deferred || !direct) << "the deferred verdict needs the device plan block (overlap strategy)";
    }
    int64_t *verdict = deferred ? verdict_armed() : nullptr;
    // dispatch plan block on the device: every size and offset the dispatch wire of this step needs; the
    // host reads it after the event below instead of building tables
    rt_plan_fresh_ = false;
    const bool with_plan = !direct;
    auto launch_plan = [&]() {
      const int Lw = this->dist_env.local_world_size;
      FLUX_CHECK_EQ(this->dist_env.nnodes, this->nnodes);
      A2AVDispatchPlanArguments pargs{
          .sps = rt_sps_dev_.data_ptr<int32_t>(),
          .uc = rt_uc_dev_.data_ptr<int32_t>(),
          .W = (int32_t)W,
          .nexperts = (int32_t)E,
          .ep_nexperts = (int32_t)this->ep_nexperts,
          .ep_start = (int32_t)this->ep_start,
          .L = (int32_t)Lw,
          .nnodes = (int32_t)this->nnodes,
          .my_node = (int32_t)this->dist_env.node_idx,
          .rank = (int32_t)this->rank,
          .relay_slots = (int32_t)this->relay_slots_,
          .copies_per_rank = (ntok / W) * this->topk,
          .max_recv = this->max_recv_ntokens_,
          .max_stage = this->max_stage_ntokens_,
          .max_relay = this->max_relay_ntokens_,
          .plan = rt_plan_dev_.data_ptr<int64_t>(),
          .verdict = verdict};
      a2av_dispatch_plan_impl(pargs, stream);
    };
    // deferred: the plan kernel runs first, so its consistency bits are in the block the demands kernel acts on
    if (with_plan && deferred) {
      launch_plan();
    }
    if (with_demands) {
      A2AVDemandsArguments dargs{
          .sps = rt_sps_dev_.data_ptr<int32_t>(),
          .uc = rt_uc_dev_.data_ptr<int32_t>(),
          .W = (int32_t)W,
          .ep_nexperts = (int32_t)this->ep_nexperts,
          .L = (int32_t)(W / this->nnodes),
          .nnodes = (int32_t)this->nnodes,
          .relay_slots = (int32_t)caps[8],
          .direct = direct ? 1 : 0,
          .caps = {caps[0], caps[1], caps[2], caps[3], caps[4], caps[5], caps[6], caps[7]},
          .out = rt_dem_dev_.data_ptr<int64_t>()};
      if (deferred) {
        dargs.verdict = verdict;
        dargs.layer = caps[9];
        dargs.force = caps[10];
        dargs.zero_sps = rt_sps_dev_.data_ptr<int32_t>();
        dargs.zero_uc = rt_uc_dev_.data_ptr<int32_t>();
        dargs.zero_splits = rt_splits_dev_.data_ptr<int32_t>();
        dargs.zero_plan = rt_plan_dev_.data_ptr<int64_t>();
        dargs.plan_words = rt_plan_dev_.numel();
      }
      a2av_demands_impl(dargs, stream);
      if (branches) {
        // join the arena branch; on a degenerate step (the demands kernel zeroed the counts) the arena is built
        // again from the zero counts (the arena must reflect the demands' result); join the pass2 branch
        CUDA_CHECK(cudaStreamWaitEvent(stream, this->br_arena_done_));
        A2AVMetaArenaArguments fix = this->arena_args(reinterpret_cast<char *>(this->a2av_meta_dev_.data_ptr()));
        fix.only_if_dead = verdict + Verdict::kDead;
        a2av_meta_arena_impl(fix, stream);
        CUDA_CHECK(cudaStreamWaitEvent(stream, this->br_pass2_done_));
        this->rt_arena_early_ = true;
      }
      // deferred verdict + device wire: no host code reads the step's demands or plan block (the verdict is read
      // once per forward; the wire kernels read the device block), so the pinned mirrors and their copy-engine
      // copies (a bubble each inside a layer graph) are skipped
      if (!deferred) {
        CUDA_CHECK(cudaMemcpyAsync(rt_dem_cpu_.data_ptr(), rt_dem_dev_.data_ptr(),
                                   8 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
      }
    }
    if (with_plan && !deferred) {
      launch_plan();
    }
    if (with_plan) {
      if (!deferred) {
        CUDA_CHECK(cudaMemcpyAsync(rt_plan_cpu_.data_ptr(), rt_plan_dev_.data_ptr(),
                                   rt_plan_dev_.numel() * sizeof(int64_t), cudaMemcpyDeviceToHost,
                                   stream));
      }
      rt_plan_fresh_ = true;
    }
    rt_deferred_ = deferred;
    // on a deferred-verdict step the counts' mirrors are skipped too. Every host reader of sps / uc is on a
    // non-deferred path (combine: host tables under !deferred; dispatch: the pointer identity only), and no host
    // wait follows these copies on this step, so a host read would race them
    if (!deferred) {
      CUDA_CHECK(cudaMemcpyAsync(rt_sps_cpu_.data_ptr(), rt_sps_dev_.data_ptr(),
                                 (size_t)W * E * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          rt_uc_cpu_.data_ptr(), rt_uc_dev_.data_ptr(),
          (size_t)W * (W + this->nnodes) * sizeof(int32_t),
          cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaEventRecord(rt_meta_event_, stream));
    // deferred verdict + device wire: no host consumer of this step's counts remains (the verdict is read once
    // per forward, the wire kernels read the device block), so the host does not wait for the planning event
    if (!deferred) {
      CUDA_CHECK(cudaEventSynchronize(rt_meta_event_));
    }
    if (with_demands) {
      // + the device counts (consumers: combine tables, device epochs) and the demands
      return {rt_splits_dev_, rt_scatter_dev_.narrow(0, 0, ntok), rt_sps_cpu_, rt_uc_cpu_,
              rt_dem_cpu_, rt_sps_dev_, rt_uc_dev_};
    }
    return {rt_splits_dev_,
            rt_scatter_dev_.narrow(0, 0, ntok),
            rt_sps_cpu_, rt_uc_cpu_};
  }


 private:
  bool routed_meta_ready_ = false;
  torch::Tensor rt_splits_dev_, rt_scatter_dev_, rt_sps_dev_, rt_uc_dev_;
  torch::Tensor rt_sps_cpu_, rt_uc_cpu_;
  torch::Tensor rt_dem_dev_, rt_dem_cpu_;  // device demand check: 7 demands + violation mask
  torch::Tensor rt_plan_dev_, rt_plan_cpu_;  // dispatch plan block (device / pinned copy)
  torch::Tensor rt_topk_ids_;                // routed ids of the latest derive_routed_meta
  // planning branches: the latest derive_routed_meta built the step's meta arena (the forward skips its launch)
  bool rt_arena_early_ = false;
  cudaStream_t br_pass2_ = nullptr, br_arena_ = nullptr;
  cudaEvent_t br_fork_ = nullptr, br_pass2_done_ = nullptr, br_arena_done_ = nullptr;
  bool rt_plan_fresh_ = false;               // the pinned plan belongs to the latest derive_routed_meta
  // the latest derive_routed_meta was a deferred-verdict step (core/verdict.h): the forward reads no count
  // on the host (outputs sized by the caller's bound, GEMM always launched, pack bounds on the device)
  bool rt_deferred_ = false;
  torch::Tensor rt_block_hist_, rt_block_offset_, rt_uc_blk_, rt_expert_base_;
  cudaEvent_t rt_meta_event_ = nullptr;


 public:
  // swap: one-shot GEMM-START MARK. set_gemm_start_mark(e)
  // arms the next forward to write e into a device int64 flag on the forward
  // stream right before the GEMM kernel launches; the swap lane's movement
  // streams wait on it (cuStreamWaitValue64 GEQ) so the NVLink expert copies
  // start with the GEMM, not with the op's metadata/staging prologue.
  torch::Tensor gemm_mark_;  // device int64[1]
  int64_t gemm_mark_epoch_ = 0;
  bool gemm_mark_armed_ = false;
  uint64_t *gemm_mark_raw_ = nullptr;  // the mark's memory (owned by gemm_mark_)

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

  torch::Tensor
  gemm_start_mark() {
    ensure_gemm_mark();
    return this->gemm_mark_;
  }
  uint64_t
  run_id() const {
    return this->run_id_;
  }
  void
  advance_run_id(int64_t n) {
    FLUX_CHECK_GE(n, 0) << "run ids never go back";
    this->run_id_ += (uint64_t)n;
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

 protected:
  std::vector<torch::Tensor>
  forward_impl(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> input_scales,
      c10::optional<std::vector<torch::Tensor>> weight_scales,
      c10::optional<std::vector<torch::Tensor>> output_scales,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      c10::optional<torch::Tensor> splits_per_source,
      c10::optional<torch::Tensor> unique_counts,
      c10::optional<torch::Tensor> weight_signal,
      int64_t weight_signal_epoch,
      int64_t weight_gate_group_start,
      c10::optional<torch::Tensor> sched_expert_order,
      int64_t sched_n_front,
      c10::optional<torch::Tensor> weight_ptr_override,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    FLUX_CHECK(
#if TORCH_SUPPORT_FP8
        inputs_shard.scalar_type() == at::ScalarType::Float8_e5m2 ||
        inputs_shard.scalar_type() == at::ScalarType::Float8_e4m3fn ||
#endif
        inputs_shard.scalar_type() == at::ScalarType::BFloat16 ||
        inputs_shard.scalar_type() == at::ScalarType::Half)
        << inputs_shard.scalar_type();
    // Step 0. do some shape checks
    int const N = this->N;
    int const K = hidden;
    // doing shape CHECK
    CHECK_INPUT(inputs_shard, this->input_dtype);
    CHECK_NDIM(inputs_shard, 2);
    const int tokens_per_rank = inputs_shard.size(0);
    CHECK_2D(inputs_shard, tokens_per_rank, K);

    const int ntokens = tokens_per_rank * world_size;

    const std::size_t num_weights_group = weights.size();
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(weights[i], this->input_dtype);
      CHECK_3D(weights[i], this->ep_nexperts, N, K);
    }

    CHECK_INPUT(splits_gpu, torch::kInt32);
    CHECK_NDIM(splits_gpu, 1);
    FLUX_CHECK_LE(this->nexperts, splits_gpu.size(0));

    CHECK_INPUT(scatter_index, torch::kInt32);
    CHECK_2D(scatter_index, ntokens, this->topk);

    // metadata-exchange result: per-source per-expert copy counts, host-side.
    // splits[e] is its column sum; every rank holds the identical matrix.
    const int32_t *cnt_host = nullptr;
    if (splits_per_source.has_value()) {
      auto const &cnt = splits_per_source.value();
      FLUX_CHECK(cnt.device().is_cpu()) << "splits_per_source must be a CPU tensor";
      FLUX_CHECK(cnt.scalar_type() == torch::kInt32);
      FLUX_CHECK(cnt.is_contiguous());
      CHECK_2D(cnt, world_size, this->nexperts);
      cnt_host = cnt.data_ptr<int32_t>();
    }

    // compress dedup counts: cols [0, W) = u[s][d] (unique tokens s -> rank d),
    // cols [W, W + nnodes) = U[s][n] (unique tokens s -> node-n union);
    // identical on all ranks, host-side metadata (extension of the
    // splits_per_source contract — NOT derivable from cnt, depends on overlap)
    const int32_t *uc_host = nullptr;
    if (unique_counts.has_value()) {
      auto const &uc = unique_counts.value();
      FLUX_CHECK(uc.device().is_cpu()) << "unique_counts must be a CPU tensor";
      FLUX_CHECK(uc.scalar_type() == torch::kInt32);
      FLUX_CHECK(uc.is_contiguous());
      CHECK_2D(uc, world_size, world_size + this->nnodes);
      uc_host = uc.data_ptr<int32_t>();
    }
    {
      FLUX_CHECK(cnt_host != nullptr && uc_host != nullptr)
          << "compress mode requires splits_per_source and unique counts";
    }

    FLUX_CHECK(!input_scales.has_value());
    FLUX_CHECK(!weight_scales.has_value());
    if (output_scales.has_value()) {
      TORCH_CHECK_EQ(output_scales->size(), num_weights_group);
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        CHECK_INPUT(output_scales->at(i), torch::kFloat32);
        CHECK_1D(output_scales->at(i), this->ep_nexperts);
      }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Step 1: get op. and prepare op buffers
    auto meta = this->get_gemm_meta(fast_accum);
    auto rt_conf = this->get_rt_conf();
    OpRegistry::OpPtr op;
    if (hparams.has_value()) {
      op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      op = OpRegistry::instance().get_op(meta, rt_conf);
    }
    const auto tile_shape = op->get_runtime_gemm_hparams().tile_shape();
    const int tile_M = cute::get<0>(tile_shape);
    const int tile_N = cute::get<1>(tile_shape);

    // Step 2: Launch AG comm as early as possible
    bool is_s8_gemm = is_s8_torch_dtype(inputs_shard.scalar_type());
    FLUX_CHECK(!is_s8_gemm) << "not support INT8 MOE AG+Scatter yet";

    int topk = this->topk;
    int ep_nexperts = this->ep_nexperts;
    int nexperts = this->nexperts;
    int ep_start = this->ep_start;
    torch::Tensor sorted_gather_index, sorted_scatter_index, sorted_splits_cumsum;
    torch::Tensor problem_schedules_gpu;
    int num_problem_schedules = 0;
    int M_this_ep = 0;
    // the derive of this step was a deferred-verdict step (consumed here)
    const bool deferred = this->rt_deferred_;
    this->rt_deferred_ = false;
    int64_t rows_bound = 0;

    {
      FLUX_CHECK_EQ((int)num_weights_group, 1) << "a2av mode supports a single weight group";
      FLUX_CHECK(!allgather_output.has_value()) << "a2av mode has no dense gathered buffer";
      FLUX_CHECK_EQ((int)splits_gpu.size(0), nexperts) << "drop-token unsupported in a2av mode";
      // deferred-verdict step (core/verdict.h): the caller's output buffer bounds the rows this rank computes
      // (min(receive capacity, copies of the step)); the true count stays on the device
      if (deferred) {
        FLUX_CHECK(outputs_buf.has_value() && outputs_buf->size() == 1)
            << "deferred verdict: the dispatch needs the caller's (capacity-sized) output buffer";
        rows_bound = outputs_buf->at(0).size(0);
      }
      A2AVDispatchState a2av_state =
          this->a2av_dispatch(
              inputs_shard, splits_gpu, scatter_index, cnt_host, uc_host, stream, deferred, rows_bound);
      sorted_gather_index = a2av_state.sorted_gather_index;
      sorted_scatter_index = a2av_state.sorted_scatter_index;
      sorted_splits_cumsum = a2av_state.sorted_splits_cumsum;
      M_this_ep = a2av_state.M_this_ep;
      {
        // static ring schedule: the prepare kernel takes the dense branch and
        // writes ProblemSchedV2 into this buffer (bucket workspace is skipped)
        num_problem_schedules = ep_nexperts * world_size * num_weights_group;
        problem_schedules_gpu = empty_with_uninitialized_data(
            std::vector<int64_t>{num_problem_schedules * (int64_t)sizeof(ProblemSchedule)},
            torch::TensorOptions(torch::kInt8).device(torch::kCUDA));
      }
    }
    // Step 4: prepare GEMM args
    int32_t *barrier_ptr = nullptr;
    torch::Tensor input_buffer;
    {
      // rows are addressed through sorted_gather_index; signals replace the barrier
      input_buffer = this->a2av_recv_buffer;
    }

    // shapes check
    std::vector<torch::Tensor> outputs = outputs_buf.value_or([&]() {
      std::vector<torch::Tensor> outputs;
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        outputs.emplace_back(empty_with_uninitialized_data(
            std::vector<int64_t>{M_this_ep, N}, inputs_shard.options()));
      };
      return outputs;
    }());

    TORCH_CHECK_EQ(outputs.size(), num_weights_group);
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(outputs[i], this->output_dtype);
      CHECK_2D(outputs[i], M_this_ep, N);
    }

    // set the output type here accordlingly
    auto args = DispatchGemmArguments{
        .rank = rank,
        .world_size = world_size,
        .dist_env = dist_env,
        .sm_margin = sm_margin,
        .num_groups = (int)num_weights_group,
        .ep_start = ep_start,
        .ep_nexperts = ep_nexperts,
        .input = input_buffer.data_ptr(),
        .M_this_ep = M_this_ep,
        .N = N,
        .K = K,
        .splits = splits_gpu.data_ptr<int>(),
        .gather_A = sorted_gather_index.data_ptr<int32_t>(),
        .scatter_D = sorted_scatter_index.data_ptr<int32_t>(),
        .problem_schedules =
            problem_schedules_gpu.defined() ? problem_schedules_gpu.data_ptr() : nullptr,
        .num_problem_schedules = num_problem_schedules,
        .accum_per_rank_ptr =
            (this->a2av_gating_cumsum_.defined())
                ? this->a2av_gating_cumsum_.data_ptr<int32_t>()
                : sorted_splits_cumsum.data_ptr<int32_t>(),
        .tile_size_m = tile_M,
        .tile_size_n = tile_N,
        .barrier_ptr = barrier_ptr};
    // static problem schedule + window-keyed per-tile spin on the arrival signals
    args.signal_ptr = reinterpret_cast<uint64_t *>(this->a2av_signal_buffer.data_ptr());
    args.signal_expected = this->run_id_;
    // device step state (core/step_state.h): the tile gates read the step's run id from its slot
    const int step_slot = step_current_slot();
    if (step_slot >= 0) {
      args.signal_expected_ptr = step_slot_word(step_slot, StepSlot::kDispatchRun);
    }
    if (weight_signal.has_value() && weight_gate_group_start >= 0) {
      // weight-gated tiles: swap-slot problems (local group >= start) spin on their
      // slot's weight epoch signal; resident-expert problems never wait.
      FLUX_CHECK(weight_signal->is_cuda());
      FLUX_CHECK(weight_signal->scalar_type() == at::ScalarType::Long)
          << "weight_signal must be int64 (u64 epoch signals)";
      FLUX_CHECK(weight_gate_group_start > 0 && weight_gate_group_start < ep_nexperts)
          << "weight_gate_group_start " << weight_gate_group_start << " out of (0, "
          << ep_nexperts << ")";
      FLUX_CHECK(weight_signal->numel() >= ep_nexperts - weight_gate_group_start)
          << "weight_signal too small for the prefetch-slot groups";
      args.weight_signal_ptr = reinterpret_cast<uint64_t *>(weight_signal->data_ptr());
      args.weight_signal_expected = static_cast<uint64_t>(weight_signal_epoch);
      if (weight_signal_epoch < 0) {
        // the swap lane's epoch of this step lives in the step slot (device step state)
        FLUX_CHECK(step_slot >= 0) << "weight_signal_epoch < 0 needs the device step state of the step";
        args.weight_signal_expected_ptr = step_slot_word(step_slot, StepSlot::kSwapEpoch);
      }
      args.weight_gate_group_start = static_cast<int>(weight_gate_group_start);
      // Moved-last schedule: per-iteration deferred set. Host computes the
      // class+rank encoding (bit 30 = deferred / moved slot).
      if (sched_expert_order.has_value()) {
        FLUX_CHECK(sched_expert_order->is_cuda());
        FLUX_CHECK(sched_expert_order->scalar_type() == at::ScalarType::Int)
            << "sched_expert_order must be int32";
        FLUX_CHECK(sched_expert_order->is_contiguous());
        FLUX_CHECK_GE(sched_expert_order->numel(), ep_nexperts)
            << "sched_expert_order too small for ep_nexperts";
        args.sched_expert_order = sched_expert_order->data_ptr<int32_t>();
        if (sched_n_front < 0) {
          // the front-class count follows the encoding on the device (written by the swap lane's arm
          // kernel; deferred verdict: the host does not know how many slots arrive)
          FLUX_CHECK_GE(sched_expert_order->numel(), ep_nexperts + 1)
              << "sched_n_front < 0: sched_expert_order needs the front-class count at [ep_nexperts]";
          args.sched_n_front_dev = sched_expert_order->data_ptr<int32_t>() + ep_nexperts;
        } else {
          FLUX_CHECK(sched_n_front > 0 && sched_n_front <= ep_nexperts)
              << "sched_n_front " << sched_n_front << " out of (0, " << ep_nexperts << "]";
          args.sched_n_front = static_cast<int>(sched_n_front);
        }
      }
    } else {
      FLUX_CHECK(!sched_expert_order.has_value())
          << "sched_expert_order requires a weight-gated a2av forward";
    }
    if (weight_ptr_override.has_value()) {
      // swap "gate on staging": per-expert B-operand base override (device
      // int64 byte addresses, 0 = default); independent of the tile gate.
      FLUX_CHECK(weight_ptr_override->is_cuda());
      FLUX_CHECK(weight_ptr_override->scalar_type() == at::ScalarType::Long)
          << "weight_ptr_override must be int64 device addresses";
      FLUX_CHECK(weight_ptr_override->is_contiguous());
      FLUX_CHECK_GE(weight_ptr_override->numel(), ep_nexperts)
          << "weight_ptr_override too small for ep_nexperts";
      args.weight_ptr_override = weight_ptr_override->data_ptr<int64_t>();
    }
    for (int gid = 0; gid < num_weights_group; gid++) {
      args.weight[gid] = weights[gid].data_ptr();
      args.output[gid] = outputs[gid].data_ptr();
      args.scaleD[gid] =
          output_scales.has_value() ? output_scales->at(gid).data_ptr<float>() : nullptr;
    }

    // GEMM-start mark (one-shot). Written even when this rank has no rows to compute: the swap
    // lane's movement streams wait on it, and a rank that receives nothing in a step would
    // otherwise never release them (the next collective would then wait forever).
    this->write_gemm_mark(stream);
    // deferred: always launched (the host does not know whether this rank computes rows; a step without
    // rows schedules zero tiles and the kernel exits at once)
    if (M_this_ep > 0 || deferred) {
      int64_t workspace_size = op->get_workspace_size(args);
      lazy_init_buffer_tensor(&this->workspace_buffer, workspace_size);

      // Step 5: launch GEMM
      op->run(args, workspace_size ? this->workspace_buffer.data_ptr() : nullptr, stream);
    }
    this->finish_tail(stream);

    return outputs;
  }


  // the end of a dispatch step: join the step's wire into the caller's stream
  void
  finish_tail(cudaStream_t stream) {
    // device wire: its kernels were enqueued before GEMM 1; join them (every reader of a cross-rank-written
    // buffer becomes an ancestor of the end-of-combine barrier)
    if (this->dwire_forked_) {
      for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, this->dwire_done_[i], 0));
      }
    }
  }

 public:
  // the rows this rank's dispatch GEMM computes in the latest step, on the device (plan block word kM, written by
  // the planning kernels in stream order before the GEMM): the bound of the row-bounded zero / activation of a
  // deferred-verdict step
  torch::Tensor
  rows_dev() {
    FLUX_CHECK(this->rt_plan_dev_.defined()) << "dispatch rows_dev: no planning step yet";
    return this->rt_plan_dev_.narrow(0, DispatchPlan::kM, 1);
  }


  // Grow the capacity-sized symmetric panels in place (collective: every rank calls with the
  // same options on an idle device). Streams, events, signals and the run epoch are untouched,
  // so no re-priming is needed; only buffers that grow are reallocated (nvshmem_free/malloc are
  // collective, so the identical decision on every rank keeps the heap layout symmetric).
  void
  resize_capacities(DispatchOptions const &options) {
    if (options.max_recv_rows > this->max_recv_ntokens_) {
      this->a2av_recv_buffer = torch::Tensor();
      this->max_recv_ntokens_ = (int)options.max_recv_rows;
      this->a2av_recv_buffer =
          nvshmem_create_tensor({this->max_recv_ntokens_, hidden}, input_dtype);
    }
    if (nnodes > 1) {
      if (options.max_stage_rows > this->max_stage_ntokens_) {
        this->a2av_stage_buffer_ = torch::Tensor();
        this->max_stage_ntokens_ = (int)options.max_stage_rows;
        this->a2av_stage_buffer_ =
            nvshmem_create_tensor({this->max_stage_ntokens_, hidden}, input_dtype);
      }
      if (options.max_relay_rows > this->max_relay_ntokens_) {
        this->a2av_relay_stage_ = torch::Tensor();
        this->max_relay_ntokens_ = (int)options.max_relay_rows;
        this->a2av_relay_stage_ =
            nvshmem_create_tensor({this->max_relay_ntokens_, hidden}, input_dtype);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    this->dwire_build_tables();  // the recv buffer may have moved
  }

  void
  clear_buffers() {
    // the signal words (arrival signals, node arrival signals, pack announces) are deliberately NOT
    // cleared: the epoch scheme relies on monotonically increasing signal values and clearing would
    // corrupt in-flight iterations. Data buffers need no clearing (rows fully overwritten per use).
  }

  torch::Tensor
  forward(
      torch::Tensor inputs_shard,
      torch::Tensor weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      c10::optional<torch::Tensor> splits_per_source,
      c10::optional<torch::Tensor> unique_counts,
      c10::optional<torch::Tensor> weight_signal,
      int64_t weight_signal_epoch,
      int64_t weight_gate_group_start,
      c10::optional<torch::Tensor> sched_expert_order,
      int64_t sched_n_front,
      c10::optional<torch::Tensor> weight_ptr_override) {
    FLUX_CHECK(!bias.has_value());
    auto outputs = forward_impl(
        std::move(inputs_shard),
        {weights},
        std::move(splits_gpu),
        std::move(scatter_index),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_scale),
        as_optional_vec(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        std::move(splits_per_source),
        std::move(unique_counts),
        std::move(weight_signal),
        weight_signal_epoch,
        weight_gate_group_start,
        std::move(sched_expert_order),
        sched_n_front,
        std::move(weight_ptr_override),
        c10::nullopt);
    return outputs[0];
  }


};

DispatchGemmOp::DispatchGemmOp(
    std::shared_ptr<Group> tp_group,
    int ep_size,
    int nnodes,
    int max_ntokens,
    int ffn_hidden,  // before TP shard
    int hidden,
    int num_experts,
    int topk,
    at::ScalarType input_dtype,
    at::ScalarType output_dtype,
    DispatchOptions const &options)
    : impl_(new DispatchGemmOpImpl(
          tp_group,
          ep_size,
          nnodes,
          max_ntokens,
          ffn_hidden,  // before TP shard
          hidden,
          num_experts,
          topk,
          input_dtype,
          output_dtype,
          options)) {}
DispatchGemmOp::~DispatchGemmOp() { delete impl_; }

void
DispatchGemmOp::resize_capacities(DispatchOptions const &options) {
  impl_->resize_capacities(options);
}

void
DispatchGemmOp::clear_buffers() {
  FLUX_CHECK(impl_ != nullptr) << "DispatchGemmOp is not initialized";
  impl_->clear_buffers();
}
void
DispatchGemmOp::advance_run_id(int64_t n) {
  FLUX_CHECK(impl_ != nullptr) << "DispatchGemmOp is not initialized";
  impl_->advance_run_id(n);
}
uint64_t
DispatchGemmOp::run_id() const {
  FLUX_CHECK(impl_ != nullptr) << "DispatchGemmOp is not initialized";
  return impl_->run_id();
}
torch::Tensor
DispatchGemmOp::gemm_start_mark() {
  FLUX_CHECK(impl_ != nullptr) << "DispatchGemmOp is not initialized";
  return impl_->gemm_start_mark();
}
void
DispatchGemmOp::set_gemm_start_mark(int64_t epoch) {
  FLUX_CHECK(impl_ != nullptr) << "DispatchGemmOp is not initialized";
  impl_->set_gemm_start_mark(epoch);
}
torch::Tensor
DispatchGemmOp::rows_dev() {
  FLUX_CHECK(impl_ != nullptr) << "DispatchGemmOp is not initialized";
  return impl_->rows_dev();
}


std::vector<torch::Tensor>
DispatchGemmOp::derive_routed_meta(
    torch::Tensor topk_ids, std::vector<int64_t> const &caps, bool direct) {
  FLUX_CHECK(impl_ != nullptr) << "DispatchGemmOp is not initialized";
  return impl_->derive_routed_meta(std::move(topk_ids), caps, direct);
}


torch::Tensor
DispatchGemmOp::forward(
    torch::Tensor inputs_shard,
    torch::Tensor weights,
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    c10::optional<torch::Tensor> outputs_buf,
    c10::optional<torch::Tensor> allgather_output,
    bool fast_accum,
    int sm_margin,
    c10::optional<torch::Tensor> splits_per_source,
    c10::optional<torch::Tensor> unique_counts,
    c10::optional<torch::Tensor> weight_signal,
    int64_t weight_signal_epoch,
    int64_t weight_gate_group_start,
    c10::optional<torch::Tensor> sched_expert_order,
    int64_t sched_n_front,
    c10::optional<torch::Tensor> weight_ptr_override) {
  FLUX_CHECK(impl_ != nullptr) << "DispatchGemmOp is not initialized";
  return impl_->forward(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      std::move(splits_per_source),
      std::move(unique_counts),
      std::move(weight_signal),
      weight_signal_epoch,
      weight_gate_group_start,
      std::move(sched_expert_order),
      sched_n_front,
      std::move(weight_ptr_override));
}

}  // namespace bytedance::flux::ths_op
