//===- gemm_combine.h -------------------------------- C++ ---===//
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
#include <c10/core/ScalarType.h>
#include <torch/all.h>

#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"

namespace bytedance::flux::ths_op {
// Buffer capacities (rows) of the combine op, sized by the caller from the routing's provable
// bounds (zepp.routing.compute_capacities).
struct CombineOptions {
  int64_t max_send_rows;  // send panel rows (rows this rank computes)
  int64_t max_conv_rows;  // source-side convergence panel rows
  int64_t max_wire_rows;  // wire panel rows (one pre-reduced partial per (token, owner node))
};

class CombineWire {
 public:
  CombineWire(
      std::shared_ptr<Group> tp_group_,
      int max_m,
      int n_dim,
      int topk,
      at::ScalarType output_dtype,
      int ep_nexperts,
      int ep_world_size,
      std::vector<torch::Tensor> barriers,
      int n_split,
      CombineOptions const &options,
      int nnodes = 1,
      bool a2av_compress = false);
  ~CombineWire();
  void reset_buffer();
  // grow the capacity-sized symmetric panels in place (collective; idle device)
  void advance_run_id(int64_t n);
  uint64_t run_id() const;  // run id of the latest step (device step state sync)
  void resize_capacities(CombineOptions const &options);
  // M-split waves: arm the combine's
  // per-schedule-step wave gates and destination order for the NEXT run().
  // node_order[i] = i-th dest node in production order (ring or size-sorted);
  // wave_of_node[i] = its cascade flag. Per-iteration state; n_waves == 0
  // disarms (one gate per split + ring order).
  // the event recorded after the combine GEMM; the remote-lane receivers of the next run() wait for it and run with
  // the wide grid
  void set_tail_reduce(cudaEvent_t gemm_end);
  void set_msplit_waves(
      std::vector<int> const &wave_of_node,
      std::vector<int> const &node_order,
      int n_waves,
      int n_chunk_flags = 0);
  // arm the combine plan block of the NEXT run(): the count-derived tables are read from host_block
  // (the host copy of the device plan block, complete when this is called; nullptr on a deferred-verdict
  // step of the device wire) and the bucket lane table from lanes_dev
  void set_plan_block(int64_t const *host_block, int32_t const *lanes_dev);
  // deferred verdict (core/verdict.h) for the NEXT run(), one-shot: outside the wire the combine reads the
  // device plan block (pack rows, pre-reduce segments), the device wave-adapt decision (nullptr: no waves)
  // and the verdict block (degenerate receivers) instead of host counts
  void set_deferred(int64_t const *plan_dev, int32_t const *msplit_dec, int64_t const *verdict);
  // capacity rows of the send / convergence / wire panels
  void panel_rows(int64_t *send, int64_t *conv, int64_t *wire);
  // epilogue-fused pack: the send panel the GEMM scatters into
  void *send_panel_ptr();
  int64_t send_panel_rows();
  torch::Tensor run(
      std::vector<torch::Tensor> gemm_outs,  // of group_size
      c10::optional<torch::Tensor> output_,
      int ep_start,
      int ep_nexperts,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> output_vec_scales,
      int num_thread_blocks,
      intptr_t cp_stream,
      // a2av_hier mode only: the [W, nexperts] splits_per_source metadata (int32
      // CPU) and the mirror-layout pack/reduce gather indices (int32 CUDA).
      // Compress adds the transposed-U dedup counts and the wire/reduce CSRs.
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> pack_index = c10::nullopt,
      c10::optional<torch::Tensor> reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> reduce_csr = c10::nullopt);

 private:
  class CombineWireImpl;
  CombineWireImpl *impl_;
};

class GemmCombineOp {
 public:
  GemmCombineOp(
      std::shared_ptr<Group> tp_group_,
      int64_t total_num_experts,
      int64_t max_m,
      int64_t n_dim,
      int64_t topk,
      at::ScalarType output_dtype,
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t max_input_groups,
      int64_t n_split,
      CombineOptions const &options,
      int64_t nnodes = 1);
  ~GemmCombineOp();
  void resize_capacities(CombineOptions const &options);
  torch::Tensor forward(
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
      // a2av_hier mode only: splits_per_source is REQUIRED ([W, nexperts] int32
      // CPU); the index tensors are optional precomputed routing-plan inputs (a
      // caller that derives them up front passes them, paying the index math once).
      // Compress (dedup) plan: unique_counts is the transposed-U dedup count
      // matrix ([W, nnodes] int32 CPU, required whenever compress is on);
      // wire_csr = [wire_ptr, wire_copy] and reduce_csr =
      // [red_ptr, red_row] are the precomputed compress CSRs
      // (all-or-none as a pair; required when compress is on, see derive_combine_meta).
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      c10::optional<torch::Tensor> pack_index = c10::nullopt,
      c10::optional<torch::Tensor> reduce_index = c10::nullopt,
      c10::optional<torch::Tensor> unique_counts = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> wire_csr = c10::nullopt,
      c10::optional<std::vector<torch::Tensor>> reduce_csr = c10::nullopt);
  //
  // derive_combine_meta: returns [pack_index, reduce_index] for a2av_hier, plus
  // [wire_ptr, wire_copy, red_ptr, red_row] when compress. splits_gpu int32
  // CUDA [nexperts], routing_idx int32 CUDA [m_full], splits_per_source int32
  // CPU [W, nexperts]; unique_counts int32 CPU [W, nnodes] (compress).
  std::vector<torch::Tensor> derive_combine_meta(
      torch::Tensor splits_gpu,
      torch::Tensor routing_idx,
      torch::Tensor splits_per_source,
      c10::optional<torch::Tensor> unique_counts = c10::nullopt,
      // (required) the device sps [W, nex] / uc [W, W + NN] int32 counts
      c10::optional<torch::Tensor> sps_dev = c10::nullopt,
      c10::optional<torch::Tensor> uc_dev = c10::nullopt,
      // (required) int32 CUDA [m_full], the expert id of every copy (the routing the scatter index is
      // the stable sort of)
      c10::optional<torch::Tensor> routing_ids = c10::nullopt);
  // the side stream the caller runs derive_combine_meta on when it overlaps the dispatch GEMM
  // (created by the op, non-blocking, never shared with another component); a cudaStream_t
  int64_t meta_stream();
  uint64_t run_id() const;  // the wire's run id of the latest step
  void advance_run_id(int64_t n);  // host shadow of a step that ran as a replayed graph
  // swap: arm the combine-side weight gate for the NEXT
  // forward (one-shot). weight_signal = int64 CUDA per-slot landed
  // epochs; gate_of_expert[e] = index into it (-1 = ungated). Gated experts'
  // problems are ordered last inside every combine wave and spin at tile
  // start until signal >= epoch. weight_signal = None disarms.
  // weight_ptr_override (swap "gate on staging"): int64 CUDA [ep_nexperts]
  // per-expert weight base override (byte addresses, 0 = default), one-shot.
  // gate_map (deferred verdict): int32 CUDA [ep_nexperts], the gate index of every local expert on the
  // device (-1 ungated) instead of gate_of_expert; the gate is then armed whatever the map holds.
  // set_prep_fork: the next forward builds its msplit tables and pack inverse on a side stream forked from
  // this event (a cudaEvent_t recorded after the combine metadata derive; one-shot)
  void set_prep_fork(int64_t event);
  void set_weight_gate(
      c10::optional<torch::Tensor> weight_signal,
      int64_t weight_signal_epoch,
      std::vector<int64_t> gate_of_expert,
      c10::optional<torch::Tensor> weight_ptr_override = c10::nullopt,
      c10::optional<torch::Tensor> gate_map = c10::nullopt);
  // swap: one-shot GEMM-start mark (device int64[1] written with
  // `epoch` on the forward stream right before the combine GEMM launches)
  torch::Tensor gemm_start_mark();
  void set_gemm_start_mark(int64_t epoch);

 private:
  class GemmCombineOpImpl;
  GemmCombineOpImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
