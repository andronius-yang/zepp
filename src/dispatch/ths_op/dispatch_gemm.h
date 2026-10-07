//===- dispatch_gemm.h ------------------------------- C++ ---===//
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

// Buffer capacities (rows) of the dispatch op, sized by the caller from the routing's provable
// bounds (zepp.routing.compute_capacities).
struct DispatchOptions {
  int64_t max_recv_rows;   // rows this rank may receive (node-union recv regions)
  int64_t max_stage_rows;  // gateway staging rows for inbound node payloads
  int64_t max_relay_rows;  // per-round relay staging rows (two-slot double buffer)
};

class DispatchGemmOp {
 public:
  DispatchGemmOp(
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
      DispatchOptions const &options);
  ~DispatchGemmOp();
  void clear_buffers();
  // grow the capacity-sized symmetric panels in place (collective; idle device)
  void resize_capacities(DispatchOptions const &options);
  // swap: one-shot GEMM-start mark (device int64[1]
  // written with `epoch` on the forward stream right before the GEMM launches)
  torch::Tensor gemm_start_mark();
  uint64_t run_id() const;  // run id of the latest step (device step state sync)
  void advance_run_id(int64_t n);  // host shadow of a step that ran as a replayed graph
  void set_gemm_start_mark(int64_t epoch);
  // device int64 [1]: the rows the dispatch GEMM computes in the latest step (deferred-verdict bound)
  torch::Tensor rows_dev();
  torch::Tensor forward(
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
      // metadata-exchange result: int32 CPU [world_size, nexperts] per-source
      // per-expert copy counts; splits is its column sum. Required (with
      // unique_counts).
      c10::optional<torch::Tensor> splits_per_source = c10::nullopt,
      // a2av_hier_compress metadata: int32 CPU [world_size, world_size + nnodes]
      // dedup counts — cols [0, W) = unique tokens source s -> rank d, cols
      // [W, W + nnodes) = unique tokens source s -> node-n union. Identical on
      // all ranks; required (with splits_per_source) in compress mode.
      c10::optional<torch::Tensor> unique_counts = c10::nullopt,
      // weight-gated tiles (swap): CUDA u64/int64 epoch
      // signals (>= n_slots elements, e.g. the swap's slot signals);
      // problems with local group >= weight_gate_group_start spin on
      // weight_signal[group - start] >= weight_signal_epoch. a2av static-
      // schedule modes only. nullopt/-1 = no weight gating.
      c10::optional<torch::Tensor> weight_signal = c10::nullopt,
      int64_t weight_signal_epoch = 0,
      int64_t weight_gate_group_start = -1,
      // moved-last schedule: int32 CUDA [ep_nexperts]
      // class+rank encoding (bit 30 = deferred class), + front-class
      // count. Weight-gated a2av static-schedule forwards only.
      c10::optional<torch::Tensor> sched_expert_order = c10::nullopt,
      int64_t sched_n_front = 0,
      // swap "gate on staging": int64 CUDA [ep_nexperts] per-expert weight
      // base override (byte addresses, 0 = default weights + eid*stride).
      c10::optional<torch::Tensor> weight_ptr_override = c10::nullopt);
  // Routed metadata of a step, derived on the device from topk_ids: {splits_dev,
  // scatter_index_dev, sps_cpu, uc_cpu}. caps (device demand check): 8 capacities
  // [recv_cap, dispatch_recv, dispatch_stage, dispatch_relay, combine_send, combine_conv,
  // combine_wire, pair_cap] + relay_slots [+ MoE layer ordinal, abort flag: a deferred-verdict
  // step]; when given, a 5th tensor (pinned int64[8]: 7 demands + violation mask) and the
  // device sps [W, nexperts] / uc [W, W + nnodes] int32 counts are appended. direct: no
  // dispatch plan block (direct strategy).
  std::vector<torch::Tensor> derive_routed_meta(
      torch::Tensor topk_ids, std::vector<int64_t> const &caps = {}, bool direct = false);

 private:
  class DispatchGemmOpImpl;
  DispatchGemmOpImpl *impl_ = nullptr;
};

}  // namespace bytedance::flux::ths_op
