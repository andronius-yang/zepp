//===- dispatch_gemm.h ------------------------------------------- C++ ---===//
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
#include "./comm_none.h"
#include "flux/utils.h"

namespace bytedance::flux {
constexpr int kMaxNumGroups = 2;


struct DispatchGemmArguments {
  int rank = 0;
  int world_size = 1;
  // node-aware rank ordering for the tile schedule; nnodes==1 reduces to the rotation from the own rank
  DistEnv dist_env;
  int sm_margin = 0;

  int num_groups = 0;  // make sure num_groups <= kMaxNumGroups
  int ep_start = 0;
  int ep_nexperts = 0;
  void *input = nullptr;             // before gather_A
  void *weight[kMaxNumGroups] = {};  // with groups
  void *output[kMaxNumGroups] = {};  // with groups
  // FP8 arguments
  float *scaleD[kMaxNumGroups] = {};  // with groups
  int M_this_ep = 0, N = 0, K = 0;
  int lda = 0, ldb = 0, ldc = 0, ldd = 0;
  int *splits = nullptr;
  // calculated on prepare workspace
  int32_t *gather_A = nullptr;   // on device memory expected
  int32_t *scatter_D = nullptr;  // on device memory expected
  void *problem_schedules = nullptr;
  int num_problem_schedules = 0;
  int *accum_per_rank_ptr = nullptr;  // on device memory expected
  int tile_size_m = 0, tile_size_n = 0;
  int *barrier_ptr = nullptr;
  // a2av dispatch mode (raw alltoallv): per-source-rank NVSHMEM signals replace
  // barrier_ptr, compared against the run-id epoch. nullptr == dense mode (barrier_ptr flags).
  uint64_t *signal_ptr = nullptr;
  uint64_t signal_expected = 0;
  // device step state (core/step_state.h): when set, the expected epoch is read from this word instead
  const uint64_t *signal_expected_ptr = nullptr;
  // weight-gated tiles (swap): problems whose LOCAL group
  // index is >= weight_gate_group_start (prefetch slots) additionally spin on
  // weight_signal_ptr[group - start] >= weight_signal_expected before
  // computing (the swap's per-slot epoch signals). nullptr == no
  // weight gating; local-expert problems never wait.
  uint64_t *weight_signal_ptr = nullptr;
  uint64_t weight_signal_expected = 0;
  const uint64_t *weight_signal_expected_ptr = nullptr;  // device step state, as signal_expected_ptr
  int weight_gate_group_start = INT32_MAX;
  // Moved-last schedule: per-expert schedule-order
  // encoding, device int32 [ep_nexperts]; bit 30 = deferred class, low 30
  // bits = rank within class; sched_n_front = front-class expert count.
  // The deferred class is an arbitrary per-iteration set (e.g. THIS iteration's
  // moved slots). nullptr = disabled. Bijective remap: front entries fill
  // [0, tp*groups*n_front) stage-major, deferred entries follow.
  const int32_t *sched_expert_order = nullptr;
  int sched_n_front = 0;
  // device-resident front-class count (deferred verdict: the swap lane's arm kernel writes it; the host
  // does not know how many slots arrive). nullptr = sched_n_front above.
  const int32_t *sched_n_front_dev = nullptr;
  // Swap "gate on staging": per-expert weight pointer override, device int64
  // [ep_nexperts] byte addresses. A non-zero entry replaces
  // weight[0] + eid*N*K*elem for that expert's problems, so the GEMM reads a
  // just-moved expert straight from the swap staging while the tile gate
  // above waits for the push to land. nullptr = disabled.
  const int64_t *weight_ptr_override = nullptr;
  // fill inside op. only Op has the information
  float alpha = 1.f;
  float beta = 0.f;
};

}  // namespace bytedance::flux
