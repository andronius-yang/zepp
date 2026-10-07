
//===- workspace_helper.cu ------------------------------------- C++ ------===//
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
#include <cutlass/gemm_coord.h>
#include <cutlass/layout/matrix.h>

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/reduce_utils.cuh"
#include "workspace_helper.h"
#include "flux/cuda/kernel_registry.h"
namespace bytedance::flux {

__device__ __forceinline__ void *
ptr_with_offset(void *ptr, intptr_t offset) {
  return (void *)((char *)ptr + offset);
}

template <typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD>
__global__ void
make_workspace_kernel(
    const MoeGatherRSWorkspaceArgs args,
    int input_elem_size,
    int output_elem_size,
    void *workspace) {
  extern __shared__ char shared_storage[];
  int *ep_splits = args.splits_gpu + args.ep_start;
  int *ep_splits_acc = (int *)shared_storage;
  block_prefix_sum_and_sync(ep_splits, ep_splits_acc, args.ep_nexperts);
  int problem_per_split = args.num_groups * args.ep_nexperts;
  int problem_count =
      args.msplit ? args.n_waves * args.ep_nexperts : problem_per_split * args.N_split;
  int N = args.N, K = args.K;
  const int new_N = args.msplit ? args.N : args.N / args.N_split;
  constexpr int kAlignment = 128;

  // the offsets
  int offset_problem_sizes = 0;
  int offset_ptr_A =
      pad_to(offset_problem_sizes + problem_count * sizeof(cutlass::gemm::GemmCoord), kAlignment);
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
  int **scatter_D = (int **)((char *)workspace + offset_scatter_D);
  int *non_empty_problem_count = (int *)((char *)workspace + offset_non_empty_problem_count);

  for (int i = threadIdx.x; i < problem_count; i += blockDim.x) {
    int sid, gid, eid, Mi;
    int64_t M_acc;
    if (args.msplit) {
      // (wave, expert) destination-row sub-problem: full-N, rows are the
      // expert block's contiguous ascending home-node segment of this wave
      // (the intermediate's within-expert rows are home-node ascending — the
      // dispatch's stable scatter order; waves never cross the ring wrap).
      sid = 0;  // no column offset
      gid = 0;  // a2av_hier enforces a single weight group
      // chunk-ordered lists carry an explicit map
      eid = args.prob_eid != nullptr ? args.prob_eid[i] : i % args.ep_nexperts;
      Mi = args.wave_M[i];
      M_acc = (int64_t)(ep_splits_acc[eid] - ep_splits[eid]) + args.wave_off[i];
    } else {
      sid = i / problem_per_split;
      int sr = i % problem_per_split;
      gid = sr / args.ep_nexperts;
      eid = sr % args.ep_nexperts;
      Mi = ep_splits[eid];
      M_acc = ep_splits_acc[eid] - Mi;
    }

    problem_sizes[i] = cutlass::gemm::GemmCoord{Mi, new_N, K};
    // 64-bit byte offsets: at many experts per rank the int32
    // products overflow — eid*N*K*2 crosses 2^31 at eid > 2^31/(N*K*2)
    // (N=3584, K=3072: eid ~97). ptr_A/ptr_D M_acc terms are the same class
    // at large M.
    ptr_A[i] = ptr_with_offset(args.input[gid], (int64_t)M_acc * K * input_elem_size);
    {
      void *wb = ptr_with_offset(args.weights[gid], (int64_t)eid * N * K * input_elem_size);
      if (args.weight_ptr_override != nullptr && gid == 0) {
        // swap staging read: the moved expert's weights live in the staging slab
        const int64_t o = args.weight_ptr_override[eid];
        if (o != 0) wb = reinterpret_cast<void *>(o);
      }
      ptr_B[i] = ptr_with_offset(wb, (int64_t)sid * new_N * K * input_elem_size);
    }
    ptr_C[i] = nullptr;
    if (args.fused_pack && (args.msplit_dec == nullptr || *args.msplit_dec != 0)) {
      // fused pack: D goes straight to the dest-major send panel through the pack
      // inverse (absolute panel rows; ldd stays N == panel row stride)
      ptr_D[i] = args.send_panel;
      scatter_D[i] = args.inv_pack + M_acc;
    } else {
      ptr_D[i] = ptr_with_offset(
          args.output[gid], ((int64_t)M_acc * N + (int64_t)sid * new_N) * output_elem_size);
      scatter_D[i] = args.iota;  // relative identity: D keeps its M_acc base
    }
    lda[i] = LayoutA::packed({(int)Mi, (int)K}).stride(0);
    ldb[i] = LayoutB::packed({(int)K, (int)N}).stride(0);
    ldc[i] = LayoutC::packed({(int)Mi, (int)N}).stride(0);
    ldd[i] = LayoutD::packed({(int)Mi, (int)N}).stride(0);
    ldr[i] = 0;

    scale_A[i] = args.input_scales[gid] == nullptr ? nullptr : args.input_scales[gid];
    scale_B[i] = args.weight_scales[gid] == nullptr ? nullptr : args.weight_scales[gid] + eid;
  }
  __syncthreads();
  if (args.msplit) {
    // Zero-target waves have no non-empty problem to fire the cascade: preset
    // their ready flags so the pack kernel's wave gate never waits on them.
    // (The GEMM's per-group targets come from non_empty_per_wave; the
    // per-split *non_empty_problem_count is never read in msplit mode.)
    const int nf = args.n_flags > 0 ? args.n_flags : args.n_waves;
    for (int w = threadIdx.x; w < nf; w += blockDim.x) {
      if (args.non_empty_per_wave[w] == 0) {
        args.barrier[w] = 1;
      }
    }
    return;
  }
  int *non_empty_splits = (int *)shared_storage;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *non_empty_splits = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < args.ep_nexperts; i += blockDim.x) {
    atomicAdd(non_empty_splits, ep_splits[i] != 0);
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *non_empty_problem_count = *non_empty_splits * args.N_split * args.num_groups;
  }
  if (args.preset_empty && args.barrier != nullptr && *non_empty_splits == 0) {
    // no tile, no cascade: the pack's split gates would never open
    for (int sid = threadIdx.x; sid < args.N_split; sid += blockDim.x) {
      args.barrier[sid] = 1;
    }
  }
}
void
make_workspace(
    const MoeGatherRSWorkspaceArgs &args,
    GemmLayoutEnum layout,
    int input_elem_size,
    int output_elem_size,
    void *workspace,
    cudaStream_t stream) {
  FLUX_CHECK_EQ(layout, GemmLayoutEnum::RCR);
  int shared_mem_size = std::max(
      sizeof(int),                      // non_empty_problem_count
      sizeof(int) * args.ep_nexperts);  // splits_acc
  make_workspace_kernel<
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor>
      <<<1, 768, shared_mem_size, stream>>>(args, input_elem_size, output_elem_size, workspace);
  CUDA_CHECK(cudaGetLastError());
}

// kernel registry (include/flux/cuda/kernel_registry.h): every kernel of this file
void
combine_workspace_kernels(KernelList &out) {
  out.push_back(ZEPP_KERNEL_ENTRY_T(
      "make_workspace_kernel<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, "
      "cutlass::layout::RowMajor, cutlass::layout::RowMajor>",
      make_workspace_kernel<
          cutlass::layout::RowMajor,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor,
          cutlass::layout::RowMajor>));
}
}  // namespace bytedance::flux
