// clang-format off
// copied from CUTLASS v3.5.0 include/cutlass/gemm/kernel/gemm_grouped.h
/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Problem visitor for grouped GEMMs
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include <type_traits>

#include "cutlass/barrier.h"
#include <cuda/atomic>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                           ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,                      ///! Epilogue
  typename ThreadblockSwizzle_,            ///! Threadblock swizzling function
  GroupScheduleMode GroupScheduleMode_,    ///! Type of scheduling to perform
  bool Transposed = false
>
struct GatherRSGemmGroupedWithAbsMax {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = Transposed;

  // Optional transpose
  using MapArguments = kernel::detail::MapArguments<
    typename Mma::IteratorA::Element,
    typename Mma::IteratorA::Layout,
    Mma::kTransformA,
    Mma::IteratorA::AccessType::kElements,
    typename Mma::IteratorB::Element,
    typename Mma::IteratorB::Layout,
    Mma::kTransformB,
    Mma::IteratorB::AccessType::kElements,
    typename Mma::LayoutC,
    kTransposed
  >;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notion.
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;

  constexpr static bool kIsFp8 = std::is_same_v<ElementA, cutlass::float_e4m3_t> || std::is_same_v<ElementA, cutlass::float_e5m2_t>;

  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = GemmGroupedProblemVisitor<
                            ThreadblockShape,
                            kGroupScheduleMode,
                            kThreadCount,
                            kThreadCount,
                            kTransposed>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord *problem_sizes{nullptr};
    int problem_count{0};
    int threadblock_count{0};

    typename EpilogueOutputOp::Params output_op{};

    ElementA ** ptr_A{nullptr};
    ElementB ** ptr_B{nullptr};
    ElementC ** ptr_C{nullptr};
    ElementC ** ptr_D{nullptr};

    ///// added by flux to support FP8 begin /////
    void ** ptr_Aux{nullptr};
    void ** ptr_Vector{nullptr};
    ///// added by flux to support FP8 end /////

    typename LayoutA::Stride::LongIndex *lda{nullptr};
    typename LayoutB::Stride::LongIndex *ldb{nullptr};
    typename LayoutC::Stride::LongIndex *ldc{nullptr};
    typename LayoutC::Stride::LongIndex *ldd{nullptr};

    ///// added by flux to support FP8 begin /////
    typename LayoutC::Stride::LongIndex *ldaux{nullptr};
    typename LayoutC::Stride::LongIndex *ldr{nullptr};
    ///// added by flux to support FP8 end /////

    // Only used by device-level operator
    GemmCoord *host_problem_sizes{nullptr};

    ///// added to set barrier /////
    int split_n = 1;
    int *barrier_ptr = nullptr;
    int *non_empty_problem_count = nullptr;
    // M-split waves: per-group non-empty targets; nullptr =
    // the uniform division
    int const *non_empty_per_group = nullptr;
    // chunked combine: problem -> group map for chunk-ordered lists
    int const *prob_group_map = nullptr;
    // epilogue-fused pack: per-problem D scatter indices (identity
    // iota without the fused pack — the ScatterD iterator always reads them)
    int **scatter_D_ptr = nullptr;
    // swap: combine-side weight gate. Per-problem index
    // into weight_signal_ptr (-1 = ungated). Gated problems (this
    // iteration's swapped-in slots) spin at tile start until the slot's
    // landed epoch >= expected — the combine twin of the dispatch per-slot
    // gate. nullptr = no gate.
    int const *prob_wgate_map = nullptr;
    uint64_t const *weight_signal_ptr = nullptr;
    uint64_t weight_signal_expected = 0;
    uint64_t const *weight_signal_expected_ptr = nullptr;  // device step state: read instead (set after ctor)

    //
    // Methods
    //

    /// Default ctor
    Arguments() = default;

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord *problem_sizes,
      int problem_count,
      int threadblock_count,
      typename EpilogueOutputOp::Params output_op,
      ElementA ** ptr_A,
      ElementB ** ptr_B,
      ElementC ** ptr_C,
      ElementC ** ptr_D,
      void ** ptr_Aux,
      void ** ptr_Vector,
      typename LayoutA::Stride::LongIndex *lda,
      typename LayoutB::Stride::LongIndex *ldb,
      typename LayoutC::Stride::LongIndex *ldc,
      typename LayoutC::Stride::LongIndex *ldd,
      typename LayoutC::Stride::LongIndex *ldr,
      typename LayoutC::Stride::LongIndex *ldaux,
      GemmCoord *host_problem_sizes=nullptr,
      int split_n = 1,
      int *barrier_ptr = nullptr,
      int *non_problem_count = nullptr,
      int const *non_empty_per_group_ = nullptr,
      int **scatter_D_ptr_ = nullptr,
      int const *prob_group_map_ = nullptr,
      int const *prob_wgate_map_ = nullptr,
      uint64_t const *weight_signal_ptr_ = nullptr,
      uint64_t weight_signal_expected_ = 0
    ):
      problem_sizes(problem_sizes),
      problem_count(problem_count),
      threadblock_count(threadblock_count),
      output_op(output_op),
      ptr_A(ptr_A),
      ptr_B(ptr_B),
      ptr_C(ptr_C),
      ptr_D(ptr_D),
      ptr_Aux(ptr_Aux),
      ptr_Vector(ptr_Vector),
      lda(lda),
      ldb(ldb),
      ldc(ldc),
      ldd(ldd),
      ldaux(ldaux),
      ldr(ldr),
      host_problem_sizes(host_problem_sizes),
      split_n(split_n),
      barrier_ptr(barrier_ptr),
      non_empty_problem_count(non_problem_count),
      non_empty_per_group(non_empty_per_group_),
      scatter_D_ptr(scatter_D_ptr_),
      prob_group_map(prob_group_map_),
      prob_wgate_map(prob_wgate_map_),
      weight_signal_ptr(weight_signal_ptr_),
      weight_signal_expected(weight_signal_expected_)
    {

    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    typename ProblemVisitor::Params problem_visitor{};
    int threadblock_count{0};

    typename EpilogueOutputOp::Params output_op{};

    ElementA ** ptr_A{nullptr};
    ElementB ** ptr_B{nullptr};
    ElementC ** ptr_C{nullptr};
    ElementC ** ptr_D{nullptr};

    ///// added by flux to support FP8 begin /////
    void ** ptr_Aux{nullptr};
    void ** ptr_Vector{nullptr};
    ///// added by flux to support FP8 end /////

    typename LayoutA::Stride::LongIndex *lda{nullptr};
    typename LayoutB::Stride::LongIndex *ldb{nullptr};
    typename LayoutC::Stride::LongIndex *ldc{nullptr};
    typename LayoutC::Stride::LongIndex *ldd{nullptr};

    ///// added by flux to support FP8 begin /////
    typename LayoutC::Stride::LongIndex *ldaux{nullptr};
    typename LayoutC::Stride::LongIndex *ldr{nullptr};
    ///// added by flux to support FP8 end /////
    int n_split;
    int *non_empty_problem_count;
    int *barrier_ptr;
    int const *non_empty_per_group{nullptr};  // M-split waves; nullptr = uniform
    int **scatter_D_ptr{nullptr};             // per-problem D scatter indices
    int const *prob_group_map{nullptr};       // chunked combine
    int const *prob_wgate_map{nullptr};       // swap: per-problem weight gate idx
    uint64_t const *weight_signal_ptr{nullptr};
    uint64_t weight_signal_expected{0};
    uint64_t const *weight_signal_expected_ptr{nullptr};
    //// added by flux to support barrier ptr /////
    // Methods
    //

    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args,
          void *workspace = nullptr,
          int tile_count = 0):
      problem_visitor(args.problem_sizes, args.problem_count, workspace, tile_count),
      threadblock_count(args.threadblock_count),
      output_op(args.output_op),
      ptr_A(args.ptr_A),
      ptr_B(args.ptr_B),
      ptr_C(args.ptr_C),
      ptr_D(args.ptr_D),
      ptr_Aux(args.ptr_Aux),
      ptr_Vector(args.ptr_Vector),
      lda(args.lda),
      ldb(args.ldb),
      ldc(args.ldc),
      ldd(args.ldd),
      ldaux(args.ldaux),
      ldr(args.ldr),
      n_split(args.split_n),
      non_empty_problem_count(args.non_empty_problem_count),
      barrier_ptr(args.barrier_ptr),
      non_empty_per_group(args.non_empty_per_group),
      scatter_D_ptr(args.scatter_D_ptr),
      prob_group_map(args.prob_group_map),
      prob_wgate_map(args.prob_wgate_map),
      weight_signal_ptr(args.weight_signal_ptr),
      weight_signal_expected(args.weight_signal_expected),
      weight_signal_expected_ptr(args.weight_signal_expected_ptr)
    {

    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr,
      int tile_count = 0) {

      problem_visitor = typename ProblemVisitor::Params(args.problem_sizes, args.problem_count,
                                                        workspace, tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      lda = args.lda;
      ldb = args.ldb;
      ldc = args.ldc;
      ldd = args.ldd;

      ptr_Aux = args.ptr_Aux;
      ptr_Vector = args.ptr_Vector;
      ldaux = args.ldaux;
      ldr = args.ldr;
      n_split = args.split_n;
      non_empty_problem_count = args.problem_count;
      barrier_ptr = args.barrier_ptr;
      non_empty_problem_count = args.non_empty_problem_count;
      non_empty_per_group = args.non_empty_per_group;
      scatter_D_ptr = args.scatter_D_ptr;
      prob_group_map = args.prob_group_map;
      prob_wgate_map = args.prob_wgate_map;
      weight_signal_ptr = args.weight_signal_ptr;
      weight_signal_expected = args.weight_signal_expected;
      weight_signal_expected_ptr = args.weight_signal_expected_ptr;
    }
  };

  /// Shared memory storage structure
  struct SharedStorage {
    union {
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
    } kernel;

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  GatherRSGemmGroupedWithAbsMax() { }

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    //
    // These types shadow the type-level definitions and support the ability to implement
    // a 'transposed' GEMM that computes the transposed problems.
    //
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    //
    // Problem visitor.
    //
    ProblemVisitor problem_visitor(
      params.problem_visitor,
      shared_storage.problem_visitor,
      blockIdx.x);

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {

      GemmCoord problem_size  = problem_visitor.problem_size();
      int32_t problem_idx     = problem_visitor.problem_index();
      int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

      GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

      cutlass::gemm::GemmCoord threadblock_offset(
        int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
        int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN,
        0);

      // Load element pointers. Exchange pointers and strides if working on the transpose
      ElementA *ptr_A = reinterpret_cast<ElementA *>((kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
      typename LayoutA::LongIndex ldm_A = (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

      ElementB *ptr_B = reinterpret_cast<ElementB *>((kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
      typename LayoutB::LongIndex ldm_B = (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
        threadblock_offset.m(),
        0,
      };

      cutlass::MatrixCoord tb_offset_B{
        0,
        threadblock_offset.n()
      };

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        LayoutA(ldm_A),
        ptr_A,
        {problem_size.m(), problem_size.k()},
        thread_idx,
        tb_offset_A);

      typename Mma::IteratorB iterator_B(
        LayoutB(ldm_B),
        ptr_B,
        {problem_size.k(), problem_size.n()},
        thread_idx,
        tb_offset_B);

      typename Mma::FragmentC accumulators;

      accumulators.clear();

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = canonical_warp_idx_sync();

      int lane_idx = threadIdx.x % 32;

      //
      // Matrix multiply phase
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // swap: combine-side weight gate. A problem whose
      // expert slot is being swapped in THIS iteration spins here until the
      // movement stream raised the slot's landed epoch (host-built per-problem
      // map; the host also orders the moved experts LAST inside every combine
      // wave, so the persistent fleet drains the unmoved tiles first and this
      // spin normally finds the signal already up). Writers are NVLink CE
      // copies + cuStreamWriteValue on the movement stream -> system-scope
      // acquire; the __syncthreads below publishes to the block. Empty
      // problems schedule no tiles, so an unsignaled empty slot never hangs.
      if (params.prob_wgate_map != nullptr && threadIdx.x == 0) {
        const int wg = params.prob_wgate_map[problem_idx];
        if (wg >= 0) {
          cuda::atomic_ref<uint64_t, cuda::thread_scope_system> wsig(
              const_cast<uint64_t &>(params.weight_signal_ptr[wg]));
          const uint64_t wexpected = params.weight_signal_expected_ptr != nullptr
                                         ? *params.weight_signal_expected_ptr
                                         : params.weight_signal_expected;
          while (wsig.load(cuda::memory_order_acquire) < wexpected) {
          }
        }
      }

      // Wait for all threads to finish their epilogue phases from the previous tile.
      __syncthreads();

      // Compute threadblock-scoped matrix multiply-add
      mma(
        gemm_k_iterations,
        accumulators,
        iterator_A,
        iterator_B,
        accumulators);

      //
      // Epilogue
      //

      // EpilogueOutputOp output_op(params.output_op);

      ElementC *ptr_C = params.ptr_C[problem_idx];
      ElementC *ptr_D = params.ptr_D[problem_idx];

      LayoutC layout_C(params.ldc[problem_idx]);
      LayoutC layout_D(params.ldd[problem_idx]);

      typename Epilogue::OutputTileIterator::Params params_C(layout_C);
      typename Epilogue::OutputTileIterator::Params params_D(layout_D);

      // Tile iterator loading from source tensor (ScatterD build: indices are
      // the shared identity iota without the fused pack, the pack inverse under the
      // fused pack — ptr_C is always nullptr so C is never read).
      typename Epilogue::OutputTileIterator iterator_C(
        params_C,
        ptr_C,
        problem_size.mn(),
        thread_idx,
        threadblock_offset.mn(),
        params.scatter_D_ptr[problem_idx]
      );

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(
        params_D,
        ptr_D,
        problem_size.mn(),
        thread_idx,
        threadblock_offset.mn(),
        params.scatter_D_ptr[problem_idx]
      );

      Epilogue epilogue(
        shared_storage.kernel.epilogue,
        thread_idx,
        warp_idx,
        lane_idx);

      // clang-format on
      typename Epilogue::ElementAuxOutput * ptr_Aux =
          params.ptr_Aux != nullptr ? static_cast<typename Epilogue::ElementAuxOutput *>(params.ptr_Aux[problem_idx])
                                    : nullptr;
      typename Epilogue::ElementVector * ptr_Vector =
          params.ptr_Vector != nullptr ? static_cast<typename Epilogue::ElementVector *>(params.ptr_Vector[problem_idx])
                                       : nullptr;
      // Tile iterator writing to auxiliary tensor.
      typename Epilogue::AuxOutputTileIterator iterator_Aux(
          typename Epilogue::AuxOutputTileIterator::Params{params.ldaux[problem_idx]},
          ptr_Aux,
          problem_size.mn(),
          thread_idx,
          threadblock_offset.mn(),
          params.scatter_D_ptr[problem_idx]);

      // Move to appropriate location for this output tile
      if (ptr_Vector) {
        int threadblock_tile_offset_m = int(threadblock_idx / grid_shape.n());
        ptr_Vector += threadblock_offset.n() + threadblock_tile_offset_m * params.ldr[problem_idx];
      }
      using ElementAccumulator = typename Epilogue::OutputOp::ElementAccumulator;
      ElementAccumulator ** scale_a_ptr = (ElementAccumulator **)params.output_op.scale_a_ptr;
      ElementAccumulator ** scale_b_ptr = (ElementAccumulator **)params.output_op.scale_b_ptr;
      // copy a scale_b_ptr (if it exists)
      auto output_op_param = params.output_op; // copy an output_op param
      output_op_param.scale_a_ptr = scale_a_ptr ? scale_a_ptr[problem_idx] : nullptr;
      output_op_param.scale_b_ptr = scale_b_ptr ? scale_b_ptr[problem_idx] : nullptr;
      EpilogueOutputOp output_op(output_op_param);
      // Execute the epilogue operator to update the destination tensor.
      epilogue(output_op,
               ptr_Vector,
               iterator_D,
               accumulators,
               iterator_C,
               iterator_Aux,
               problem_size.mn(),
               threadblock_offset.mn());

      set_barrier_ptr(params, problem_idx, grid_shape.m() * grid_shape.n(), thread_idx);

      // Next tile
      problem_visitor.advance(gridDim.x);
    }
  }

  CUTLASS_DEVICE
  void set_barrier_ptr(const Params &params, int problem_idx, int problem_tile_count, int thread_idx) {
    using Barrier = cutlass::Barrier;
    __syncthreads();
    if (thread_idx == 0) {
      // barrier_ptr:
      //  [0, split_n) + offset=0 is barrier with 0/1 value. 1 for split_n group is done. 0 for init state.
      //  [0, split_n) + offset=split_n is problem counters. if problem counter reaches grid_shape, set
      //  [0, problem_count) + offset=split_n is counters. if counter reaches grid_shape, set
      //  barrier[problem_idx/problem_per_idx] to 1
      // NOTE: can't handle problems with empty tiles. each problem has at least 1 problem.
      int * tile_counter_ptr = params.barrier_ptr + cutlass::round_nearest(params.n_split, 128) * 2;
      int counter = atomicAdd(tile_counter_ptr + problem_idx, 1);
      // printf("problem: %d counter: %d/%d\n", problem_idx, counter, problem_tile_count);
      if (counter == problem_tile_count - 1) { // current problem done
        // group_idx must come from the FULL problem list (problem_idx spans
        // empty problems too -- they produce no tiles and never reach this
        // callback), while the completion target counts only NON-EMPTY
        // problems. Dividing problem_idx by the non-empty stride would
        // mis-bucket completions whenever an expert has zero rows (routing
        // often leaves experts empty), so a split flag would never fire and
        // the cascade would hang.
        int problems_per_split_full = params.problem_visitor.problem_count / params.n_split;
        // chunked combine: chunk-ordered lists ship an explicit map; the
        // division is only valid for the wave-outer/expert-inner order
        int group_idx = params.prob_group_map != nullptr
                            ? params.prob_group_map[problem_idx]
                            : problem_idx / problems_per_split_full;
        // M-split waves: a wave can lack rows for an expert
        // that is non-empty elsewhere, so the completion target must be
        // per-group. nullptr = the uniform division.
        int problem_per_split = params.non_empty_per_group != nullptr
                                    ? params.non_empty_per_group[group_idx]
                                    : (*params.non_empty_problem_count / params.n_split);
        int * problem_counter_ptr = params.barrier_ptr + cutlass::round_nearest(params.n_split, 128);
        int problem_counter = atomicAdd(problem_counter_ptr + group_idx, 1);
        // printf("tile: %d counter: %d/%d\n", group_idx, problem_counter, problem_per_split);
        if (problem_counter == problem_per_split - 1) { // current split done
          cuda::atomic_ref<int32_t, cuda::thread_scope_device>(*(params.barrier_ptr + group_idx))
              .fetch_add(1, cuda::memory_order_release);
        }
      }
    }
  }
  // clang-format off
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format on
