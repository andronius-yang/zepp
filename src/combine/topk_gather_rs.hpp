//===- topk_gather_rs.hpp --------------------------------------------- C++ ---===//
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
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "flux/args/gemm_combine.h"
#include "flux/flux.h"

namespace bytedance {
namespace flux {






// a2av_hier combine (implemented in combine_kernels.cu): persistent split-major pack
// kernel behind the GEMM cascade flags, and the per-split destination topk reduce

// ctor-time kernel preload: forces the module load of every combine kernel so
// no first-launch lazy load can deadlock behind the pre-reduce spin kernel
// (CUDA_MODULE_LOADING=LAZY; see combine_kernels.cu)
void a2av_combine_preload(DataTypeEnum dtype);

void a2av_combine_pack(
    CombinePackArguments const &args, DataTypeEnum dtype, cudaStream_t stream);



// single-node receiver (nnodes == 1): per-split top-k gather-reduce behind the wait-all gate
void a2av_combine_reduce(
    CombineReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream);

// compress (nnodes > 1): source-side pre-reduce (persistent) feeding the wire panel
void a2av_combine_prereduce(
    CombinePreReduceArguments const &args, DataTypeEnum dtype, cudaStream_t stream);

}  // namespace flux
}  // namespace bytedance
