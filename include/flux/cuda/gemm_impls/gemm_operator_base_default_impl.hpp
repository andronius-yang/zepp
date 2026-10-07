//===- gemm_operator_base_default_impl.hpp ------------------------ C++ ---===//
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
#include "cutlass/fast_math.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_operator_base.h"
#include "cutlass/workspace.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/device_kernel.h"
#include <cxxabi.h>
#include <cstdlib>
#include <string>
#include <typeinfo>

namespace bytedance::flux {

// Usage:
//   let a gemm impl class to derive from this in CRTP form.
//   so that the impl class is derived from GemmOperatorBase and
//   with default implementation of member functions
template <class DerivedImpl>
struct GemmOperatorBaseDefaultImplMixin : public GemmOperatorBase {
 public:
  // meta programming checkers
  template <class T, typename = std::void_t<>>
  struct has_hparams : std::false_type {};

  template <class T>
  struct has_hparams<T, std::void_t<decltype(T::hparams)>>
      : detail::is_gemm_hparams<decay_and_strip_t<decltype(T::hparams)>> {};

  template <class T, typename = std::void_t<>>
  struct is_cutlass3_gemm_universal_adapter : std::false_type {};

  template <class T>
  struct is_cutlass3_gemm_universal_adapter<
      T,
      std::enable_if_t<std::is_same_v<
          T,
          cutlass::gemm::device::GemmUniversalAdapter<
              typename T::GemmKernel,
              cute::enable_if_t<
                  cutlass::gemm::detail::IsCutlass3GemmKernel<typename T::GemmKernel>::value>>>>>
      : std::true_type {};

 private:
  std::any gemm_op_;

 public:
  // FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmOperatorBaseDefaultImplMixin)
  GemmOperatorBaseDefaultImplMixin() {
    using Gemm = identity_t<decltype(derived()->gemm_device())>;
    gemm_op_ = Gemm{};
  }

  ~GemmOperatorBaseDefaultImplMixin() override = default;

  DerivedImpl *
  derived() {
    return static_cast<DerivedImpl *>(this);
  }
  DerivedImpl const *
  derived() const {
    return static_cast<DerivedImpl const *>(this);
  }

  void
  initialize(std::any const &args, void *workspace = nullptr, void *stream = nullptr) override {
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(workspace);
    std::size_t workspace_offset = 0;

    void *args_workspace = workspace_ptr + workspace_offset;
    workspace_offset += this->get_args_workspace_size(args);
    workspace_offset = cutlass::round_nearest(workspace_offset, cutlass::MinWorkspaceAlignment);
    this->initialize_args_workspace(args, args_workspace, stream);

    using Gemm = identity_t<decltype(derived()->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    GemmArguments const &gemm_args =
        static_cast<DerivedImpl *>(this)->to_gemm_args(args, args_workspace);
    CUTLASS_CHECK(Gemm::can_implement(gemm_args));
    auto cu_stream = static_cast<cudaStream_t>(stream);
    void *gemm_workspace = workspace_ptr + workspace_offset;
    Gemm &gemm_op = std::any_cast<Gemm &>(this->gemm_op_);
    CUTLASS_CHECK(gemm_op.initialize(gemm_args, gemm_workspace, cu_stream));
  }

  void
  run(std::any const &args,
      void *workspace = nullptr,
      void *stream = nullptr,
      bool launch_with_pdl = false) override {
    this->initialize(args, workspace, stream);
    this->run(stream, launch_with_pdl);
  }

  void
  run(void *stream = nullptr, bool launch_with_pdl = false) override {
    using Gemm = identity_t<decltype(derived()->gemm_device())>;
    auto cu_stream = static_cast<cudaStream_t>(stream);
    Gemm &gemm_op = std::any_cast<Gemm &>(this->gemm_op_);
    if constexpr (is_cutlass3_gemm_universal_adapter<Gemm>::value) {
      CUTLASS_CHECK(gemm_op.run(cu_stream, /*cuda_adapter=*/nullptr, launch_with_pdl));
    } else {
      CUTLASS_ASSERT(launch_with_pdl == false);
      CUTLASS_CHECK(gemm_op.run(cu_stream));
    }
  }

  std::size_t
  get_workspace_size(std::any const &args) const override {
    std::size_t workspace_size = 0;
    workspace_size += this->get_args_workspace_size(args);
    workspace_size = cutlass::round_nearest(workspace_size, cutlass::MinWorkspaceAlignment);
    using Gemm = identity_t<decltype(derived()->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    const GemmArguments &gemm_args =
        static_cast<DerivedImpl const *>(this)->to_gemm_args(args, nullptr);
    workspace_size += Gemm::get_workspace_size(gemm_args);
    workspace_size = cutlass::round_nearest(workspace_size, cutlass::MinWorkspaceAlignment);
    return workspace_size;
  }

  std::size_t
  get_barrier_workspace_size(std::any const &) const override {
    return 0;
  }

  UnifiedGemmHParams
  get_runtime_gemm_hparams() const override {
    static_assert(has_hparams<DerivedImpl>::value);
    return unify_type(DerivedImpl::hparams);
  }

  // kernel registry: the kernel the CUTLASS device adapter launches (cutlass::Kernel for the 2.x
  // grouped GEMMs, cutlass::Kernel2 for the 2.x universal GEMM, cutlass::device_kernel for 3.x)
  std::string
  kernel_name() const override {
    using Gemm = identity_t<decltype(derived()->gemm_device())>;
    if constexpr (has_base_kernel<Gemm>::value) {
      return "Kernel<" + demangled_type_name<typename Gemm::BaseKernel>() + ">";
    } else if constexpr (is_cutlass3_gemm_universal_adapter<Gemm>::value) {
      return "device_kernel<" + demangled_type_name<typename Gemm::GemmKernel>() + ">";
    } else {
      return "Kernel2<" + demangled_type_name<typename Gemm::GemmKernel>() + ">";
    }
  }

  const void *
  kernel_func() const override {
    using Gemm = identity_t<decltype(derived()->gemm_device())>;
    if constexpr (has_base_kernel<Gemm>::value) {
      return reinterpret_cast<const void *>(&cutlass::Kernel<typename Gemm::BaseKernel>);
    } else if constexpr (is_cutlass3_gemm_universal_adapter<Gemm>::value) {
      return reinterpret_cast<const void *>(&cutlass::device_kernel<typename Gemm::GemmKernel>);
    } else {
      return reinterpret_cast<const void *>(&cutlass::Kernel2<typename Gemm::GemmKernel>);
    }
  }

 private:
  template <class T, typename = std::void_t<>>
  struct has_base_kernel : std::false_type {};

  template <class T>
  struct has_base_kernel<T, std::void_t<typename T::BaseKernel>> : std::true_type {};

  template <class T>
  static std::string
  demangled_type_name() {
    int status = 0;
    char *name = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
    std::string out = (status == 0 && name != nullptr) ? std::string(name) : std::string(typeid(T).name());
    std::free(name);
    return out;
  }
};

}  // namespace bytedance::flux
