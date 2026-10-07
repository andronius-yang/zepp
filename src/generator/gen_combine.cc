// Kernel-space generator: enumerates the CUTLASS grouped-GEMM instantiations that the
// op registry can select at run time.  Only the configuration this project ships is
// emitted: bf16 in/out, sm80, 108 SMs (A100).
#include "flux/flux.h"
#include "./generator_utils.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"

namespace bytedance::flux::generator {
using namespace cute;
struct GemmGroupedV2GatherRS_Space {
  static constexpr auto AllGemmMeta = make_space_gemm_meta(
      cute::make_tuple(make_gemm_dtype_config(_BF16{})),
      cute::make_tuple(_Sm80{}),
      cute::make_tuple(_A100{}),
      cute::make_tuple(_GatherRS{}),
      cute::make_tuple(_RCR{}),
      cute::make_tuple(_GemmGroupedV2{}),
      cute::make_tuple(make_gemm_v2_meta(_False{})),
      cute::make_tuple(make_gather_rs_meta(cute::Int<1>{})));
  static constexpr auto AllGemmHParams = make_space_gemm_hparams();
  static auto
  get_space() {
    return merge_gen_space({build_gen_space(AllGemmMeta, AllGemmHParams)});
  }
};
}  // namespace bytedance::flux::generator

int
main(int argc, char const **args) {
  using namespace bytedance::flux::generator;
  Options options;
  options.parse(argc, args);
  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  return main_template(
      options,
      {cute::make_tuple(
          GemmGroupedV2GatherRS_Space::get_space(),
          std::string("combine/gemm_combine_kernel.hpp"),
          std::string("GemmGroupedV2GatherRS"))});
}
