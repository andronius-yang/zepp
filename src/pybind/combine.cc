#include "combine/ths_op/gemm_combine.h"

#include <c10/core/ScalarType.h>
#include <c10/util/intrusive_ptr.h>

#include <vector>

#include <c10/cuda/CUDAStream.h>

#include "flux/args/gemm_combine.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"
#include "flux/ths_op/util.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using CombineCls = TorchClassWrapper<GemmCombineOp>;

// Op-free entries of the combine's device plan kernels for the unit tests (tests/test_plan_device.py):
// the same kernels derive_combine_meta / forward run, on freshly allocated buffers.
// combine_plan: (block int64 [layout total], lanes int32 [2 W + 1]) from device sps [W, W * gpe] and
// uc [W, W + NN].
static std::vector<torch::Tensor>
combine_plan(torch::Tensor sps, torch::Tensor uc, int64_t W, int64_t L, int64_t gpe, int64_t rank, int64_t seq) {
  TORCH_CHECK(sps.is_cuda() && uc.is_cuda() && sps.dtype() == torch::kInt32 && uc.dtype() == torch::kInt32);
  TORCH_CHECK(sps.is_contiguous() && uc.is_contiguous() && W % L == 0 && rank >= 0 && rank < W);
  const int64_t NN = W / L;
  TORCH_CHECK(sps.numel() == W * W * gpe && uc.numel() == W * (W + NN));
  const CombinePlanLayout lay = combine_plan_layout(W, NN, gpe);
  auto block = torch::full({lay.total}, -1, torch::TensorOptions(torch::kCUDA).dtype(torch::kLong));
  auto lanes = torch::full({2 * W + 1}, -1, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt32));
  A2AVCombinePlanBlockArguments a{
      .sps = sps.data_ptr<int32_t>(),
      .uc = uc.data_ptr<int32_t>(),
      .W = (int32_t)W,
      .ep_nexperts = (int32_t)gpe,
      .L = (int32_t)L,
      .nnodes = (int32_t)NN,
      .rank = (int32_t)rank,
      .seq = seq,
      .layout = lay,
      .block = block.data_ptr<int64_t>(),
      .lanes = lanes.data_ptr<int32_t>()};
  a2av_combine_plan_block(a, c10::cuda::getCurrentCUDAStream());
  return {block, lanes};
}

// msplit problem tables + weight-gate map from a plan block's node_base (device int64 [gpe * (NN + 1)]).
// Returns (tables int32 [4 n_waves gpe + n_waves], wgate int32 [n_wgate]).
// deferred verdict (tests/test_deferred_verdict.py): gate_dev (int32 [gpe] device gate map instead of gate),
// dec (int32 [1] out: the wave-adapt decision, from remote_rows int64 [1] and the adapt constants)
static std::vector<torch::Tensor>
msplit_tables(torch::Tensor node_base, int64_t gpe, int64_t NN, std::vector<int64_t> wave_lo,
              std::vector<int64_t> wave_hi, std::vector<int64_t> gate, bool reorder, int64_t n_wgate,
              c10::optional<torch::Tensor> gate_dev, c10::optional<torch::Tensor> dec,
              c10::optional<torch::Tensor> remote_rows, int64_t adapt_reread, int64_t adapt_ratio,
              int64_t adapt_row_bytes) {
  TORCH_CHECK(node_base.is_cuda() && node_base.dtype() == torch::kLong && node_base.is_contiguous());
  TORCH_CHECK(node_base.numel() == gpe * (NN + 1) && (int64_t)gate.size() == gpe);
  TORCH_CHECK(wave_lo.size() == wave_hi.size() && (int64_t)wave_lo.size() <= kA2AVMaxNodes);
  TORCH_CHECK(gpe <= kMsplitMaxExperts);
  const int64_t nw = wave_lo.size();
  auto dev = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt32);
  auto tables = torch::full({std::max<int64_t>(4 * nw * gpe + nw, 1)}, -7, dev);
  auto wg = torch::full({std::max<int64_t>(n_wgate, 1)}, -7, dev);
  A2AVMsplitTablesArguments a{};
  a.node_base = node_base.data_ptr<int64_t>();
  a.out = nw > 0 ? tables.data_ptr<int32_t>() : nullptr;
  a.wgate = n_wgate > 0 ? wg.data_ptr<int32_t>() : nullptr;
  a.E = (int32_t)gpe;
  a.NN = (int32_t)NN;
  a.n_waves = (int32_t)nw;
  a.reorder = reorder ? 1 : 0;
  a.n_wgate = (int32_t)n_wgate;
  for (int64_t w = 0; w < nw; w++) {
    a.wave_lo[w] = (int32_t)wave_lo[w];
    a.wave_hi[w] = (int32_t)wave_hi[w];
  }
  for (int64_t e = 0; e < gpe; e++) {
    a.gate[e] = (int16_t)gate[e];
  }
  if (gate_dev.has_value()) {
    TORCH_CHECK(gate_dev->is_cuda() && gate_dev->dtype() == torch::kInt32 && gate_dev->numel() >= gpe);
    a.gate_dev = gate_dev->data_ptr<int32_t>();
  }
  if (dec.has_value()) {
    TORCH_CHECK(dec->is_cuda() && dec->dtype() == torch::kInt32 && remote_rows.has_value() &&
                remote_rows->is_cuda() && remote_rows->dtype() == torch::kLong);
    a.dec = dec->data_ptr<int32_t>();
    a.remote_rows = remote_rows->data_ptr<int64_t>();
    a.adapt_reread = adapt_reread;
    a.adapt_ratio = adapt_ratio;
    a.adapt_row_bytes = adapt_row_bytes;
  }
  a2av_msplit_tables(a, c10::cuda::getCurrentCUDAStream());
  return {tables.narrow(0, 0, 4 * nw * gpe + nw), wg.narrow(0, 0, n_wgate)};
}

// Op-free entry of the deferred-verdict scale fold (tests/test_deferred_verdict.py): input rows [0, *rows_dev)
// times scales[row] (fp32 product rounded to the element type) when fold and (*dec != 0 or no dec)
static void
fold_scales(torch::Tensor input, c10::optional<torch::Tensor> scales, torch::Tensor rows_dev,
            c10::optional<torch::Tensor> dec, bool fold, c10::optional<torch::Tensor> verdict) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 2);
  TORCH_CHECK(rows_dev.is_cuda() && rows_dev.dtype() == torch::kLong);
  TORCH_CHECK(!fold || (scales.has_value() && scales->is_cuda() && scales->dtype() == torch::kFloat32));
  a2av_fold_scales(
      A2AVFoldScalesArguments{
          .input = input.data_ptr(),
          .scales = scales.has_value() ? scales->data_ptr<float>() : nullptr,
          .rows_dev = rows_dev.data_ptr<int64_t>(),
          .dec = dec.has_value() ? dec->data_ptr<int32_t>() : nullptr,
          .fold = fold ? 1 : 0,
          .k = input.size(1),
          .rows_bound = input.size(0),
          .verdict = verdict.has_value() ? verdict->data_ptr<int64_t>() : nullptr},
      from_torch_dtype(input.scalar_type()), c10::cuda::getCurrentCUDAStream());
}

static int _register_combine [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("combine", [](py::module &m) {
    m.def("combine_plan", &combine_plan, py::arg("sps"), py::arg("uc"), py::arg("W"), py::arg("L"),
          py::arg("gpe"), py::arg("rank"), py::arg("seq") = 1);
    m.def("msplit_tables", &msplit_tables, py::arg("node_base"), py::arg("gpe"), py::arg("NN"),
          py::arg("wave_lo"), py::arg("wave_hi"), py::arg("gate"), py::arg("reorder"), py::arg("n_wgate"),
          py::arg("gate_dev") = py::none(), py::arg("dec") = py::none(), py::arg("remote_rows") = py::none(),
          py::arg("adapt_reread") = 0, py::arg("adapt_ratio") = 0, py::arg("adapt_row_bytes") = 0);
    m.def("fold_scales", &fold_scales, py::arg("input"), py::arg("scales"), py::arg("rows_dev"),
          py::arg("dec") = py::none(), py::arg("fold") = true, py::arg("verdict") = py::none());
    py::class_<CombineOptions>(m, "CombineOptions")
        .def(
            py::init([](int64_t max_send_rows, int64_t max_conv_rows, int64_t max_wire_rows) {
              return CombineOptions{max_send_rows, max_conv_rows, max_wire_rows};
            }),
            py::arg("max_send_rows"),
            py::arg("max_conv_rows"),
            py::arg("max_wire_rows"));
    py::class_<CombineCls>(m, "GemmCombineOp")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int64_t total_num_experts,
                        int64_t max_m,
                        int64_t n_dim,
                        int64_t topk,
                        at::ScalarType output_dtype,
                        int64_t ep_world_size,
                        int64_t nnodes,
                        CombineOptions const &options) {
              return new CombineCls(
                  std::make_shared<C10dProcessGroup>("", tp_group),
                  total_num_experts,
                  max_m,
                  n_dim,
                  topk,
                  output_dtype,
                  /*tp_world_size=*/1,
                  ep_world_size,
                  /*max_input_groups=*/1,
                  /*n_split=*/1,
                  options,
                  nnodes);
            }),
            py::arg("tp_group"),
            py::arg("total_num_experts"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("topk"),
            py::arg("output_dtype"),
            py::arg("ep_world_size"),
            py::arg("nnodes"),
            py::arg("options"))
        .def(
            "forward",
            &CombineCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("splits"),
            py::arg("scatter_idx"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_vec_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("with_stream_sync") = false,
            py::arg("splits_per_source") = py::none(),
            py::arg("pack_index") = py::none(),
            py::arg("reduce_index") = py::none(),
            py::arg("unique_counts") = py::none(),
            py::arg("wire_csr") = py::none(),
            py::arg("reduce_csr") = py::none())
        .def("gemm_start_mark", &CombineCls::gemm_start_mark)
        .def("set_gemm_start_mark", &CombineCls::set_gemm_start_mark, py::arg("epoch"))
        .def("set_prep_fork", &CombineCls::set_prep_fork, py::arg("event"))
        .def("resize_capacities", &CombineCls::resize_capacities, py::arg("options"))
        .def(
            "set_weight_gate",
            &CombineCls::set_weight_gate,
            py::arg("weight_signal") = py::none(),
            py::arg("weight_signal_epoch") = 0,
            py::arg("gate_of_expert") = std::vector<int64_t>{},
            py::arg("weight_ptr_override") = py::none(),
            py::arg("gate_map") = py::none())
        .def(
            "derive_combine_meta",
            &CombineCls::derive_combine_meta,
            py::arg("splits_gpu"),
            py::arg("routing_idx"),
            py::arg("splits_per_source"),
            py::arg("unique_counts") = py::none(),
            py::arg("sps_dev") = py::none(),
            py::arg("uc_dev") = py::none(),
            py::arg("routing_ids") = py::none())
        .def("meta_stream", &CombineCls::meta_stream)
        .def("run_id", &CombineCls::run_id)
        .def("advance_run_id", &CombineCls::advance_run_id, py::arg("n"));
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
