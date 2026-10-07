#include "dispatch/ths_op/dispatch_gemm.h"
#include "dispatch/sort_util.h"
#include "core/verdict.h"

#include <ATen/cuda/CUDAContext.h>

#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using DispatchCls = TorchClassWrapper<DispatchGemmOp>;

// Op-free entry points of the device metadata chain for the unit tests
// (tests/test_meta_device.py): the same kernels the dispatch op runs in
// derive_routed_meta, on freshly allocated scratch.
static std::vector<torch::Tensor>
routed_meta(torch::Tensor topk_ids, int64_t W, int64_t L, int64_t gpe) {
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.dtype() == torch::kInt32 && topk_ids.dim() == 2 &&
              topk_ids.is_contiguous());
  const int64_t ntok = topk_ids.size(0), topk = topk_ids.size(1), E = W * gpe, NN = W / L;
  TORCH_CHECK(ntok % W == 0 && W % L == 0);
  const int64_t tpr = ntok / W;
  const int32_t tps = a2av_meta_tiles_per_src(tpr, (int32_t)topk);
  const int64_t nb = W * tps;
  auto dev = torch::TensorOptions(torch::kCUDA).dtype(torch::kInt32);
  auto block_hist = torch::empty({nb, E}, dev), block_offset = torch::empty({nb, E}, dev);
  auto uc_blk = torch::empty({nb, W + NN}, dev), expert_base = torch::empty({E + 1}, dev);
  auto splits = torch::empty({E}, dev), sps = torch::empty({W, E}, dev);
  auto uc = torch::empty({W, W + NN}, dev), scatter = torch::empty({ntok, topk}, dev);
  A2AVMetaArguments a{topk_ids.data_ptr<int32_t>(), ntok, (int32_t)topk, (int32_t)E, (int32_t)gpe,
                      (int32_t)W, (int32_t)NN, (int32_t)L, tpr, tps,
                      block_hist.data_ptr<int32_t>(), block_offset.data_ptr<int32_t>(),
                      uc_blk.data_ptr<int32_t>(), expert_base.data_ptr<int32_t>(),
                      splits.data_ptr<int32_t>(), sps.data_ptr<int32_t>(), uc.data_ptr<int32_t>(),
                      scatter.data_ptr<int32_t>()};
  a2av_meta_impl(a, c10::cuda::getCurrentCUDAStream());
  return {splits, scatter, sps, uc};
}

// verdict (deferred capacity verdict, core/verdict.h; tests/test_deferred_verdict.py): an int64 block of
// Verdict::kWords words the kernel ORs into; it then zeroes sps / uc / splits / plan in place once the block is set
static torch::Tensor
a2av_demands(torch::Tensor sps, torch::Tensor uc, int64_t W, int64_t L, int64_t gpe, int64_t relay_slots,
             bool direct, std::vector<int64_t> caps, c10::optional<torch::Tensor> verdict, int64_t layer,
             bool force, c10::optional<torch::Tensor> splits, c10::optional<torch::Tensor> plan) {
  TORCH_CHECK(sps.is_cuda() && uc.is_cuda() && sps.dtype() == torch::kInt32 && uc.dtype() == torch::kInt32);
  TORCH_CHECK(caps.size() == 8, "a2av_demands: 8 capacities");
  auto out = torch::zeros({8}, torch::TensorOptions(torch::kCUDA).dtype(torch::kLong));
  A2AVDemandsArguments d{
      .sps = sps.data_ptr<int32_t>(),
      .uc = uc.data_ptr<int32_t>(),
      .W = (int32_t)W,
      .ep_nexperts = (int32_t)gpe,
      .L = (int32_t)L,
      .nnodes = (int32_t)(W / L),
      .relay_slots = (int32_t)relay_slots,
      .direct = direct ? 1 : 0,
      .caps = {caps[0], caps[1], caps[2], caps[3], caps[4], caps[5], caps[6], caps[7]},
      .out = out.data_ptr<int64_t>()};
  if (verdict.has_value()) {
    TORCH_CHECK(verdict->is_cuda() && verdict->dtype() == torch::kLong && verdict->numel() >= Verdict::kWords);
    TORCH_CHECK(splits.has_value() && splits->is_cuda() && splits->dtype() == torch::kInt32 &&
                splits->numel() == W * gpe);
    TORCH_CHECK(sps.is_contiguous() && uc.is_contiguous());
    d.verdict = verdict->data_ptr<int64_t>();
    d.layer = layer;
    d.force = force ? 1 : 0;
    d.zero_sps = sps.data_ptr<int32_t>();
    d.zero_uc = uc.data_ptr<int32_t>();
    d.zero_splits = splits->data_ptr<int32_t>();
    if (plan.has_value()) {
      TORCH_CHECK(plan->is_cuda() && plan->dtype() == torch::kLong && plan->is_contiguous());
      d.zero_plan = plan->data_ptr<int64_t>();
      d.plan_words = plan->numel();
    }
  }
  a2av_demands_impl(d, c10::cuda::getCurrentCUDAStream());
  return out;
}

// Op-free entry point of the dispatch plan kernel (tests/test_plan_device.py): the block one rank's
// dispatch wire reads (sort_util.h DispatchPlan), from device sps / uc.
static torch::Tensor
a2av_dispatch_plan(torch::Tensor sps, torch::Tensor uc, int64_t W, int64_t L, int64_t gpe, int64_t rank,
                   int64_t relay_slots, int64_t copies_per_rank, int64_t max_recv, int64_t max_stage,
                   int64_t max_relay) {
  TORCH_CHECK(sps.is_cuda() && uc.is_cuda() && sps.dtype() == torch::kInt32 && uc.dtype() == torch::kInt32);
  TORCH_CHECK(W % L == 0);
  const int64_t NN = W / L;
  auto plan = torch::empty({DispatchPlan::words((int)L, (int)NN)},
                           torch::TensorOptions(torch::kCUDA).dtype(torch::kLong));
  A2AVDispatchPlanArguments a{
      .sps = sps.data_ptr<int32_t>(),
      .uc = uc.data_ptr<int32_t>(),
      .W = (int32_t)W,
      .nexperts = (int32_t)(W * gpe),
      .ep_nexperts = (int32_t)gpe,
      .ep_start = (int32_t)(rank * gpe),
      .L = (int32_t)L,
      .nnodes = (int32_t)NN,
      .my_node = (int32_t)(rank / L),
      .rank = (int32_t)rank,
      .relay_slots = (int32_t)relay_slots,
      .copies_per_rank = copies_per_rank,
      .max_recv = max_recv,
      .max_stage = max_stage,
      .max_relay = max_relay,
      .plan = plan.data_ptr<int64_t>()};
  a2av_dispatch_plan_impl(a, c10::cuda::getCurrentCUDAStream());
  return plan;
}

static int _register_dispatch [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("dispatch", [](py::module &m) {
    m.def("routed_meta", &routed_meta, py::arg("topk_ids"), py::arg("W"), py::arg("L"), py::arg("gpe"));
    m.def("a2av_dispatch_plan", &a2av_dispatch_plan, py::arg("sps"), py::arg("uc"), py::arg("W"), py::arg("L"),
          py::arg("gpe"), py::arg("rank"), py::arg("relay_slots"), py::arg("copies_per_rank"),
          py::arg("max_recv"), py::arg("max_stage"), py::arg("max_relay"));
    m.def("a2av_demands", &a2av_demands, py::arg("sps"), py::arg("uc"), py::arg("W"), py::arg("L"),
          py::arg("gpe"), py::arg("relay_slots"), py::arg("direct"), py::arg("caps"),
          py::arg("verdict") = py::none(), py::arg("layer") = 0, py::arg("force") = false,
          py::arg("splits") = py::none(), py::arg("plan") = py::none());
    py::class_<DispatchOptions>(m, "DispatchOptions")
        .def(
            py::init([](int64_t max_recv_rows, int64_t max_stage_rows, int64_t max_relay_rows) {
              return DispatchOptions{max_recv_rows, max_stage_rows, max_relay_rows};
            }),
            py::arg("max_recv_rows"),
            py::arg("max_stage_rows"),
            py::arg("max_relay_rows"));
    py::class_<DispatchCls>(m, "DispatchGemmOp")
        .def(
            py::init([](DistEnvTPWithEP &env, MoeArguments &moe_args, DispatchOptions const &options) {
              return new DispatchCls(
                  std::make_shared<C10dProcessGroup>("", env.tp_group),
                  env.ep_size,
                  env.nnodes,
                  moe_args.max_ntokens,
                  moe_args.ffn_hidden,
                  moe_args.hidden,
                  moe_args.nexperts,
                  moe_args.topk,
                  moe_args.input_dtype,
                  moe_args.output_dtype,
                  options);
            }),
            py::arg("env"),
            py::arg("moe_args"),
            py::arg("options"))
        .def("clear_buffers", &DispatchCls::clear_buffers)
        .def("resize_capacities", &DispatchCls::resize_capacities, py::arg("options"))
        .def("gemm_start_mark", &DispatchCls::gemm_start_mark)
        .def("run_id", &DispatchCls::run_id)
        .def("advance_run_id", &DispatchCls::advance_run_id, py::arg("n"))
        .def("set_gemm_start_mark", &DispatchCls::set_gemm_start_mark, py::arg("epoch"))
        .def("rows_dev", &DispatchCls::rows_dev)
        // the planning sync inside runs without the GIL, so other Python threads (e.g. the serving
        // framework's watchdog) can run while it waits on the device
        .def(
            "derive_routed_meta",
            &DispatchCls::derive_routed_meta,
            py::arg("topk_ids"),
            py::arg("caps") = std::vector<int64_t>{},
            py::arg("direct") = false,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "forward",
            &DispatchCls::forward,
            py::arg("inputs_shard"),
            py::arg("weights"),
            py::arg("splits_gpu"),
            py::arg("scatter_index"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("outputs_buf") = py::none(),
            py::arg("allgather_output") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("splits_per_source") = py::none(),
            py::arg("unique_counts") = py::none(),
            py::arg("weight_signal") = py::none(),
            py::arg("weight_signal_epoch") = 0,
            py::arg("weight_gate_group_start") = -1,
            py::arg("sched_expert_order") = py::none(),
            py::arg("sched_n_front") = 0,
            py::arg("weight_ptr_override") = py::none());
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
