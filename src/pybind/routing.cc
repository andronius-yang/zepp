// Capacity-constrained routing kernel (device-side, one launch pair per iteration).
#include "planner/routing.h"
#include "flux/ths_op/ths_pybind.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;

static int _register_routing [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("routing", [](py::module &m) {
    m.def("route_workspace_ints", &bytedance::flux::route_workspace_ints, py::arg("num_experts"), py::arg("ranks"));
    m.def("route_fused", &bytedance::flux::route_fused, py::arg("topk_own"), py::arg("loads"), py::arg("l2p"),
          py::arg("lcnts"), py::arg("my_rank"), py::arg("slots_per_rank"), py::arg("ranks_per_node"), py::arg("c_num"),
          py::arg("c_den"), py::arg("ws"), py::arg("probs_own"), py::arg("send"));
    m.def(
        "route",
        &bytedance::flux::route,
        py::arg("topk_own"),
        py::arg("loads"),
        py::arg("l2p"),
        py::arg("lcnts"),
        py::arg("my_rank"),
        py::arg("slots_per_rank"),
        py::arg("ranks_per_node"),
        py::arg("c_num"),
        py::arg("c_den"),
        py::arg("workspace"));
    m.def("swap_decide_out_ints", &bytedance::flux::swap_decide_out_ints, py::arg("ranks"), py::arg("num_experts"),
          py::arg("slots_per_rank"), py::arg("cap"));
    m.def(
        "swap_decide",
        &bytedance::flux::swap_decide,
        py::arg("loads"),
        py::arg("ntok"),
        py::arg("bucket"),
        py::arg("topk"),
        py::arg("p2l"),
        py::arg("l2p"),
        py::arg("lcnts"),
        py::arg("ranks_per_node"),
        py::arg("slots_per_rank"),
        py::arg("c"),
        py::arg("cap"),
        py::arg("max_rounds"),
        py::arg("out"));
    m.def("zero_rows_bounded", &bytedance::flux::zero_rows_bounded, py::arg("buf"), py::arg("rows_dev"));
    m.def("silu_mul_bounded", &bytedance::flux::silu_mul_bounded, py::arg("h"), py::arg("out"), py::arg("ffn"),
          py::arg("rows_dev"));
    m.def("planner_tail", &bytedance::flux::planner_tail, py::arg("gather"), py::arg("probs"), py::arg("vce"),
          py::arg("R"), py::arg("n"), py::arg("nlp"), py::arg("gpe"));
    m.def("graph_stage_in", &bytedance::flux::graph_stage_in, py::arg("x"), py::arg("ids"), py::arg("w"), py::arg("pad"),
          py::arg("xb"), py::arg("idsb"), py::arg("wb"));
    m.def("lane_arm", &bytedance::flux::lane_arm, py::arg("blk"), py::arg("ranks"), py::arg("rank"),
          py::arg("slots_per_rank"), py::arg("cap"), py::arg("gate"), py::arg("sched"), py::arg("ovr0"),
          py::arg("ovr1"), py::arg("stag_addr"), py::arg("epoch"), py::arg("always") = false,
          py::arg("gmap") = py::none());
    m.def("lane_push", &bytedance::flux::lane_push, py::arg("blk"), py::arg("ranks"), py::arg("ranks_per_node"),
          py::arg("rank"), py::arg("slots_per_rank"), py::arg("cap"), py::arg("k"), py::arg("slots"),
          py::arg("slot_bytes"), py::arg("peer_stag"), py::arg("peer_gate"), py::arg("epoch"), py::arg("counter"));
    m.def("lane_commit", &bytedance::flux::lane_commit, py::arg("blk"), py::arg("ranks"), py::arg("rank"),
          py::arg("slots_per_rank"), py::arg("cap"), py::arg("k"), py::arg("gate"), py::arg("slots"),
          py::arg("slot_bytes"), py::arg("stag_addr"), py::arg("ovr"), py::arg("epoch"));
    m.def("pad_rebuild", &bytedance::flux::pad_rebuild, py::arg("blk"), py::arg("p2l"), py::arg("lcnts"),
          py::arg("rank"), py::arg("slots_per_rank"), py::arg("home"), py::arg("pad"),
          py::arg("pad_prev") = py::none(), py::arg("flag") = py::none());
    m.def("lane_device_preload", &bytedance::flux::lane_device_preload);
    m.def("dual3_push_w1", &bytedance::flux::dual3_push_w1, py::arg("blk"), py::arg("ranks"), py::arg("ranks_per_node"),
          py::arg("rank"), py::arg("slots_per_rank"), py::arg("cap"), py::arg("slots"), py::arg("slot_bytes"),
          py::arg("peer_stag"), py::arg("peer_gate"), py::arg("mark"), py::arg("epoch"), py::arg("counter"));
    m.def("dual3_pull_w2", &bytedance::flux::dual3_pull_w2, py::arg("blk"), py::arg("ranks"), py::arg("ranks_per_node"),
          py::arg("rank"), py::arg("slots_per_rank"), py::arg("cap"), py::arg("peer_w2"), py::arg("slot_bytes"),
          py::arg("stag_addr"), py::arg("gate"), py::arg("peer_ack"), py::arg("mark"), py::arg("epoch"),
          py::arg("counter"));
    m.def("dual3_wait", &bytedance::flux::dual3_wait, py::arg("blk"), py::arg("ranks"), py::arg("ranks_per_node"),
          py::arg("rank"), py::arg("cap"), py::arg("words"), py::arg("pushed"), py::arg("epoch"));
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
