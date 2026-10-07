// `direct` communication strategy: a plain all-to-all wire and an un-fused GEMM.
#include "direct/all2all_single_2d.h"
#include "direct/gemm_only.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using All2AllSingleCls = TorchClassWrapper<All2AllSingle>;
using GemmOnlyCls = TorchClassWrapper<GemmOnly>;

static int _register_direct [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("direct", [](py::module &m) {
    py::class_<All2AllSingleCls>(m, "All2AllSingle")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> pg,
                        int64_t max_split,
                        int64_t n_dim,
                        int64_t local_world_size,
                        at::ScalarType input_dtype) {
              return new All2AllSingleCls(
                  std::make_shared<C10dProcessGroup>("", pg),
                  max_split,
                  n_dim,
                  local_world_size,
                  input_dtype,
                  /*ep_team=*/0);
            }),
            py::arg("pg"),
            py::arg("max_split"),
            py::arg("n_dim"),
            py::arg("local_world_size"),
            py::arg("input_dtype"))
        .def(
            "forward",
            &All2AllSingleCls::forward,
            py::arg("input"),
            py::arg("output"),
            py::arg("input_splits"),
            py::arg("output_splits"),
            py::arg("num_comm_sm"));

    py::class_<GemmOnlyCls>(m, "GemmOnly")
        .def(
            py::init([](torch::ScalarType input_dtype,
                        torch::ScalarType weight_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight) {
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);
              return new GemmOnlyCls(
                  input_dtype, weight_dtype, output_dtype, transpose_weight, /*use_fp8_gemm=*/false);
            }),
            py::arg("input_dtype"),
            py::arg("weight_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false)
        .def(
            "forward",
            &GemmOnlyCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output_buf") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
