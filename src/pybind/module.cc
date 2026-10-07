// Python module `zepp._C`: process-group bootstrap, symmetric-heap tensors, and the
// registry of op bindings (each op source registers itself).
#include "flux/ths_op/ths_op.h"

#include <c10/cuda/CUDAStream.h>

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/kernel_registry.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"
#include "flux/ths_op/util.h"
#include "core/wire_runtime.h"
#include "core/verdict.h"
#include "core/step_state.h"
#include <nvshmem.h>
#include <nvshmemx.h>

namespace bytedance::flux::ths_op {

ThsOpsInitRegistry &
ThsOpsInitRegistry::instance() {
  static ThsOpsInitRegistry inst;
  return inst;
}

void
ThsOpsInitRegistry::register_one(std::string name, OpInitFunc &&func) {
  std::lock_guard<std::mutex> guard(register_mutex_);
  registry_.emplace(std::move(name), std::move(func));
}

void
ThsOpsInitRegistry::initialize_all(py::module &m) const {
  std::lock_guard<std::mutex> guard(register_mutex_);
  for (auto const &par : registry_) {
    auto [name, func] = par;
    func(m);
  }
}

PYBIND11_MODULE(FLUX_TORCH_EXTENSION_NAME, m) {
  m.def("init_shm", [](c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    init_flux_shm(std::make_unique<C10dProcessGroup>("", pg).get());
  });
  m.def(
      "create_tensor_list",
      py::overload_cast<
          const std::vector<int64_t> &,
          c10::ScalarType,
          c10::intrusive_ptr<c10d::ProcessGroup>,
          bool,
          bool>(&flux_create_tensor_list),
      py::arg("shape"),
      py::arg("dtype"),
      py::arg("pg"),
      py::arg("ring_mode") = false,
      py::arg("init_zero") = false);
  m.def(
      "create_symmetric_tensor",
      [](const std::vector<int64_t> &shape, c10::ScalarType dtype) {
        return nvshmem_create_tensor(shape, dtype);
      },
      py::arg("shape"),
      py::arg("dtype"));
  // kernel registry (flux/cuda/kernel_registry.h): load every kernel of the library once (lazy-loading
  // guard; call with the device idle, after NVSHMEM init); the names need no GPU (unit test)
  m.def("preload_all", []() { return bytedance::flux::preload_all_kernels(); });
  m.def("kernel_registry_names", []() { return bytedance::flux::kernel_registry_names(); });
  // deferred capacity verdict (core/verdict.h): the process's device block and its
  // forward window. verdict_begin zeroes the block on the current stream and arms it (the next layer-steps
  // are deferred-verdict steps), verdict_end disarms it, verdict_reset zeroes it between forwards.
  m.def("verdict_tensor", []() {
    return torch::from_blob(
        bytedance::flux::verdict_dev(), {bytedance::flux::Verdict::kWords},
        torch::TensorOptions(torch::kCUDA, c10::cuda::current_device()).dtype(torch::kLong));
  });
  m.def("verdict_begin", []() { bytedance::flux::verdict_begin(c10::cuda::getCurrentCUDAStream()); });
  // device step state (core/step_state.h): the head of a layer-step on the current stream
  // (advances the run ids and, with swap_advance, the swap epoch, into slot `slot`), the counter sync on an idle
  // device, and the end of the step's host enqueue
  m.def("step_head", [](int slot, int swap_advance) {
    bytedance::flux::step_head(slot, swap_advance, c10::cuda::getCurrentCUDAStream());
  }, py::arg("slot"), py::arg("swap_advance"));
  m.def("step_release", []() { bytedance::flux::step_release(); });
  m.def("step_sync_counters", [](uint64_t d, uint64_t r, uint64_t e) {
    bytedance::flux::step_sync_counters(d, r, e, c10::cuda::getCurrentCUDAStream());
  }, py::arg("dispatch_run"), py::arg("combine_run"), py::arg("swap_epoch"));
  m.def("step_counters_read", []() {
    // the device counters (synchronous; idle device)
    uint64_t w[bytedance::flux::StepSlot::kWords];
    const uint64_t *c = bytedance::flux::step_block() +
                        (size_t)bytedance::flux::StepSlot::kSlots * bytedance::flux::StepSlot::kWords;
    CUDA_CHECK(cudaMemcpy(w, c, sizeof(w), cudaMemcpyDeviceToHost));
    return std::vector<uint64_t>(w, w + bytedance::flux::StepSlot::kWords);
  });
  m.def("verdict_end", []() { bytedance::flux::verdict_end(); });
  m.def("verdict_reset", []() { bytedance::flux::verdict_reset(c10::cuda::getCurrentCUDAStream()); });
  m.def("verdict_layout", []() {
    using V = bytedance::flux::Verdict;
    py::dict d;
    d["words"] = V::kWords;
    d["mask"] = V::kMask;
    d["latch"] = V::kLatch;
    d["demands"] = V::kDem;
    d["err"] = V::kErr;
    d["dead"] = V::kDead;
    d["capacity_bits"] = V::kCapacityBits;
    d["force_bit"] = V::kForceBit;
    d["error_bit"] = V::kErrorBit;
    return d;
  });
  // process kill word: a non-zero value makes the spinning helper kernels give up their waits (a watchdog
  // can then turn a hang into an error); set it only to abandon the process
  m.def(
      "kill_word_set",
      [](int64_t value) { bytedance::flux::kill_word_set(static_cast<uint64_t>(value)); },
      py::arg("value") = 1);
  m.def("kill_word_get", []() { return static_cast<int64_t>(bytedance::flux::kill_word_get()); });
  m.def("barrier_all_on_stream", [](intptr_t stream) {
    nvshmemx_barrier_all_on_stream(reinterpret_cast<cudaStream_t>(stream));
  });
  m.def(
      "stream_wait_geq",
      [](torch::Tensor word, int64_t index, int64_t value, intptr_t stream) {
        TORCH_CHECK(word.is_cuda() && word.scalar_type() == at::ScalarType::Long);
        stream_wait_geq(
            static_cast<const char *>(word.data_ptr()) + 8 * index,
            static_cast<uint64_t>(value),
            reinterpret_cast<cudaStream_t>(stream));
      },
      py::arg("word"),
      py::arg("index"),
      py::arg("value"),
      py::arg("stream"));

  using GroupBarrierCls = TorchClassWrapper<GroupBarrier>;
  py::class_<GroupBarrierCls>(m, "GroupBarrier")
      .def(
          py::init([](c10::intrusive_ptr<c10d::ProcessGroup> pg, bool ring_mode) {
            return new GroupBarrierCls(std::make_shared<C10dProcessGroup>("", pg), ring_mode);
          }),
          py::arg("process_group"),
          py::arg("ring_mode") = false)
      .def("barrier_all", [](GroupBarrierCls &self, intptr_t stream) {
        self.barrier_all((cudaStream_t)stream);
      });

  py::class_<DistEnvTPWithEP>(m, "DistEnv")
      .def(
          py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                      int32_t nnodes,
                      c10::intrusive_ptr<c10d::ProcessGroup> ep_group) {
            return new DistEnvTPWithEP(tp_group, nnodes, ep_group);
          }),
          py::arg("tp_group"),
          py::arg("nnodes"),
          py::arg("ep_group") = py::none())
      .def("__repr__", &DistEnvTPWithEP::toString);

  py::class_<MoeArguments, c10::intrusive_ptr<MoeArguments>>(m, "MoeArguments")
      .def(
          py::init([](int32_t max_ntokens,
                      int32_t hidden,
                      int32_t ffn_hidden,
                      int32_t nexperts,
                      int32_t topk,
                      torch::ScalarType input_dtype,
                      py::object py_output_dtype) {
            auto output_dtype = py_output_dtype.is(py::none())
                                    ? input_dtype
                                    : torch::python::detail::py_object_to_dtype(py_output_dtype);
            return new MoeArguments(
                max_ntokens, hidden, ffn_hidden, nexperts, topk, input_dtype, output_dtype);
          }),
          py::arg("max_ntokens"),
          py::arg("hidden"),
          py::arg("ffn_hidden"),
          py::arg("nexperts"),
          py::arg("topk"),
          py::arg("input_dtype"),
          py::arg("output_dtype") = py::none());

  ThsOpsInitRegistry::instance().initialize_all(m);
}

}  // namespace bytedance::flux::ths_op
