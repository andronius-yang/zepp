// Deferred capacity verdict: one process-wide block of int64 words in device memory that replaces the
// per-layer host capacity check inside a serving forward (overlap strategy).
//
// Outside a forward (and with the direct strategy), every MoE layer lands its routing counts on the host, checks
// the exact buffer demands against the capacities and grows the buffers before anything is launched
// (python/zepp/serving.py). Inside a forward the check runs on the device: the demands kernel of every layer
// ORs its violation bits into the block, and the first violating layer latches its demands. From that layer
// on, the forward is degenerate: every device builder that sizes work reads the block first and emits zero
// sizes (zero GEMM tiles, signal-only wire, zero-row pack / pre-reduce / receivers, no swap rounds), so the
// ranks finish the forward without writing past any buffer and without hanging. The host reads the block
// once, at the end of the forward; on a violation it grows the buffers (collective, identical on every rank
// because the counts are replicated) and runs the forward again.
//
// Lifetime of the block within a forward: verdict_begin (zero + arm, on the forward's stream) -> layers
// -> verdict_end (disarm). Kernels see the block only while it is armed (verdict_armed() is null
// otherwise), so steps outside a forward (warm-up, priming) keep the host check.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace bytedance::flux {

struct Verdict {
  // words
  static constexpr int kMask = 0;   // sticky OR of the bits below
  static constexpr int kLatch = 1;  // 1 + MoE layer ordinal of the first violating layer, 0 = none
  static constexpr int kDem = 2;    // [7] that layer's demands (capacity.py Demands order)
  static constexpr int kErr = 9;    // device-assert bits (inconsistent counts), OR
  // 1 once a layer's demands kernel found the block set and zeroed that layer's counts: the step is degenerate
  // from its start. The per-copy builders key on this word, not on kMask, so that a device assert raised later
  // in a layer (it only adds to kMask) never changes that layer's shape halfway; the next layer is degenerate.
  static constexpr int kDead = 10;
  static constexpr int kWords = 16;
  // kMask bits: 0..7 = capacity violations (the demands kernel's mask: python/zepp/capacity.py _FIT
  // order, direct strategy bits 0 and 7), plus
  static constexpr int64_t kCapacityBits = 0xff;
  static constexpr int64_t kForceBit = int64_t(1) << 16;  // forced abort (test hook)
  static constexpr int64_t kErrorBit = int64_t(1) << 17;  // a device assert fired (kErr holds which)
  // kErr bits
  static constexpr int64_t kErrDispatchPlan = 0xff;           // sort_util.h DispatchPlan::Err, non-capacity
  static constexpr int64_t kErrCombineRows = int64_t(1) << 8;   // rows this rank computes above the bound
  static constexpr int64_t kErrCombineCols = int64_t(1) << 9;   // a home's copies != copies per rank
  static constexpr int64_t kErrCombineSend = int64_t(1) << 10;  // send panel rows
  static constexpr int64_t kErrCombineConv = int64_t(1) << 11;  // convergence panel rows
  static constexpr int64_t kErrCombineWire = int64_t(1) << 12;  // wire panel rows
  static constexpr int64_t kErrCombineLanes = int64_t(1) << 13;  // receive image above the receive panel
  static constexpr int64_t kErrCombineUnion = int64_t(1) << 14;  // wire rows without convergence rows
};

// the device block (allocated and zeroed on the first call: call it on an idle device, e.g. from an op
// constructor)
int64_t *verdict_dev();
// the device block while a forward is open, nullptr otherwise
int64_t *verdict_armed();
// zero the block on `stream` and arm it
void verdict_begin(cudaStream_t stream);
// disarm (the block keeps its value until the next begin / reset)
void verdict_end();
// zero the block on `stream` (between forwards: before a re-prime or a warm-up step)
void verdict_reset(cudaStream_t stream);

}  // namespace bytedance::flux
