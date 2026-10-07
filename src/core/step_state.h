// Device step state: the per-layer-step values the kernels compare remote-written words
// against (the dispatch and combine run ids, the swap lane epoch), kept in device memory so that a captured
// layer-step (CUDA graph) reads the current values instead of the ones baked in at capture.
//
// One slot per MoE layer (int64 words). A one-thread head kernel at the start of every layer-step advances the
// process-wide counters and writes them into that layer's slot; every consumer of the step reads its own slot
// through a pointer. A slot is rewritten only by the head of the SAME layer in a later forward, so a later layer's
// head can never change a value that a still-resident persistent kernel of an earlier layer re-reads. The counters
// only increase (warm-up, redo and growth included) and advance identically on every rank: every rank runs the same
// layer-steps.
//
// The ops keep host copies of the same counters (the stream memops and the run_host fallbacks read them);
// step_sync_counters() sets the device counters to the host values on an idle device (after the warm-up / a
// re-prime), and the head advances both in lockstep from then on.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace bytedance::flux {

struct StepSlot {
  static constexpr int kDispatchRun = 0;  // dispatch run id of the step (tile-gate epoch of GEMM1)
  static constexpr int kCombineRun = 1;   // combine run id of the step (pack / pre-reduce / receivers)
  static constexpr int kSwapEpoch = 2;    // swap lane epoch of the step (weight gates, lane kernels)
  static constexpr int kWords = 8;
  static constexpr int kSlots = 256;      // MoE layers per model, upper bound
};

// the device block: [kSlots][kWords] slots, then the counters [kWords]; allocated and zeroed on the first call (call
// it on an idle device)
uint64_t *step_block();
// pointer to one word of a slot
const uint64_t *step_slot_word(int slot, int word);
// enqueue the head of a layer-step on `stream`: counters[dispatch] += 1, counters[combine] += 1, counters[swap] +=
// swap_advance, then copy them into slot `slot`; marks `slot` as the current one for the ops of this step
void step_head(int slot, int swap_advance, cudaStream_t stream);
// the slot of the step being enqueued (-1 before the first head, or after step_release())
int step_current_slot();
void step_release();
// set the device counters (idle device): dispatch run, combine run, swap epoch
void step_sync_counters(uint64_t dispatch_run, uint64_t combine_run, uint64_t swap_epoch, cudaStream_t stream);

}  // namespace bytedance::flux
