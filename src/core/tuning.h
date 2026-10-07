#pragma once
// Frozen tuning constants of the fused dispatch / combine ops, selected on the A100 /
// Slingshot testbed. Not user knobs.
namespace bytedance::flux::tuning {
constexpr int kRelaySlots = 2;              // per-round relay staging is a two-slot double buffer
constexpr int kCombineWireStreams = 16;     // combine wire lanes (blocking puts in flight per rank)
constexpr int kCombinePackBlocks = 10;      // SMs reserved for the combine pack kernel
constexpr int kCombineReduceBlocks = 8;     // SMs reserved for the per-wave reduce
constexpr int kCombinePreReduceBlocks = 6;  // SMs reserved for the pre-reduce
constexpr int kCombineWaveAdapt = 48;       // destination waves collapse below this many GEMM tiles
constexpr int kCombineWaveNodes = 1;        // ring-consecutive destination nodes per combine wave
constexpr int kWireNbiGroup = 3;            // device wire: non-blocking puts per quiet (dispatch: at most kRelaySlots)
constexpr int kDwirePackBlocks = 64;        // device wire: blocks of the dispatch pack-push kernel
constexpr int kDwireRelayBlocks = 16;       // device wire: blocks of the dispatch relay kernel
constexpr int kDwireForwardBlocks = 32;     // device wire: blocks of the dispatch gateway-forward kernel
}  // namespace bytedance::flux::tuning
