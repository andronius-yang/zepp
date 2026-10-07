#pragma once
#include <torch/all.h>
#include <tuple>

namespace bytedance::flux {
int64_t workspace_ints(int64_t G, int64_t R);
int64_t route_workspace_ints(int64_t G, int64_t R);
std::tuple<torch::Tensor, torch::Tensor> route(
    const torch::Tensor topk_own, const torch::Tensor d, const torch::Tensor l2p,
    const torch::Tensor lcnts, int64_t my_rank, int64_t nlp, int64_t ranks_per_node,
    int64_t C_num, int64_t C_den, torch::Tensor ws);
// route() in three launches, also writing the routing exchange's send row [phys | gate weight bits] (the planner's
// router; route() is its reference)
std::tuple<torch::Tensor, torch::Tensor> route_fused(
    const torch::Tensor topk_own, const torch::Tensor d, const torch::Tensor l2p,
    const torch::Tensor lcnts, int64_t my_rank, int64_t nlp, int64_t ranks_per_node,
    int64_t C_num, int64_t C_den, torch::Tensor ws, const torch::Tensor probs_own, torch::Tensor send);
// Intra-node expert-swap decision on the device (swap_decide.cu): result block size, then the launch
// (rewrites p2l / l2p in place when a swap is decided).
int64_t swap_decide_out_ints(int64_t R, int64_t G, int64_t nlp, int64_t cap);
void swap_decide(const torch::Tensor loads, const torch::Tensor ntok, int64_t S_b, int64_t K,
                 torch::Tensor p2l, torch::Tensor l2p, const torch::Tensor lcnts, int64_t L, int64_t nlp,
                 double C, int64_t cap, int64_t max_rounds, torch::Tensor out);
// Staged swap lane on the device (lane_device.cu): arm / push / commit read the swap_decide result
// block; pad_rebuild writes the next step's pad table from the new placement.
// always / gmap: arm every step (deferred verdict; lane_device.cu lane_arm_kernel)
void lane_arm(const torch::Tensor blk, int64_t R, int64_t rank, int64_t nlp, int64_t cap, torch::Tensor gate,
              torch::Tensor sched, torch::Tensor ovr0, torch::Tensor ovr1, const torch::Tensor stag_addr,
              int64_t epoch, bool always = false, c10::optional<torch::Tensor> gmap = c10::nullopt);
void lane_push(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t nlp, int64_t cap, int64_t k,
               const torch::Tensor slots, int64_t slot_bytes, const torch::Tensor peer_stag,
               const torch::Tensor peer_gate, int64_t epoch, torch::Tensor counter);
void lane_commit(const torch::Tensor blk, int64_t R, int64_t rank, int64_t nlp, int64_t cap, int64_t k,
                 const torch::Tensor gate, torch::Tensor slots, int64_t slot_bytes, int64_t stag_addr,
                 torch::Tensor ovr, int64_t epoch);
// pad_prev / flag: keep the overwritten table for a forward that runs again (deferred verdict)
void pad_rebuild(const torch::Tensor blk, const torch::Tensor p2l, const torch::Tensor lcnts, int64_t rank,
                 int64_t nlp, int64_t home, torch::Tensor pad, c10::optional<torch::Tensor> pad_prev = c10::nullopt,
                 c10::optional<torch::Tensor> flag = c10::nullopt);
// deferred-verdict steps: zero the first min(*rows_dev, buf.size(0)) rows of buf; SwiGLU over the first
// min(*rows_dev, h.size(0)) rows of h [m, 2*ffn] into out [>= m, ffn] (bf16)
void zero_rows_bounded(torch::Tensor buf, const torch::Tensor rows_dev);
void silu_mul_bounded(const torch::Tensor h, torch::Tensor out, int64_t ffn, const torch::Tensor rows_dev);
void planner_tail(const torch::Tensor gather, torch::Tensor probs, torch::Tensor vce, int64_t R, int64_t n,
                  int64_t nlp, int64_t gpe);
void graph_stage_in(const torch::Tensor x, const torch::Tensor ids, const torch::Tensor w, const torch::Tensor pad,
                    torch::Tensor xb, torch::Tensor idsb, torch::Tensor wb);
// dual3 lane on the device (lane_device.cu): W1 pushed under the sender's dispatch GEMM, W2 pulled under the
// receiver's combine GEMM, both spinning on the GEMM's start mark; dual3_wait joins a commit on the words
void dual3_push_w1(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t nlp, int64_t cap,
                   const torch::Tensor slots, int64_t slot_bytes, const torch::Tensor peer_stag,
                   const torch::Tensor peer_gate, const torch::Tensor mark, int64_t epoch, torch::Tensor counter);
void dual3_pull_w2(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t nlp, int64_t cap,
                   const torch::Tensor peer_w2, int64_t slot_bytes, int64_t stag_addr, torch::Tensor gate,
                   const torch::Tensor peer_ack, const torch::Tensor mark, int64_t epoch, torch::Tensor counter);
void dual3_wait(const torch::Tensor blk, int64_t R, int64_t L, int64_t rank, int64_t cap, const torch::Tensor words,
                bool pushed, int64_t epoch);
void lane_device_preload();
}  // namespace bytedance::flux
