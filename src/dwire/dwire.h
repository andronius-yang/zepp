// Device-issued wire: the dispatch and combine wire ops are issued by kernels (no copy engine, no host
// thread): NVLink rows are stored straight into the peers' buffers by the kernels that produce them, and the
// inter-node puts are NVSHMEM device puts issued by a one-warp kernel in schedule order (on Slingshot / libfabric,
// NVSHMEM's CPU proxy still executes the RMA). The puts go in groups: non-blocking puts, one quiet, then the
// group's signals, so a signal never becomes visible before its data (a non-blocking put-with-signal does not
// guarantee that on libfabric / CXI).
//
// Ordering rules every kernel here follows:
//  - data stores of a destination by many blocks: each block fences (__threadfence_system) and arrives on a per-
//    destination counter (atomicInc wrap = self-resetting); the LAST arrival release-stores the signal. Every block
//    arrives once per destination, also with zero rows and on degenerate (aborted) layers, so a signal is never missed.
//  - consumers spin with system-scope acquire loads and poll the process kill word (a hang becomes an error).
//  - spinning kernels (relay, forward, wire warp) are small (128 threads x <= 32 registers per block, or one warp) so
//    they fit beside a resident GEMM block on any SM (otherwise the block scheduler would hold them behind the
//    persistent GEMM that waits for them); no kernel mixes producer and consumer roles across ranks.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace bytedance::flux {

struct DwireDispatchArgs {
  const int64_t *plan;     // DispatchPlan block of the step (sort_util.h)
  const uint64_t *run;     // step run id (device step slot), or nullptr -> run_host
  uint64_t run_host;
  int L, NN, my_node, my_lr, rank, relay_slots;
  int64_t row_bytes;       // multiple of 16
  int64_t relay_slot_rows; // rows of one relay staging slot
  // pack-push
  const char *src;           // the step's input rows [tokens][row_bytes]
  const int64_t *pack_gather;  // send row -> input token
  char *send_base;           // local send buffer
  char *const *recv_peer;    // [L] node peers' receive buffers (index = local rank, mine included)
  uint64_t *const *sig_peer;   // [L] node peers' arrival signal arrays
  uint64_t *const *segsig_peer;  // [L] node peers' pack-announce arrays
  uint64_t *pack_words;      // local, [nseg + 1] epoch words (written = run)
  unsigned *pack_counters;   // [nseg] last-block counters (zeroed once at allocation)
  // relay gather + wire warp
  const char *const *peer_send;  // [L] node peers' send buffers (mine included)
  char *relay_base;          // local relay staging (symmetric heap: a put source)
  uint64_t *slot_ready;      // local [NN] words: round dn's staging slot is complete
  uint64_t *slot_free;       // local [NN] words: round dn's put returned (slot reusable)
  unsigned *relay_counters;  // [NN]
  char *stage_base;          // symmetric gateway staging (put destination, same address on the target)
  uint64_t *node_sig;        // symmetric [NN] arrival words (put signals)
  // gateway forward
  unsigned *fwd_counters;    // [NN]
  const uint64_t *kill;      // process kill word (host-mapped)
  uint64_t *const *gw_sig_peer;  // unused (reserved)
  int remote_first = 0;      // pack-push order: remote segments (the inter-node wire's inputs) before own-node ones
  int k_lo = 0, k_hi = -1;   // pack-push: the segment positions [k_lo, k_hi) of that order (-1: all)
  int nbi_group = 0;         // wire warp: rounds in groups of this size (<= relay slots): nbi puts, one quiet, then
                             // the signals (signal after data on CXI)
};

// launches (each on its own stream; the caller orders the pack-push before the others and before GEMM 1)
void dwire_dispatch_pack_push(DwireDispatchArgs const &a, int blocks, cudaStream_t stream);
void dwire_dispatch_relay(DwireDispatchArgs const &a, int blocks, cudaStream_t stream);
void dwire_dispatch_forward(DwireDispatchArgs const &a, int blocks, cudaStream_t stream);
void dwire_dispatch_wire(DwireDispatchArgs const &a, cudaStream_t stream);

// combine wire warp: per split, per remote target node in the pre-reduce's order, wait for the target's wire
// rows (wire_flags, set by the pre-reduce) and issue the non-blocking put of the wire panel segment into the
// same-local-rank home's receive panel; per group of targets one quiet, then the signals recv_sig[rank * n_split +
// sid] = run at those homes (empty: signal only)
struct DwireCombineArgs {
  const int64_t *plan;  // combine plan block (args/gemm_combine.h CombinePlanLayout)
  int64_t lay_wire_rows, lay_wire_seg, lay_dst_off;
  int nbi_group = 0;  // positions in groups of this size: nbi puts, one quiet, then the signals
  const uint64_t *run;  // step slot combine run, or nullptr -> run_host
  uint64_t run_host;
  int L, NN, my_node, my_lr, rank, n_split;
  int node_order[64];   // pre-reduce order of the remote target nodes (NN - 1 entries)
  char *wire_base;      // symmetric wire panel (put source)
  char *recv_base;      // symmetric receive panel (put destination, same address at the home)
  int64_t row_bytes, wire_cap, recv_cap;  // rows per split
  const uint64_t *wire_flags;  // local [NN * n_split]
  uint64_t *recv_sig;          // symmetric [W * n_split]
  const uint64_t *kill;
};
void dwire_combine_wire(DwireCombineArgs const &a, cudaStream_t stream);

}  // namespace bytedance::flux
