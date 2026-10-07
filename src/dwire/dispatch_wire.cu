// Device-issued dispatch wire (dwire/dwire.h): the four kernels of one dispatch step.
//
//   pack-push   (before GEMM 1, main stream): own-node segments are gathered from the input rows straight into
//               each node peer's receive buffer (round 0 and the self copy), then the last arriving block raises
//               signal[rank] at that peer; remote segments are packed into the local send buffer, then the last
//               block writes the segment's pack word and announces it to the node peers.
//   relay       per round dn: waits for the pieces' pack announces, gathers them from the node peers' send
//               buffers into the round's relay staging slot (after the slot's previous put returned), then the
//               last block marks the slot ready.
//   wire warp   per round dn, in order, in groups of rounds: waits for its slot (or its own pack word for an
//               own-only round) and issues the round's non-blocking put to the same-local-rank gateway of the
//               target node (none for an empty round); after the group's puts one quiet, then the group's
//               signals and slot releases.
//   forward     per source node ns: waits for that node's arrival word, copies the window from the gateway
//               staging into every node peer's receive buffer, then the last block per (ns, peer) raises the
//               window's signal at that peer.
// The GEMM 1 tiles spin on these signal words (per-lane windows, epoch = the step's run id).
#include "dwire/dwire.h"

#include <nvshmem.h>
#include <nvshmemx.h>

#include <stdexcept>
#include <string>

#include "core/tuning.h"
#include "dispatch/sort_util.h"
#include "flux/cuda/kernel_registry.h"

namespace bytedance::flux {
namespace dw {
using DP = DispatchPlan;

__device__ __forceinline__ uint64_t
ld_acq(const uint64_t *p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void
st_rel(uint64_t *p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t
run_of(DwireDispatchArgs const &a) {
  return a.run != nullptr ? *a.run : a.run_host;
}
__device__ __forceinline__ bool
killed(const uint64_t *kill) {
  return kill != nullptr && *(volatile const uint64_t *)kill != 0;
}
// block-wide wait until *w >= v; false if the kill word was raised (the whole block returns)
__device__ __forceinline__ bool
block_wait(const uint64_t *w, uint64_t v, const uint64_t *kill) {
  __shared__ int s_ok;
  if (threadIdx.x == 0) {
    int ok = 1;
    for (unsigned spin = 0; ld_acq(w) < v; ++spin) {
      if ((spin & 63) == 0 && killed(kill)) {
        ok = 0;
        break;
      }
      __nanosleep(128);
    }
    s_ok = ok;
  }
  __syncthreads();
  const bool ok = s_ok != 0;
  __syncthreads();
  return ok;
}
// every block fences its stores and arrives once; true in thread 0 of the last arriving block (counter self-resets)
__device__ __forceinline__ bool
block_arrive(unsigned *counter) {
  __shared__ int s_last;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    s_last = atomicInc(counter, gridDim.x - 1) == gridDim.x - 1;
    if (s_last) {
      __threadfence_system();  // the other blocks' fenced stores happen-before the signal below
    }
  }
  __syncthreads();
  const bool last = s_last != 0 && threadIdx.x == 0;
  __syncthreads();
  return last;
}
// grid-strided 16-byte copies
__device__ __forceinline__ void
copy_vec(uint4 *d, const uint4 *s, int64_t total) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  for (; i < total; i += stride) {
    d[i] = s[i];
  }
}
__device__ __forceinline__ void
copy_rows(char *dst, const char *src, int64_t rows, int64_t row_bytes) {
  copy_vec(reinterpret_cast<uint4 *>(dst), reinterpret_cast<const uint4 *>(src), rows * (row_bytes / 16));
}
__device__ __forceinline__ int
remote_seg(int tn, int my_node, int L) {
  return tn < my_node ? tn : tn + L - 1;
}

__global__ void __launch_bounds__(512)
pack_push_kernel(DwireDispatchArgs a) {
  const int64_t *P = a.plan;
  const uint64_t run = run_of(a);
  const int L = a.L, NN = a.NN, my_node = a.my_node;
  const int64_t *seg = P + DP::seg_off();
  const int nseg = L + NN - 1;
  const int kk_hi = a.k_hi < 0 ? nseg : a.k_hi;
  // every segment of this launch in ONE grid-strided pass (the same rows to the same places), then one system
  // fence for all of the block's stores, then the per-segment arrivals: the last arriving block of a segment
  // raises its signals after every block's fenced stores, and every block still arrives once per segment
  constexpr int kMaxSeg = 40;
  __shared__ char *s_dst[kMaxSeg];
  __shared__ int64_t s_lo[kMaxSeg], s_cumr[kMaxSeg + 1];
  __shared__ int s_k[kMaxSeg];
  const int ns = kk_hi - a.k_lo;
  const int64_t vpr = a.row_bytes / 16;
  if (threadIdx.x == 0) {
    s_cumr[0] = 0;
    for (int i = 0; i < ns; ++i) {
      const int kk = a.k_lo + i;
      // own-node segments k < L, remote segments k >= L (wire order); remote_first packs the wire's inputs first
      const int k = a.remote_first == 0 ? kk : (kk < NN - 1 ? L + kk : kk - (NN - 1));
      int64_t rows, lo;
      char *dst;
      if (k < L) {
        // own-node segment of local destination k: rows go straight to the peer (self: my receive buffer)
        const int s = my_node + k;
        const int64_t *r0 = P + DP::round0(L, NN) + 3 * k;  // rows, send row, receive row
        rows = r0[0];
        lo = seg[s];
        dst = a.recv_peer[k] + r0[2] * a.row_bytes;
      } else {
        // remote segment: packed into the local send buffer for the relays' pulls
        const int dn = k - L + 1, tn = (my_node - dn + NN) % NN, s = remote_seg(tn, my_node, L);
        rows = seg[s + 1] - seg[s];
        lo = seg[s];
        dst = a.send_base + seg[s] * a.row_bytes;
      }
      s_dst[i] = dst;
      s_lo[i] = lo;
      s_k[i] = k;
      s_cumr[i + 1] = s_cumr[i] + rows;
    }
  }
  __syncthreads();
  // one warp per row: the segment, source row and destination once per row, then every lane four independent
  // 16-byte loads before its four stores
  constexpr int kU = 4;
  const int lane = threadIdx.x & 31;
  const int64_t nwarps = ((int64_t)gridDim.x * blockDim.x) >> 5;
  for (int64_t row = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5; row < s_cumr[ns]; row += nwarps) {
    int i = 0;
    while (row >= s_cumr[i + 1]) {
      ++i;
    }
    const int64_t r = row - s_cumr[i];
    uint4 *d = reinterpret_cast<uint4 *>(s_dst[i] + r * a.row_bytes);
    const uint4 *s = reinterpret_cast<const uint4 *>(a.src + a.pack_gather[s_lo[i] + r] * a.row_bytes);
    for (int64_t c0 = 0; c0 < vpr; c0 += 32 * kU) {
      uint4 v[kU];
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int64_t c = c0 + u * 32 + lane;
        if (c < vpr) {
          v[u] = s[c];
        }
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int64_t c = c0 + u * 32 + lane;
        if (c < vpr) {
          d[c] = v[u];
        }
      }
    }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < ns; ++i) {
      const int k = s_k[i];
      if (k < L) {
        const int s = my_node + k;
        if (atomicInc(a.pack_counters + s, gridDim.x - 1) == gridDim.x - 1) {
          __threadfence_system();
          st_rel(a.sig_peer[k] + a.rank, run);  // the destination's tiles of source `rank` may run
          st_rel(a.pack_words + s, run);
        }
      } else {
        const int dn = k - L + 1, tn = (my_node - dn + NN) % NN, s = remote_seg(tn, my_node, L);
        if (atomicInc(a.pack_counters + s, gridDim.x - 1) == gridDim.x - 1) {
          __threadfence_system();
          st_rel(a.pack_words + s, run);
          for (int dl = 0; dl < L; ++dl) {
            if (dl != a.my_lr) {
              st_rel(a.segsig_peer[dl] + (int64_t)a.my_lr * NN + tn, run);
            }
          }
        }
      }
    }
  }
}

__global__ void __launch_bounds__(128, 16)
relay_kernel(DwireDispatchArgs a) {
  const int64_t *P = a.plan;
  const uint64_t run = run_of(a);
  const int L = a.L, NN = a.NN, my_node = a.my_node, S = a.relay_slots;
  const uint64_t *segsig = a.segsig_peer[a.my_lr];  // my own announce array (peers write it)
  for (int dn = 1; dn < NN; ++dn) {
    const int tn = (my_node - dn + NN) % NN;
    const int64_t *r = P + DP::tn(L, NN, tn);
    const bool own_only = r[DP::kOwnOnly] != 0;
    const int64_t np = r[DP::kNpieces];
    const int64_t slot_base = (int64_t)((dn - 1) % S) * a.relay_slot_rows;
    if (!own_only && np > 0) {
      if (dn > S && !block_wait(a.slot_free + (dn - S), run, a.kill)) {
        return;
      }
      for (int64_t q = 0; q < np; ++q) {
        const int64_t *z = r + DP::kPieces + 4 * q;  // source local rank, source row, slot row, rows
        const int sl = (int)z[0];
        const uint64_t *ready =
            sl == a.my_lr ? a.pack_words + remote_seg(tn, my_node, L) : segsig + (int64_t)sl * NN + tn;
        if (!block_wait(ready, run, a.kill)) {
          return;
        }
        copy_rows(a.relay_base + (slot_base + z[2]) * a.row_bytes, a.peer_send[sl] + z[1] * a.row_bytes, z[3],
                  a.row_bytes);
      }
    }
    if (block_arrive(a.relay_counters + dn)) {
      st_rel(a.slot_ready + dn, run);
    }
  }
}

__global__ void __launch_bounds__(32, 1)
wire_kernel(DwireDispatchArgs a) {
  if (threadIdx.x != 0) {
    return;
  }
  const int64_t *P = a.plan;
  const uint64_t run = run_of(a);
  const int L = a.L, NN = a.NN, my_node = a.my_node, S = a.relay_slots;
  // rounds in groups of nbi_group (<= relay slots, so a group's rounds use distinct slots and a slot's next use is
  // in a later group): each round's put is issued nbi as soon as its source is ready, then ONE quiet (every put of
  // this PE complete at its target) and only then the rounds' signals and slot releases: signal after data
  // without serializing the puts' fixed costs
  for (int dn0 = 1; dn0 < NN; dn0 += a.nbi_group) {
    const int dn1 = min(NN, dn0 + a.nbi_group);
    for (int dn = dn0; dn < dn1; ++dn) {
      const int tn = (my_node - dn + NN) % NN;
      const int g = tn * L + a.my_lr;
      const int64_t *pt = P + DP::tn(L, NN, tn);
      const int64_t rows = pt[DP::kBme] - pt[DP::kAme];
      if (rows == 0) {
        continue;
      }
      const bool own_only = pt[DP::kOwnOnly] != 0;
      const uint64_t *ready = own_only ? a.pack_words + remote_seg(tn, my_node, L) : a.slot_ready + dn;
      for (unsigned spin = 0; ld_acq(ready) < run; ++spin) {
        if ((spin & 63) == 0 && killed(a.kill)) {
          return;
        }
        __nanosleep(128);
      }
      const char *srcp = own_only ? a.send_base + pt[DP::kOwnSrc] * a.row_bytes
                                  : a.relay_base + (int64_t)((dn - 1) % S) * a.relay_slot_rows * a.row_bytes;
      nvshmem_putmem_nbi(a.stage_base + pt[DP::kWireDst] * a.row_bytes, srcp, (size_t)(rows * a.row_bytes), g);
    }
    nvshmem_quiet();
    for (int dn = dn0; dn < dn1; ++dn) {
      const int tn = (my_node - dn + NN) % NN;
      nvshmemx_signal_op(a.node_sig + my_node, run, NVSHMEM_SIGNAL_SET, tn * L + a.my_lr);
      st_rel(a.slot_free + dn, run);
    }
  }
}

__global__ void __launch_bounds__(128, 16)
forward_kernel(DwireDispatchArgs a) {
  const int64_t *P = a.plan;
  const uint64_t run = run_of(a);
  const int L = a.L, NN = a.NN, my_node = a.my_node;
  for (int dn = 1; dn < NN; ++dn) {
    const int ns = (my_node + dn) % NN;
    const int64_t *pn = P + DP::ns(L, NN, ns);
    const int64_t win_a = pn[DP::kWinA], rows = pn[DP::kWinB] - pn[DP::kWinA];
    const char *wstage = a.stage_base + pn[DP::kWstage] * a.row_bytes;
    if (!block_wait(a.node_sig + ns, run, a.kill)) {  // relay (ns, my_lr)'s put of the window landed
      return;
    }
    // a window to all L destinations in one pass: each 16-byte word of the window is loaded once and stored to every
    // destination; one system fence covers all of the block's stores, then the per-destination arrivals in ring
    // order (the last arriving block of a destination raises its slot)
    // (launch bounds 128 x 16 cap the registers at 32: the destinations live in shared memory, two loads in flight)
    constexpr int kU = 2;
    __shared__ uint4 *dst[8];
    if (threadIdx.x < L) {
      dst[threadIdx.x] =
          reinterpret_cast<uint4 *>(a.recv_peer[threadIdx.x] + (pn[DP::kFwd + threadIdx.x] + win_a) * a.row_bytes);
    }
    __syncthreads();
    const uint4 *src = reinterpret_cast<const uint4 *>(wstage);
    const int64_t total = rows * (a.row_bytes / 16), stride = (int64_t)gridDim.x * blockDim.x;
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    for (; i + (kU - 1) * stride < total; i += kU * stride) {
      uint4 v[kU];
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        v[u] = src[i + u * stride];
      }
      for (int dl = 0; dl < L; ++dl) {
#pragma unroll
        for (int u = 0; u < kU; ++u) {
          dst[dl][i + u * stride] = v[u];
        }
      }
    }
    for (; i < total; i += stride) {
      const uint4 v = src[i];
      for (int dl = 0; dl < L; ++dl) {
        dst[dl][i] = v;
      }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int dl = 0; dl < L; ++dl) {
        const int dlg = (a.my_lr + 1 + dn + dl) % L;  // ring rotation: destination order staggered per (gateway, round)
        if (atomicInc(a.fwd_counters + ns * L + dlg, gridDim.x - 1) == gridDim.x - 1) {
          __threadfence_system();
          st_rel(a.sig_peer[dlg] + (int64_t)ns * L + a.my_lr, run);  // the window's gating slot at dlg
        }
      }
    }
    __syncthreads();
  }
}
__global__ void __launch_bounds__(32, 1)
combine_wire_kernel(DwireCombineArgs a) {
  if (threadIdx.x != 0) {
    return;
  }
  const int64_t *P = a.plan;
  const uint64_t run = a.run != nullptr ? *a.run : a.run_host;
  // positions in groups: each node's put issued nbi once its pre-reduce flag is up, then ONE quiet, then the
  // group's signals (signal after data)
  for (int sid = 0; sid < a.n_split; ++sid) {
    for (int g0 = 0; g0 < a.NN - 1; g0 += a.nbi_group) {
      const int g1 = min(a.NN - 1, g0 + a.nbi_group);
      for (int gi = g0; gi < g1; ++gi) {
        const int tn = a.node_order[gi];
        const int d = tn * a.L + a.my_lr;
        for (unsigned spin = 0; ld_acq(a.wire_flags + (int64_t)tn * a.n_split + sid) < run; ++spin) {
          if ((spin & 63) == 0 && killed(a.kill)) {
            return;
          }
          __nanosleep(128);
        }
        const int64_t rows = P[a.lay_wire_rows + tn];
        if (rows > 0) {
          const int64_t seg = P[a.lay_wire_seg + (tn < a.my_node ? tn : tn - 1)];
          char *dst = a.recv_base + ((int64_t)sid * a.recv_cap + P[a.lay_dst_off + d]) * a.row_bytes;
          const char *src = a.wire_base + ((int64_t)sid * a.wire_cap + seg) * a.row_bytes;
          nvshmem_putmem_nbi(dst, src, (size_t)(rows * a.row_bytes), d);
        }
      }
      nvshmem_quiet();
      for (int gi = g0; gi < g1; ++gi) {
        const int tn = a.node_order[gi];
        const int d = tn * a.L + a.my_lr;
        nvshmemx_signal_op(a.recv_sig + (int64_t)a.rank * a.n_split + sid, run, NVSHMEM_SIGNAL_SET, d);
      }
    }
  }
}
}  // namespace dw

namespace {
void
check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("device wire: ") + what + ": " + cudaGetErrorString(e));
  }
}
// The spinning wire kernels become resident BEFORE the GEMMs whose tiles they feed. An SM's L1 / shared-memory split
// is fixed while blocks are resident, so a kernel that needs no shared memory would leave its SMs configured too small
// for a GEMM block (66 KB of shared memory) and keep GEMM blocks off them until it drains. Every such kernel prefers
// the maximum shared-memory carveout.
void
prefer_max_shared(const void *fn) {
  check(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared),
        "carveout");
}
void
init_carveouts() {
  static const bool done = [] {
    prefer_max_shared(reinterpret_cast<const void *>(&dw::relay_kernel));
    prefer_max_shared(reinterpret_cast<const void *>(&dw::forward_kernel));
    prefer_max_shared(reinterpret_cast<const void *>(&dw::wire_kernel));
    prefer_max_shared(reinterpret_cast<const void *>(&dw::combine_wire_kernel));
    prefer_max_shared(reinterpret_cast<const void *>(&dw::pack_push_kernel));
    return true;
  }();
  (void)done;
}
}  // namespace

void
dwire_dispatch_pack_push(DwireDispatchArgs const &a, int blocks, cudaStream_t stream) {
  init_carveouts();
  dw::pack_push_kernel<<<blocks, 512, 0, stream>>>(a);
  check(cudaGetLastError(), "pack_push_kernel");
}
void
dwire_dispatch_relay(DwireDispatchArgs const &a, int blocks, cudaStream_t stream) {
  init_carveouts();
  dw::relay_kernel<<<blocks, 128, 0, stream>>>(a);
  check(cudaGetLastError(), "relay_kernel");
}
void
dwire_dispatch_forward(DwireDispatchArgs const &a, int blocks, cudaStream_t stream) {
  init_carveouts();
  if (a.L > 8) {
    throw std::runtime_error("device wire: the gateway forward serves at most 8 ranks per node");
  }
  dw::forward_kernel<<<blocks, 128, 0, stream>>>(a);
  check(cudaGetLastError(), "forward_kernel");
}
void
dwire_combine_wire(DwireCombineArgs const &a, cudaStream_t stream) {
  init_carveouts();
  DwireCombineArgs b = a;
  b.nbi_group = tuning::kWireNbiGroup;
  dw::combine_wire_kernel<<<1, 32, 0, stream>>>(b);
  check(cudaGetLastError(), "combine_wire_kernel");
}
void
dwire_dispatch_wire(DwireDispatchArgs const &a, cudaStream_t stream) {
  init_carveouts();
  DwireDispatchArgs b = a;
  // a group's rounds on distinct slots
  b.nbi_group = tuning::kWireNbiGroup > a.relay_slots ? a.relay_slots : tuning::kWireNbiGroup;
  dw::wire_kernel<<<1, 32, 0, stream>>>(b);
  check(cudaGetLastError(), "wire_kernel");
}

// kernel registry (include/flux/cuda/kernel_registry.h): every kernel of this file
void
dwire_kernels(KernelList &out) {
  out.push_back(ZEPP_KERNEL_ENTRY(dw::pack_push_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(dw::relay_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(dw::wire_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(dw::forward_kernel));
  out.push_back(ZEPP_KERNEL_ENTRY(dw::combine_wire_kernel));
}

}  // namespace bytedance::flux
