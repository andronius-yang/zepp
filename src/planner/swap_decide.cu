// Intra-node expert-swap decision on the device: the band-triggered orbit of python/zepp/swap.py
// (decide_swaps with per_pair = 1), bit for bit, in one single-block launch, together with the
// pad-row correction of the gathered loads (LayerState.pad_hist), the l2p rebuild (rebuild_l2p) and
// the per-rank pull lists (net_moves). The placement tables are rewritten in place, so the same
// step routes on the new placement; a compact result block tells the host what moved.
//
// Decision (identical to the host):
//   load_g[e]   = sum_r loads[r, e] - pad rows of every rank (closed form of its cyclic pad list)
//   w[s]        = load_g[e] // max(lcnts[e], 1) for the expert e in slot s (0 for an empty slot)
//   L_r         = sum of w over rank r's slots; node out of band iff max L_r * L > (1 + C) * sum L_r
//   round       per out-of-band node, pair the k-th heaviest with the k-th lightest rank (k < L/2),
//               pick the exchange among the top-8 slots of each side that most reduces the pair max
//               (ties: smallest heavy expert id, then smallest light expert id); stop when no
//               exchange exists or a rank would exceed `cap` changed slots; at most `max_rounds`.
//
// Result block (int64): [0] rounds (0 = no swap), [1] total moves, [2] error flags,
//   [3, 3+R) moves per rank, then R x cap x 4 moves (dst slot, src rank, src slot, expert) sorted by
//   destination slot, then the new p2l [R*nlp], then load_g [G] (after the pad correction).
#include <torch/all.h>
#include "planner/routing.h"
#include "flux/cuda/kernel_registry.h"
#include "core/verdict.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <climits>

#define SWAPDEC_CUDA_CHECK(expr)                                \
  do {                                                          \
    cudaError_t _e = (expr);                                    \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",              \
                cudaGetErrorString(_e));                        \
  } while (0)

namespace bytedance::flux {
namespace swapdec {

constexpr int kThreads = 256;
constexpr int kMaxL = 8;       // ranks per node
constexpr int kCand = 8;       // candidate slots per side (host: K = min(8, nlp))
constexpr int kMaxNlp = 256;   // slots per rank (no per-thread arrays depend on it)
constexpr long long kBig = 1LL << 60;

__device__ __forceinline__ long long floordiv(long long a, long long b) {  // Python a // b, b > 0
  long long q = a / b;
  if ((a % b != 0) && (a < 0)) --q;
  return q;
}

struct Smem {
  long long *lg, *lc, *w, *Lr;
  int *a0, *cur, *nxt, *H, *Lo, *sw_has, *sw_sh, *sw_sl, *sw_eh, *sw_el, *node_out;
};

__device__ __forceinline__ bool band_out(const long long *Lr, int n, int L, double C) {
  long long tot = 0, mx = LLONG_MIN;
  for (int j = 0; j < L; ++j) {
    long long v = Lr[n * L + j];
    tot += v;
    mx = v > mx ? v : mx;
  }
  return (double)mx * (double)L > (1.0 + C) * (double)tot;
}

// the sort key of slot s in the host's np.lexsort order: key1, then key2, then slot index, ascending
// (heavy side: key1 = -w; light side: key1 = w; empty slots last)
__device__ __forceinline__ void slot_key(const int *cur, const long long *w, int s, bool heavy,
                                         long long &k1, long long &k2) {
  const int e = cur[s];
  if (e >= 0) {
    k1 = heavy ? -w[s] : w[s];
    k2 = e;
  } else {
    k1 = kBig;
    k2 = kBig;
  }
}

// one warp per pair evaluates the pair's exchange (slot ranks, memberships and the Kc x Kc candidates in parallel,
// then a warp max of the selection key); the key is unique per (heavy expert, light expert), so the choice is the
// host's serial one, bit for bit
__global__ void swap_decide_kernel(
    const int *__restrict__ loads, const long long *__restrict__ ntok, long long S_b, int K_topk,
    long long *__restrict__ p2l, int *__restrict__ l2p, const int *__restrict__ lcnts, int R, int G, int L,
    int nlp, double C, int cap, int max_rounds, long long *__restrict__ out,
    const long long *__restrict__ verdict) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int S = R * nlp, NN = R / L, P = L / 2, Q = NN * P, tid = threadIdx.x, nt = blockDim.x;
  Smem sm;
  {
    long long *p = reinterpret_cast<long long *>(smem_raw);
    sm.lg = p; p += G;
    sm.lc = p; p += G;
    sm.w = p; p += S;
    sm.Lr = p; p += R;
    int *q = reinterpret_cast<int *>(p);
    sm.a0 = q; q += S;
    sm.cur = q; q += S;
    sm.nxt = q; q += S;
    sm.H = q; q += Q;
    sm.Lo = q; q += Q;
    sm.sw_has = q; q += Q;
    sm.sw_sh = q; q += Q;
    sm.sw_sl = q; q += Q;
    sm.sw_eh = q; q += Q;
    sm.sw_el = q; q += Q;
    sm.node_out = q; q += NN;
  }
  __shared__ int s_flag, s_rounds, s_maxdiff, s_nsw, s_dead;
  const long long off_cnt = 3, off_mv = off_cnt + R, off_p2l = off_mv + (long long)R * cap * 4,
                  off_lg = off_p2l + S;

  // 1. placement, replica counts, gathered loads
  for (int s = tid; s < S; s += nt) sm.a0[s] = (int)p2l[s];
  for (int e = tid; e < G; e += nt) {
    sm.lc[e] = lcnts[e] > 1 ? lcnts[e] : 1;
    long long v = 0;
    for (int r = 0; r < R; ++r) v += loads[(long long)r * G + e];
    sm.lg[e] = v;
  }
  __syncthreads();
  // 2. pad rows: rank r pads rows n_r..S_b-1; its pad list is its single-replica hosted experts,
  //    else its hosted experts, else its home shard (slot order); entry j of the flattened pad
  //    block carries expert list[j % m]
  for (int r = tid; r < R; r += nt) {
    const long long lo = ntok[r] * K_topk, hi = S_b * K_topk;
    if (hi <= lo) continue;
    int n_single = 0, n_hosted = 0;
    for (int j = 0; j < nlp; ++j) {
      int e = sm.a0[r * nlp + j];
      if (e >= 0) {
        ++n_hosted;
        if (lcnts[e] == 1) ++n_single;
      }
    }
    const int mode = n_single ? 0 : (n_hosted ? 1 : 2);
    const long long home = G / R;
    const long long m = mode == 0 ? n_single : (mode == 1 ? n_hosted : home);
    long long i = 0;
    for (int j = 0; j < (mode == 2 ? (int)home : nlp); ++j) {
      int e;
      if (mode == 2) {
        e = (int)(r * home + j);
      } else {
        e = sm.a0[r * nlp + j];
        if (e < 0 || (mode == 0 && lcnts[e] != 1)) continue;
      }
      long long cnt = floordiv(hi - 1 - i, m) - floordiv(lo - 1 - i, m);
      atomicAdd(reinterpret_cast<unsigned long long *>(&sm.lg[e]), (unsigned long long)(-cnt));
      ++i;
    }
  }
  __syncthreads();
  // 3. slot weights and reference loads
  for (int s = tid; s < S; s += nt) {
    int e = sm.a0[s];
    sm.w[s] = e >= 0 ? floordiv(sm.lg[e], sm.lc[e]) : 0;
    sm.cur[s] = sm.a0[s];
  }
  __syncthreads();
  for (int r = tid; r < R; r += nt) {
    long long v = 0;
    for (int j = 0; j < nlp; ++j) v += sm.w[r * nlp + j];
    sm.Lr[r] = v;
  }
  if (tid == 0) {
    s_flag = 0;
    s_rounds = 0;
    // deferred verdict set by an earlier layer of the forward (core/verdict.h): the forward is degenerate
    // and will run again, so this layer decides no swap (zero rounds; the lane's epoch still advances)
    s_dead = verdict != nullptr && *reinterpret_cast<volatile const long long *>(verdict) != 0;
  }
  __syncthreads();
  // 4. trigger: any node out of band
  for (int n = tid; n < NN; n += nt)
    if (band_out(sm.Lr, n, L, C)) atomicOr(&s_flag, 1);
  __syncthreads();
  const bool triggered = s_flag != 0 && !s_dead;
  __syncthreads();
  // 5. orbit rounds
  for (int round = 0; triggered && round < max_rounds; ++round) {
    for (int n = tid; n < NN; n += nt) {
      int ord[kMaxL];
      long long key[kMaxL];
      for (int j = 0; j < L; ++j) { key[j] = -sm.Lr[n * L + j] * L + j; ord[j] = j; }
      for (int i = 1; i < L; ++i) {
        int t = ord[i];
        int j = i - 1;
        while (j >= 0 && key[ord[j]] > key[t]) { ord[j + 1] = ord[j]; --j; }
        ord[j + 1] = t;
      }
      for (int k = 0; k < P; ++k) {
        sm.H[n * P + k] = n * L + ord[k];
        sm.Lo[n * P + k] = n * L + ord[L - 1 - k];
      }
      sm.node_out[n] = band_out(sm.Lr, n, L, C) ? 1 : 0;
    }
    __syncthreads();
    constexpr int kWarps = kThreads / 32;
    __shared__ int ws_h[kWarps][kCand], ws_l[kWarps][kCand];
    __shared__ unsigned char ws_hin[kWarps][kCand], ws_lin[kWarps][kCand];
    const int warp = tid >> 5, lane = tid & 31, nwarps = nt >> 5;
    const int Kc = nlp < kCand ? nlp : kCand;
    for (int q = warp; q < Q; q += nwarps) {
      const int h = sm.H[q], l = sm.Lo[q];
      const long long lrH = sm.Lr[h], lrL = sm.Lr[l];
      const bool pair_ok = (lrH - lrL > 1) && sm.node_out[h / L];
      // the top Kc slots of each side as ranks in the total order (key1, key2, slot): ws_h / ws_l[p] is the slot of
      // rank p
      for (int j = lane; j < nlp; j += 32) {
        long long a1, a2, b1, b2;
        slot_key(sm.cur, sm.w, h * nlp + j, true, a1, a2);
        slot_key(sm.cur, sm.w, l * nlp + j, false, b1, b2);
        int rh = 0, rl = 0;
        for (int i = 0; i < nlp; ++i) {
          long long c1, c2;
          slot_key(sm.cur, sm.w, h * nlp + i, true, c1, c2);
          rh += (c1 < a1 || (c1 == a1 && (c2 < a2 || (c2 == a2 && i < j)))) ? 1 : 0;
          slot_key(sm.cur, sm.w, l * nlp + i, false, c1, c2);
          rl += (c1 < b1 || (c1 == b1 && (c2 < b2 || (c2 == b2 && i < j)))) ? 1 : 0;
        }
        if (rh < Kc) ws_h[warp][rh] = j;
        if (rl < Kc) ws_l[warp][rl] = j;
      }
      __syncwarp();
      // memberships: the heavy candidate's expert hosted by l, the light candidate's expert hosted by h
      for (int c = lane; c < 2 * Kc; c += 32) {
        const bool hv = c < Kc;
        const int idx = hv ? c : c - Kc;
        const int e = hv ? sm.cur[h * nlp + ws_h[warp][idx]] : sm.cur[l * nlp + ws_l[warp][idx]];
        const int other = hv ? l : h;
        bool in = false;
        for (int j = 0; j < nlp; ++j) in |= (sm.cur[other * nlp + j] == e);
        if (hv) {
          ws_hin[warp][idx] = in ? 1 : 0;
        } else {
          ws_lin[warp][idx] = in ? 1 : 0;
        }
      }
      __syncwarp();
      long long best = -kBig;
      int bsh = -1, bsl = -1, beh = -1, bel = -1;
      for (int c = lane; c < Kc * Kc; c += 32) {
        const int a = c / Kc, b = c - (c / Kc) * Kc;
        const int sh = h * nlp + ws_h[warp][a], sl = l * nlp + ws_l[warp][b];
        const int eh = sm.cur[sh], el = sm.cur[sl];
        const long long wh = sm.w[sh], wl = sm.w[sl];
        const long long x1 = lrH - wh + wl, x2 = lrL - wl + wh;
        const long long new_max = x1 > x2 ? x1 : x2;
        const long long gain = (lrH > lrL ? lrH : lrL) - new_max;
        const bool ok = eh >= 0 && el >= 0 && !ws_hin[warp][a] && !ws_lin[warp][b] && eh != el && wh > wl &&
                        pair_ok && gain >= 1;
        if (!ok) continue;
        const long long sel = gain * ((long long)G * G + 1) - ((long long)eh * G + el);
        if (sel > best) { best = sel; bsh = sh; bsl = sl; beh = eh; bel = el; }
      }
      for (int o = 16; o > 0; o >>= 1) {
        const long long ob = __shfl_down_sync(0xffffffffu, best, o);
        const int osh = __shfl_down_sync(0xffffffffu, bsh, o), osl = __shfl_down_sync(0xffffffffu, bsl, o);
        const int oeh = __shfl_down_sync(0xffffffffu, beh, o), oel = __shfl_down_sync(0xffffffffu, bel, o);
        if (ob > best) { best = ob; bsh = osh; bsl = osl; beh = oeh; bel = oel; }
      }
      if (lane == 0) {
        sm.sw_has[q] = best > -kBig ? 1 : 0;
        sm.sw_sh[q] = bsh; sm.sw_sl[q] = bsl; sm.sw_eh[q] = beh; sm.sw_el[q] = bel;
      }
      __syncwarp();
    }
    if (tid == 0) { s_nsw = 0; s_maxdiff = 0; }
    __syncthreads();
    for (int q = tid; q < Q; q += nt)
      if (sm.sw_has[q]) atomicAdd(&s_nsw, 1);
    for (int s = tid; s < S; s += nt) sm.nxt[s] = sm.cur[s];
    __syncthreads();
    if (s_nsw == 0) break;
    for (int q = tid; q < Q; q += nt)
      if (sm.sw_has[q]) { sm.nxt[sm.sw_sh[q]] = sm.sw_el[q]; sm.nxt[sm.sw_sl[q]] = sm.sw_eh[q]; }
    __syncthreads();
    for (int r = tid; r < R; r += nt) {
      int d = 0;
      for (int j = 0; j < nlp; ++j) d += (sm.nxt[r * nlp + j] != sm.a0[r * nlp + j]);
      atomicMax(&s_maxdiff, d);
    }
    __syncthreads();
    if (s_maxdiff > cap) break;  // per_pair == 1: the round is dropped, the orbit stops
    for (int s = tid; s < S; s += nt) sm.cur[s] = sm.nxt[s];
    __syncthreads();
    if (tid == 0) {
      ++s_rounds;
      for (int q = 0; q < Q; ++q) {
        if (!sm.sw_has[q]) continue;
        const int sx2[2] = {sm.sw_sh[q], sm.sw_sl[q]};
        for (int t = 0; t < 2; ++t) {
          const int sx = sx2[t];
          const int e = sm.cur[sx];
          const long long w_new = e >= 0 ? floordiv(sm.lg[e], sm.lc[e]) : 0;
          sm.Lr[sx / nlp] += w_new - sm.w[sx];
          sm.w[sx] = w_new;
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  const int rounds = s_rounds;
  // 6. result block + tables
  for (int e = tid; e < G; e += nt) out[off_lg + e] = sm.lg[e];
  if (rounds == 0) {
    if (tid == 0) { out[0] = 0; out[1] = 0; out[2] = 0; }
    for (int r = tid; r < R; r += nt) out[off_cnt + r] = 0;
    for (int s = tid; s < S; s += nt) out[off_p2l + s] = sm.a0[s];
    return;
  }
  for (int s = tid; s < S; s += nt) {
    p2l[s] = sm.cur[s];
    out[off_p2l + s] = sm.cur[s];
  }
  for (int e = tid; e < G; e += nt) {  // l2p [G, R]: ascending physical slots of expert e, then -1
    int col = 0;
    for (int s = 0; s < S; ++s)
      if (sm.cur[s] == e) l2p[(long long)e * R + col++] = s;
    for (; col < R; ++col) l2p[(long long)e * R + col] = -1;
  }
  if (tid == 0) { s_nsw = 0; s_flag = 0; }
  __syncthreads();
  for (int r = tid; r < R; r += nt) {  // pull lists: (dst slot, src rank, src slot, expert), dst ascending
    const int node = r / L;
    int cnt = 0;
    for (int j = 0; j < nlp; ++j) {
      const int phys = r * nlp + j;
      const int e = sm.cur[phys];
      if (e < 0 || sm.a0[phys] == e) continue;
      int sr = -1, ss = -1;
      for (int rr = node * L; rr < (node + 1) * L; ++rr)
        for (int jj = 0; jj < nlp; ++jj)
          if (sm.a0[rr * nlp + jj] == e) { sr = rr; ss = jj; }
      if (sr < 0 || cnt >= cap) { atomicOr(&s_flag, sr < 0 ? 1 : 2); continue; }
      long long *mv = out + off_mv + ((long long)r * cap + cnt) * 4;
      mv[0] = j; mv[1] = sr; mv[2] = ss; mv[3] = e;
      ++cnt;
    }
    out[off_cnt + r] = cnt;
    atomicAdd(&s_nsw, cnt);
  }
  __syncthreads();
  if (tid == 0) { out[0] = rounds; out[1] = s_nsw; out[2] = s_flag; }
}

}  // namespace swapdec

int64_t swap_decide_out_ints(int64_t R, int64_t G, int64_t nlp, int64_t cap) {
  return 3 + R + R * cap * 4 + R * nlp + G;
}

void swap_decide(const torch::Tensor loads, const torch::Tensor ntok, int64_t S_b, int64_t K,
                 torch::Tensor p2l, torch::Tensor l2p, const torch::Tensor lcnts, int64_t L, int64_t nlp,
                 double C, int64_t cap, int64_t max_rounds, torch::Tensor out) {
  const int64_t R = loads.size(0), G = loads.size(1), S = R * nlp;
  TORCH_CHECK(loads.is_cuda() && loads.is_contiguous() && loads.scalar_type() == torch::kInt && loads.dim() == 2,
              "swap_decide: loads [R, G] int32 cuda");
  TORCH_CHECK(ntok.is_cuda() && ntok.scalar_type() == torch::kLong && ntok.numel() == R, "swap_decide: ntok [R] int64");
  TORCH_CHECK(p2l.is_cuda() && p2l.is_contiguous() && p2l.scalar_type() == torch::kLong && p2l.numel() == S,
              "swap_decide: p2l [R*nlp] int64");
  TORCH_CHECK(l2p.is_cuda() && l2p.is_contiguous() && l2p.scalar_type() == torch::kInt && l2p.size(0) == G &&
                  l2p.size(1) == R,
              "swap_decide: l2p [G, R] int32");
  TORCH_CHECK(lcnts.is_cuda() && lcnts.scalar_type() == torch::kInt && lcnts.numel() == G, "swap_decide: lcnts [G] int32");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kLong && out.numel() >= swap_decide_out_ints(R, G, nlp, cap),
              "swap_decide: out int64 of swap_decide_out_ints");
  TORCH_CHECK(R % L == 0 && L >= 2 && L <= swapdec::kMaxL && nlp >= 1 && nlp <= swapdec::kMaxNlp && cap >= 1,
              "swap_decide: 2 <= L <= 8, nlp <= 64");
  const int64_t NN = R / L, Q = NN * (L / 2);
  const size_t smem = sizeof(long long) * (2 * G + S + R) + sizeof(int) * (3 * S + 7 * Q + NN);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem > 48 * 1024) {
    SWAPDEC_CUDA_CHECK(
        cudaFuncSetAttribute(swapdec::swap_decide_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  }
  swapdec::swap_decide_kernel<<<1, swapdec::kThreads, smem, stream>>>(
      loads.data_ptr<int>(), reinterpret_cast<const long long *>(ntok.data_ptr<int64_t>()), S_b, (int)K,
      reinterpret_cast<long long *>(p2l.data_ptr<int64_t>()), l2p.data_ptr<int>(), lcnts.data_ptr<int>(), (int)R,
      (int)G, (int)L, (int)nlp, C, (int)cap, (int)max_rounds, reinterpret_cast<long long *>(out.data_ptr<int64_t>()),
      reinterpret_cast<const long long *>(verdict_armed()));
  SWAPDEC_CUDA_CHECK(cudaGetLastError());
}

// kernel registry (include/flux/cuda/kernel_registry.h): every kernel of this file
void swap_decide_kernels(KernelList &out) {
  out.push_back(ZEPP_KERNEL_ENTRY(swapdec::swap_decide_kernel));
}

}  // namespace bytedance::flux
