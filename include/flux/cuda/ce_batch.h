// Copy-engine batch issue with host-known sizes.
//
// Many intra-node copies become ONE cudaMemcpyBatchAsync (CUDA 12.8+; a loop of cudaMemcpyAsync on
// older toolchains), and each consumer signal becomes a front-end stream write (cuStreamWriteValue64)
// at the peer-mapped slot, ordered after the batch on the same stream. That is the data-before-signal
// order NVSHMEM's intra-node put_signal lowers to (a copy, then an 8-byte copy of the signal), with one
// host call per batch instead of per copy and no device kernel: the copies run on the copy engines
// whatever the SMs are doing (the batch asks the driver to prefer that). Inter-node puts keep the
// blocking NVSHMEM path.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"

namespace bytedance::flux {

// Device-to-device copies (local or peer-mapped destinations) issued as one batch.
struct CeBatch {
  std::vector<void *> dst;
  std::vector<void *> src;
  std::vector<size_t> bytes;

  void
  add(void *d, const void *s, size_t n) {
    if (n == 0) {
      return;
    }
    dst.push_back(d);
    src.push_back(const_cast<void *>(s));
    bytes.push_back(n);
  }
  bool
  empty() const {
    return dst.empty();
  }
  void
  issue(cudaStream_t stream) {
    if (dst.empty()) {
      return;
    }
#if CUDART_VERSION >= 12080
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    // a hint: copy engines rather than SM copy kernels, so the batch overlaps the spinning GEMM
    attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attr_idx = 0, fail_idx = 0;
    CUDA_CHECK(cudaMemcpyBatchAsync(
        dst.data(), src.data(), bytes.data(), dst.size(), &attr, &attr_idx, 1, &fail_idx, stream));
#else
    for (size_t i = 0; i < dst.size(); i++) {
      CUDA_CHECK(cudaMemcpyAsync(dst[i], src[i], bytes[i], cudaMemcpyDeviceToDevice, stream));
    }
#endif
    dst.clear();
    src.clear();
    bytes.clear();
  }
};

// Peer-mapped address of a symmetric address at pe (the local address for my pe); null when the peer
// is not reachable through a P2P mapping (then the caller keeps the NVSHMEM call). One nvshmem_ptr
// call: for init / resize time only, the per-step issue uses a PeerTable.
inline void *
ce_peer(const void *sym, int pe) {
  return nvshmem_ptr(const_cast<void *>(sym), pe);
}

// Peer-mapped bases of every symmetric buffer an op's wire targets, resolved once per pe at op init
// and after every resize, so a per-step lookup is host arithmetic: no NVSHMEM call.
class PeerTable {
 public:
  void
  clear() {
    bufs_.clear();
  }
  void
  add(const void *base, size_t bytes, int npes) {
    Buf b;
    b.base = static_cast<const char *>(base);
    b.bytes = bytes;
    b.peer.resize(npes);
    for (int pe = 0; pe < npes; pe++) {
      b.peer[pe] = static_cast<char *>(ce_peer(base, pe));
    }
    bufs_.push_back(std::move(b));
  }
  // null: not P2P-mapped at pe
  void *
  peer(const void *sym, int pe) const {
    const char *p = static_cast<const char *>(sym);
    for (const Buf &b : bufs_) {
      if (p >= b.base && p < b.base + b.bytes && pe >= 0 && pe < (int)b.peer.size()) {
        return b.peer[pe] == nullptr ? nullptr : b.peer[pe] + (p - b.base);
      }
    }
    // not registered: correct but slow (an NVSHMEM call); report once
    static bool warned = false;
    if (!warned) {
      warned = true;
      fprintf(stderr, "[zepp] peer table: symmetric address %p not registered, using nvshmem_ptr\n", sym);
    }
    return ce_peer(sym, pe);
  }

 private:
  struct Buf {
    const char *base = nullptr;
    size_t bytes = 0;
    std::vector<char *> peer;
  };
  std::vector<Buf> bufs_;
};

// Pending SET signals of a batch: issued (in order) after the batch's copies. Each carries its
// peer-mapped slot resolved when it is added (PeerTable); a null slot keeps the NVSHMEM signal call.
struct CeSignals {
  struct Sig {
    uint64_t *slot;  // symmetric slot
    void *peer;      // peer-mapped slot at pe, or null
    int pe;
    uint64_t val;
  };
  std::vector<Sig> sigs;

  void
  add(uint64_t *sym_sig, void *peer_slot, int pe, uint64_t val) {
    sigs.push_back({sym_sig, peer_slot, pe, val});
  }
  void
  issue(cudaStream_t stream) {
    for (auto &s : sigs) {
      if (s.peer != nullptr) {
        CU_CHECK(CUStreamWriteValue64(
            (CUstream)stream, reinterpret_cast<CUdeviceptr>(s.peer), s.val,
            CU_STREAM_WRITE_VALUE_DEFAULT));
      } else {
        nvshmemx_signal_op_on_stream(s.slot, s.val, NVSHMEM_SIGNAL_SET, s.pe, stream);
      }
    }
    sigs.clear();
  }
};

}  // namespace bytedance::flux
