"""The GPU block-scheduler head-of-line rule: every kernel that can START while a spinning kernel is resident
must fit beside one block of it on an SM.

The work distributor places the kernels of one priority in launch order, and a block that fits on no SM holds
back every later kernel of that priority, including the wire kernels a spinning GEMM waits for: a deadlock.
The dispatch GEMM block is 128 threads x 240 registers (30.7K of an SM's 64K), the combine GEMM block 128 x
254 (32.8K); their blocks spread over every SM, so a kernel launched beside them must keep its block at
<= 32768 registers (32-thread warps, registers per thread rounded up to a multiple of 8, i.e. 256 per warp)
and <= 100 KB of shared memory (static + the most dynamic shared memory its launch site can request).

The listed kernels are those launched between the dispatch GEMM's launch and the end of the combine on a
stream other than the GEMM's own (or behind a spinning kernel): the overlapped combine derive (beside the
dispatch GEMM), the combine's pack / pre-reduce / receivers (beside the combine GEMM and each other), the
one-warp stream waits and the swap lane's kernels. Threads per block come from the launch sites (cited).
Kernels ordered on the GEMM's own stream before it starts (metadata arena, demands, workspace builders, swap
decision, routing) are printed for reference and not bound by the rule. NVSHMEM's own kernels
(libnvshmem_host.so; their registers are printed for reference) and torch's kernels are outside this library.

CPU only:  python tests/test_r5_resources.py
"""
import re
import subprocess
from pathlib import Path

from test_kernel_registry import LIB, demangle, key, short_name, tool

REG_BUDGET = 32768
SMEM_BUDGET = 100 * 1024

# short name -> (threads per block, max dynamic shared memory bytes, launch site)
R5_BESIDE_SPINNER = {
    # overlapped combine derive (side stream, while the dispatch GEMM spins: plan_overlap 2)
    "a2av_combine_tables_kernel": (128, 96 * 1024, "combine_kernels.cu a2av_combine_tables: kCtThreads, "
                                                   "smem FLUX_CHECK_LE 96 KiB"),
    "compress_plan_token_kernel": (256, 0, "combine_kernels.cu a2av_compress_plan"),
    "compress_plan_scan_kernel": (256, 0, "combine_kernels.cu a2av_compress_plan"),
    "compress_plan_conv_kernel": (256, 0, "combine_kernels.cu a2av_compress_plan"),
    "compress_plan_red_kernel": (256, 0, "combine_kernels.cu a2av_compress_plan"),
    "combine_plan_pack_kernel": (256, 0, "combine_kernels.cu a2av_combine_plan"),
    "combine_plan_reduce_kernel": (256, 0, "combine_kernels.cu a2av_combine_plan"),
    "a2av_invert_index_kernel": (256, 0, "combine_kernels.cu a2av_invert_index"),
    # the msplit tables beside the dispatch GEMM (set_prep_fork; one block)
    "a2av_msplit_tables_kernel": (128, 0, "combine_kernels.cu a2av_msplit_tables: kCpThreads"),
    # combine pack / pre-reduce / receivers (beside the combine GEMM and the resident pre-reduce)
    "a2av_combine_pack_kernel<__nv_bfloat16, true>": (512, 0, "combine_kernels.cu a2av_combine_pack: kThreads (<= 64 regs)"),
    "a2av_combine_pack_kernel<__nv_bfloat16, false>": (512, 0, "combine_kernels.cu a2av_combine_pack: kThreads (<= 64 regs)"),
    "a2av_combine_pack_kernel<__half, true>": (512, 0, "combine_kernels.cu a2av_combine_pack: kThreads (<= 64 regs)"),
    "a2av_combine_pack_kernel<__half, false>": (512, 0, "combine_kernels.cu a2av_combine_pack: kThreads (<= 64 regs)"),
    "a2av_combine_prereduce_kernel<__nv_bfloat16>": (512, 0, "combine_kernels.cu a2av_combine_prereduce"),
    "a2av_combine_prereduce_kernel<__half>": (512, 0, "combine_kernels.cu a2av_combine_prereduce"),
    "a2av_combine_reduce_kernel<__nv_bfloat16>": (512, 0, "combine_kernels.cu a2av_combine_reduce"),
    "a2av_combine_reduce_kernel<__half>": (512, 0, "combine_kernels.cu a2av_combine_reduce"),
    "a2av_bucket_map_kernel": (256, 0, "combine_kernels.cu a2av_bucket_map"),
    "a2av_bucket_scan_kernel": (32, 0, "combine_kernels.cu a2av_bucket_scan"),
    "a2av_bucket_scatter_kernel": (256, 0, "combine_kernels.cu a2av_bucket_scatter"),
    "a2av_combine_bucket_reduce_kernel<__nv_bfloat16>": (512, 0, "combine_kernels.cu a2av_combine_bucket_reduce"),
    "a2av_combine_bucket_reduce_kernel<__half>": (512, 0, "combine_kernels.cu a2av_combine_bucket_reduce"),
    # one-warp stream wait (swap lane, joins)
    "wait_geq_kernel": (32, 0, "cuda_common.cu stream_wait_geq"),
    # swap lane on the device
    "lane_arm_kernel": (32, 0, "lane_device.cu lane_arm"),
    "lane_push_kernel": (512, 0, "lane_device.cu lane_push: kCopyThreads"),
    "lane_commit_kernel": (512, 0, "lane_device.cu lane_commit: kCopyThreads"),
    "dual3_push_w1_kernel": (512, 0, "lane_device.cu dual3_push_w1: kCopyThreads (beside the dispatch GEMM)"),
    "dual3_pull_w2_kernel": (512, 0, "lane_device.cu dual3_pull_w2: kCopyThreads (beside the combine GEMM)"),
    "dual3_wait_kernel": (32, 0, "lane_device.cu dual3_wait: one warp"),
    "pad_rebuild_kernel": (256, 0, "lane_device.cu pad_rebuild"),
}

# ordered before the GEMM on its own stream (reference only; not bound by the rule)
ORDERED_BEFORE_GEMM = {
    "a2av_meta_arena_kernel": (512, "sort_util.cu a2av_meta_arena_impl (before GEMM 1)"),
    "a2av_demands_kernel": (512, "sort_util.cu a2av_demands_impl (planning)"),
    "prepare_workspace_kernel<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, "
    "cutlass::layout::RowMajor>": (768, "workspace_util.cu prepare_workspace (before GEMM 1)"),
    "make_workspace_kernel<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, "
    "cutlass::layout::RowMajor>": (768, "workspace_helper.cu make_workspace (before GEMM 2, after GEMM 1)"),
    "swap_decide_kernel": (256, "swap_decide.cu (planning)"),
    "a2av_fold_scales_kernel<__nv_bfloat16>": (256, "combine_kernels.cu a2av_fold_scales (deferred verdict, "
                                                    "before GEMM 2)"),
    "route_tables_kernel": (128, "routing.cu (planning)"),
}


def resources(lib=LIB):
    """{key: (demangled name, registers, static shared bytes)} from cuobjdump --dump-resource-usage."""
    out = subprocess.run([tool("cuobjdump", "CUOBJDUMP"), "--dump-resource-usage", str(lib)], capture_output=True,
                         text=True, check=True).stdout
    rows = re.findall(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:\d+ SHARED:(\d+)", out)
    assert rows, f"no resource usage parsed from {lib}"
    names = demangle([r[0] for r in rows])
    return {key(n): (n, int(r[1]), int(r[2])) for n, r in zip(names, rows)}


def block_registers(threads, regs):
    warps = (threads + 31) // 32
    return warps * 32 * (-(-regs // 8) * 8)


def test_r5_fit():
    res = resources()
    bad, missing = [], []
    print(f"{'kernel':52s} {'thr':>5s} {'reg':>4s} {'blockreg':>8s} {'smem':>7s}  launch site")
    for name, (threads, dyn, site) in R5_BESIDE_SPINNER.items():
        k = "".join(name.split())
        if k not in res:
            missing.append(name)
            continue
        _full, regs, smem = res[k]
        breg = block_registers(threads, regs)
        tot_smem = smem + dyn
        ok = breg <= REG_BUDGET and tot_smem <= SMEM_BUDGET
        print(f"{short_name(name)[:52]:52s} {threads:5d} {regs:4d} {breg:8d} {tot_smem:7d}  {site}"
              + ("" if ok else "   <-- over budget"))
        if not ok:
            bad.append((name, breg, tot_smem))
    print("ordered before the GEMM on its stream (reference):")
    for name, (threads, site) in ORDERED_BEFORE_GEMM.items():
        k = "".join(name.split())
        if k in res:
            _full, regs, smem = res[k]
            print(f"  {short_name(name)[:50]:50s} {threads:5d} {regs:4d} {block_registers(threads, regs):8d} "
                  f"{smem:7d}  {site}")
    assert not missing, f"listed kernels not found in {LIB.name} (renamed? update the table): {missing}"
    assert not bad, f"kernels launched beside a spinning kernel exceed the block budget: {bad}"


def test_nvshmem_reference():
    """Print the NVSHMEM on-stream kernels' registers (their launch geometry is internal to NVSHMEM)."""
    import os
    home = os.environ.get("NVSHMEM_HOME")
    lib = Path(home) / "lib" / "libnvshmem_host.so" if home else None
    if lib is None or not lib.exists():
        print("[nvshmem] NVSHMEM_HOME not set: NVSHMEM kernels not listed")
        return
    res = resources(lib)
    for k, (full, regs, smem) in sorted(res.items()):
        if any(t in full for t in ("proxy_rma", "signal_op_kernel", "proxy_quiet", "barrier_on_stream")):
            print(f"[nvshmem] {short_name(full)[:70]:70s} reg {regs:3d} smem {smem}")


if __name__ == "__main__":
    test_r5_fit()
    test_nvshmem_reference()
    print("PASS")
