"""The kernel registry equals the kernel list of libzepp_cuda.so (the lazy-loading guard).

Under CUDA_MODULE_LOADING=LAZY a kernel's code loads at its first launch, and while a load runs with a kernel
resident the GPU takes no new work from any thread; the fused ops' GEMM tiles spin on work other threads and
ranks issue, so every kernel the library may launch is loaded before serving (C.preload_all(), the kernel
registry of include/flux/cuda/kernel_registry.h, one list per source file). This test keeps the registry
complete: every entry function of the library (`cuobjdump -symbols`, demangled with c++filt) must be in the
registry, and the registry must name nothing the library lacks.

CPU only by default (the names come from C.kernel_registry_names(), which makes no CUDA call). With --gpu
(one GPU) it also loads every registered kernel and checks the count.

    python tests/test_kernel_registry.py [--gpu]
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import zepp
from zepp._ext import C

LIB = Path(zepp.__file__).resolve().parent / "lib" / "libzepp_cuda.so"


def tool(name, env_var=None):
    """A CUDA binary utility: $<env_var>, $CUDA_HOME/bin/<name>, or the PATH."""
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin" / name).exists():
        return str(Path(home) / "bin" / name)
    found = shutil.which(name)
    assert found, f"{name} not found (set CUDA_HOME or {env_var})"
    return found


def demangle(symbols):
    out = subprocess.run([shutil.which("c++filt") or "c++filt"], input="\n".join(symbols), capture_output=True,
                         text=True, check=True).stdout.splitlines()
    assert len(out) == len(symbols)
    return out


def short_name(demangled):
    """'void ns::(anonymous namespace)::k<T, true>(Args)' -> 'k<T, true>': no return type, parameter list or
    namespace qualifiers (template arguments are kept as printed)."""
    s = demangled.strip()
    if s.startswith("void "):
        s = s[len("void "):]
    if s.endswith(")"):                                 # the trailing parameter list
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == ")":
                depth += 1
            elif s[i] == "(":
                depth -= 1
                if depth == 0:
                    s = s[:i]
                    break
    angle = paren = cut = 0
    i = 0
    while i < len(s):                                   # the last '::' outside <...> and (...)
        c = s[i]
        if c == "<":
            angle += 1
        elif c == ">":
            angle -= 1
        elif c == "(":
            paren += 1
        elif c == ")":
            paren -= 1
        elif c == ":" and s.startswith("::", i) and angle == 0 and paren == 0:
            cut = i + 2
            i += 2
            continue
        i += 1
    return s[cut:].strip()


def key(name):
    return "".join(short_name(name).split())


def library_kernels(lib=LIB):
    """Entry functions (kernels) of every fatbin in the library: {key: demangled name}."""
    out = subprocess.run([tool("cuobjdump", "CUOBJDUMP"), "-symbols", str(lib)], capture_output=True, text=True,
                         check=True).stdout
    mangled = [line.split()[-1] for line in out.splitlines() if "STT_FUNC" in line and "STO_ENTRY" in line]
    assert mangled, f"no kernel entry symbols found in {lib}"
    return {key(d): d for d in demangle(mangled)}


def registry_kernels():
    return {key(n): n for n in C.kernel_registry_names()}


def test_registry_equals_library():
    lib, reg = library_kernels(), registry_kernels()
    missing = sorted(set(lib) - set(reg))
    stale = sorted(set(reg) - set(lib))
    print(f"[kernel registry] library {len(lib)} kernels, registry {len(reg)} names "
          f"({len(C.kernel_registry_names())} entries)")
    assert not missing, ("kernels in libzepp_cuda.so missing from the registry (add them to the list function "
                         "of their source file, include/flux/cuda/kernel_registry.h):\n  "
                         + "\n  ".join(lib[k] for k in missing))
    assert not stale, "registry names with no kernel in libzepp_cuda.so:\n  " + "\n  ".join(reg[k] for k in stale)


def test_short_name():
    cases = {
        "void bytedance::flux::(anonymous namespace)::a2av_combine_pack_kernel<__nv_bfloat16, true>"
        "(bytedance::flux::CombinePackArguments)": "a2av_combine_pack_kernel<__nv_bfloat16, true>",
        "bytedance::flux::wait_geq_kernel(unsigned long long const*, unsigned long long)": "wait_geq_kernel",
        "void cutlass::Kernel<flux_x>(flux_x::Params)": "Kernel<flux_x>",
        "void barrier_on_stream_kernel_threadgroup<(threadgroup_t)1>(int, int)":
            "barrier_on_stream_kernel_threadgroup<(threadgroup_t)1>",
        "router::route_tables_kernel": "route_tables_kernel",
        "void bytedance::flux::prepare_workspace_kernel<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>"
        "(bytedance::flux::DispatchGemmArguments, int, int, int, void*)":
            "prepare_workspace_kernel<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>",
    }
    for full, want in cases.items():
        assert short_name(full) == want, (full, short_name(full), want)


def test_preload_gpu(gpu=False):
    """One GPU (opt-in: --gpu): every registered kernel loads (cudaFuncGetAttributes), and a second call is a
    no-op."""
    if not gpu:
        print("[kernel registry] GPU check skipped (--gpu)")
        return
    import torch
    torch.cuda.init()
    names = C.kernel_registry_names()
    n = int(C.preload_all())
    assert n == len(names), (n, len(names))
    assert int(C.preload_all()) == n
    print(f"[kernel registry] preload_all loaded {n} kernels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also load every kernel on the current GPU")
    a = ap.parse_args()
    test_short_name()
    test_registry_equals_library()
    test_preload_gpu(a.gpu)
    print("PASS")
