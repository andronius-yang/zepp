"""Loads the compiled extension. NVSHMEM host libraries must be resolved before the
extension is imported, so they are preloaded from NVSHMEM_HOME."""
import ctypes
import importlib
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _preload(name: str, hint: Path | None = None) -> None:
    for d in ([hint] if hint else []) + [_HERE / "lib"]:
        try:
            ctypes.CDLL(str(d / name))
            return
        except OSError:
            continue
    ctypes.CDLL(name)  # fall back to LD_LIBRARY_PATH


def _load():
    nvshmem_home = os.environ.get("NVSHMEM_HOME")
    if nvshmem_home is None:
        raise RuntimeError("NVSHMEM_HOME is not set; source env/setup.sh")
    lib = Path(nvshmem_home) / "lib"
    _preload("libnvshmem_host.so.3", lib)
    _preload("nvshmem_bootstrap_uid.so.3", lib)
    _preload("libzepp_cuda.so")
    return importlib.import_module("zepp._C")


C = _load()

_SHM_GROUP = None


def ensure_shm(group) -> None:
    """Initialise NVSHMEM over `group` once (the symmetric heap and the bootstrap are process-wide;
    every EPMoE instance shares them). Collective: every rank of the group must call it."""
    global _SHM_GROUP
    if _SHM_GROUP is None:
        C.init_shm(group)
        _SHM_GROUP = group
    elif _SHM_GROUP is not group:
        raise RuntimeError("NVSHMEM is already initialised over a different process group")
