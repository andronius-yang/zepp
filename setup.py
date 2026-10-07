"""Builds the `zepp._C` extension against the CMake-built libzepp_cuda.so."""
import os
from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
NVSHMEM_HOME = Path(os.environ["NVSHMEM_HOME"])

ext = CppExtension(
    name="zepp._C",
    sources=sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src" / "pybind").glob("*.cc")),
    include_dirs=[
        str(ROOT / "include"),
        str(ROOT / "src"),
        str(ROOT / "3rdparty/cutlass/include"),
        str(ROOT / "3rdparty/cutlass/tools/util/include"),
        str(CUDA_HOME / "include"),
        str(NVSHMEM_HOME / "include"),
    ],
    library_dirs=[
        str(ROOT / "build/lib"),
        str(CUDA_HOME / "lib64"),
        str(CUDA_HOME / "lib64/stubs"),
        str(NVSHMEM_HOME / "lib"),
    ],
    libraries=["zepp_cuda", "cuda", "cudart", "nvidia-ml", "nvshmem_host"],
    extra_compile_args=["-O3", "-DTORCH_CUDA=1", "-DFLUX_SHM_USE_NVSHMEM",
                        "-fvisibility=hidden", "-Wno-deprecated-declarations"],
)

setuptools.setup(
    name="zepp",
    version="0.1.0",
    description="Expert-parallel MoE layer with fused, overlapped dispatch/combine",
    package_dir={"": "python"},
    packages=["zepp"],
    package_data={"zepp": ["lib/*.so"]},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch", "numpy"],
    license_files=("LICENSE", "NOTICE"),
)
