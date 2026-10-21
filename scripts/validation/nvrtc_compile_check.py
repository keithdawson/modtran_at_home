"""Compile the CUDA kernel with NVRTC -- no GPU needed.

    pip install nvidia-cuda-nvrtc-cu12
    python scripts/validation/nvrtc_compile_check.py

Catches syntax and type errors in ``mwir_atmos/optics/lbl_cuda.py``'s kernel
source for the RTX 4090 (sm_89) and Blackwell (sm_120) before a GPU run.
Correctness is covered separately by the kernel's NumPy twin
(``tests/test_lbl_cuda.py``).
"""
from __future__ import annotations

import ctypes
import glob
import os
import site
import sys

from mwir_atmos.optics import lbl_cuda

ARCHES = ("compute_89", "compute_120")


def _library() -> ctypes.CDLL:
    names = ("libnvrtc.so*", "nvrtc64_*.dll")
    # CUDA 12 wheels: nvidia/cuda_nvrtc/{lib,bin}; CUDA 13: nvidia/cu13/{lib,bin/x86_64}.
    folders = (("cuda_nvrtc", "lib"), ("cuda_nvrtc", "bin"),
               ("cu13", "lib"), ("cu13", "bin", "x86_64"))
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        for folder in folders:
            for name in names:
                found = sorted(glob.glob(os.path.join(root, "nvidia", *folder, name)))
                # Windows wheels also ship nvrtc64_*.alt.dll, which needs a
                # builtins DLL they do not include; it sorts first.
                found = [path for path in found if ".alt." not in os.path.basename(path)]
                if found:
                    if os.name == "nt":
                        # NVRTC loads its builtins DLL from the same folder at compile time.
                        folder_path = os.path.dirname(found[0])
                        os.add_dll_directory(folder_path)
                        os.environ["PATH"] = folder_path + os.pathsep + os.environ.get("PATH", "")
                    return ctypes.CDLL(found[0])
    raise SystemExit("NVRTC not found: pip install nvidia-cuda-nvrtc-cu12 (or the cuda13 extra)")


def main() -> int:
    lib = _library()
    source = lbl_cuda._source().encode()
    failures = 0
    for arch in ARCHES:
        program = ctypes.c_void_p()
        lib.nvrtcCreateProgram(ctypes.byref(program), source, b"voigt.cu", 0, None, None)
        options = (ctypes.c_char_p * 2)(f"--gpu-architecture={arch}".encode(), b"--std=c++14")
        status = lib.nvrtcCompileProgram(program, 2, options)
        size = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        lib.nvrtcGetProgramLog(program, log)
        print(f"{arch}: {'ok' if status == 0 else f'FAILED ({status})'}")
        if log.value.strip():
            print(log.value.decode())
        failures += status != 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
