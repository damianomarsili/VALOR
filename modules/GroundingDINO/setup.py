# coding=utf-8
# Copyright 2022 The IDEA Authors.
#
# Vendored + patched for VALOR.

import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CppExtension,
    CUDA_HOME,
)

# ───────────────────────────────────────────────────────────────
# 1.  Metadata
# ───────────────────────────────────────────────────────────────

PKG = "groundingdino"
VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent

# Keep the git SHA if .git exists; otherwise mark as unknown
if (ROOT / ".git").exists():
    try:
        SHA = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        SHA = "Unknown"
else:
    SHA = "Unknown"


def write_version():
    """
    Write a simple groundingdino/version.py so the package can
    introspect its own version.
    """
    version_py = ROOT / PKG / "version.py"
    version_py.write_text(f"__version__ = '{VERSION}'\n", encoding="utf-8")


# ───────────────────────────────────────────────────────────────
# 2.  Torch guard (no pip here!)
# ───────────────────────────────────────────────────────────────


def require_torch():
    """
    Ensure torch is importable. We do NOT try to install it here
    (that breaks uv / isolated builds), we only give a clear error.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GroundingDINO requires PyTorch to be installed in the environment "
            "before building. Please add an appropriate `torch` version to your "
            "project dependencies, run your resolver (e.g. `uv sync`), and then "
            "install GroundingDINO again."
        ) from exc


# ───────────────────────────────────────────────────────────────
# 3.  Patch the CUDA kernels for newer PyTorch
# ───────────────────────────────────────────────────────────────


def patch_ms_deform_attn():
    """
    Edit the old MsDeformAttn CUDA kernels in-place so they compile
    on newer PyTorch (e.g. 2.6, 2.7) and CUDA 12.x.

    This is idempotent: re-running won't re-patch already-patched files.
    """
    kernel_dir = ROOT / PKG / "models" / "GroundingDINO" / "csrc" / "MsDeformAttn"
    if not kernel_dir.is_dir():
        return

    for cu in kernel_dir.glob("*.cu"):
        original = cu.read_text(encoding="utf-8")
        src = original

        # 3-a) .type().is_cuda() ➜ .is_cuda()
        src = re.sub(r"\.type\(\)\.is_cuda\(\)", ".is_cuda()", src)
        # In case some earlier patch left scalar_type().is_cuda()
        src = re.sub(r"\.scalar_type\(\)\.is_cuda\(\)", ".is_cuda()", src)

        # 3-b) dispatch macro: value.type() ➜ value.scalar_type()
        src = re.sub(
            r"AT_DISPATCH_FLOATING_TYPES\s*\(\s*value\.type\(\)\s*,",
            "AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(),",
            src,
        )

        if src != original:
            cu.write_text(src, encoding="utf-8")
            print(f"[GroundingDINO setup] Patched {cu.relative_to(ROOT)}")


# ───────────────────────────────────────────────────────────────
# 4.  Extension builder
# ───────────────────────────────────────────────────────────────


def get_extensions():
    """
    Build the GroundingDINO CUDA extension if CUDA + torch are available.
    Returns a list of Extension instances, or [] if we fall back to CPU.
    """
    # Make sure torch is importable (and present) before we try to
    # look at CUDA / headers.
    require_torch()
    import torch  # now safe

    patch_ms_deform_attn()

    ext_root = ROOT / PKG / "models" / "GroundingDINO" / "csrc"
    sources = {str(p) for p in ext_root.rglob("*.cpp")}
    cuda_sources = [str(p) for p in ext_root.rglob("*.cu")]

    sources = list(sources)

    macros = []
    extra_compile_args = {"cxx": []}

    # Decide whether to build CUDA or not
    if CUDA_HOME and (
        torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ
    ):
        print("[GroundingDINO setup] Compiling with CUDA")
        ext_type = CUDAExtension
        sources = sources + cuda_sources
        macros.append(("WITH_CUDA", None))
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            "-std=c++17",
        ]
    else:
        # You can either:
        #  - return [] to skip the extension (CPU-only install), or
        #  - still build a pure-CPU CppExtension if that is supported.
        print(
            "[GroundingDINO setup] CUDA not available; installing without C++ extension"
        )
        return []

    return [
        ext_type(
            f"{PKG}._C",
            sources,
            include_dirs=[str(ext_root)],
            define_macros=macros,
            extra_compile_args=extra_compile_args,
        )
    ]


# ───────────────────────────────────────────────────────────────
# 5.  Requirements parser (from original repo)
# ───────────────────────────────────────────────────────────────


def parse_reqs(fname="requirements.txt", with_version=True):
    """
    Parse a requirements.txt while preserving git / -e style lines.
    """
    from os.path import exists

    if not exists(fname):
        return []

    def _lines(path):
        with open(path) as f:
            for ln in map(str.strip, f):
                if ln and not ln.startswith("#"):
                    yield ln

    pkgs = []
    for line in _lines(fname):
        if line.startswith("-r "):
            # include other requirement files
            pkgs.extend(parse_reqs(line.split(" ", 1)[1], with_version))
            continue

        if "@git+" in line or line.startswith("-e "):
            pkg = line
        else:
            parts = re.split(r"(>=|==|>|<=|!=)", line, maxsplit=1)
            pkg = parts[0].strip()
            if with_version and len(parts) > 1:
                pkg += "".join(parts[1:]).strip()
        pkgs.append(pkg)

    return pkgs


# ───────────────────────────────────────────────────────────────
# 6.  setup()
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[GroundingDINO setup] Building wheel {PKG}-{VERSION} (SHA: {SHA})")
    write_version()

    setup(
        name=PKG,
        version=VERSION,
        author="International Digital Economy Academy, Shilong Liu",
        url="https://github.com/IDEA-Research/GroundingDINO",
        description="open-set object detector",
        license=(ROOT / "LICENSE").read_text(encoding="utf-8"),
        install_requires=parse_reqs("requirements.txt"),
        packages=find_packages(exclude=("configs", "tests")),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
