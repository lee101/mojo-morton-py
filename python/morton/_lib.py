"""Build and load the Mojo batch kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_MORTON_LIB") or os.path.join(ROOT, "dist", "libmojo-morton-py.so")
I = ctypes.c_int64
_SIGNATURES = {
    "morton_pack_u64": ([I, I, I, I, I], None),
    "morton_unpack_u64": ([I, I, I, I, I], None),
    "morton_split_u64": ([I, I, I, I, I], None),
    "morton_compact_u64": ([I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_MORTON_MOJO")
    if override:
        return override.split()
    if found := shutil.which("mojo"):
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_MORTON_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if os.environ.get("MOJO_MORTON_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        _mojo_command() + ["build", "--emit", "shared-lib", source, "-o", LIB],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array) -> int:
    return array.ctypes.data
