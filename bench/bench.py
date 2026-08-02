"""Batch kernel benchmark against the released morton-py implementation."""

from __future__ import annotations

import importlib.util
import math
import platform
import sysconfig
import time
from pathlib import Path

import numpy as np

from morton import Morton


def upstream_morton():
    spec = importlib.util.spec_from_file_location(
        "upstream_morton", Path(sysconfig.get_paths()["purelib"]) / "morton.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Morton


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    print(f"machine: {platform.platform()} ({platform.processor() or 'unknown CPU'})")
    print("| case | Mojo | morton-py 1.3 | ratio | result |")
    print("|---|---:|---:|---:|---|")
    rng = np.random.default_rng(2026)
    for dimensions, bits, n in [(2, 32, 250_000), (3, 21, 250_000), (4, 16, 200_000)]:
        values = rng.integers(0, 1 << bits, size=(n, dimensions), dtype=np.uint64)
        ours, theirs = Morton(dimensions, bits), upstream_morton()(dimensions, bits)
        ours.pack_array(values)
        native = timeit(lambda: ours.pack_array(values))
        python = timeit(lambda: [theirs.pack(*map(int, row)) for row in values])
        result = "faster" if native < python else "slower"
        print(f"| pack {n:,} x {dimensions} ({bits}-bit) | {native * 1e3:.1f} ms | {python * 1e3:.1f} ms | {python / native:.2f}x | {result} |")

    dimensions, bits, n = 3, 21, 250_000
    values = rng.integers(0, 1 << bits, size=(n, dimensions), dtype=np.uint64)
    ours, theirs = Morton(dimensions, bits), upstream_morton()(dimensions, bits)
    codes = ours.pack_array(values)
    native = timeit(lambda: ours.unpack_array(codes))
    python = timeit(lambda: [theirs.unpack(int(code)) for code in codes])
    result = "faster" if native < python else "slower"
    print(f"| unpack {n:,} x {dimensions} ({bits}-bit) | {native * 1e3:.1f} ms | {python * 1e3:.1f} ms | {python / native:.2f}x | {result} |")


if __name__ == "__main__":
    main()
