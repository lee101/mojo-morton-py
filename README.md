# mojo-morton-py

`mojo-morton-py` is a Mojo acceleration layer for [morton-py](https://github.com/gojuno/morton-py), the small Python library for Morton (Z-order) curve packing. It preserves the upstream `morton.Morton` scalar API and adds NumPy batch operations for the encode/decode hot path.

## Install

This repository is self-contained; Pixi installs the pinned Mojo nightly, NumPy, pytest, and the released `morton-py` 1.3 package used by the parity suite.

```bash
pixi install
pixi run build
pixi run test
```

For a checkout, `PYTHONPATH=python` is set by Pixi. The first batch call will also build the shared library if it is missing or stale.

## Usage

The upstream API has the same import, constructor, and method signatures:

```python
from morton import Morton

m = Morton(dimensions=2, bits=32)
code = m.pack(13, 42)
assert code == 2265
assert m.unpack(code) == [13, 42]
assert m.sunpack(m.spack(-3, 7)) == [-3, 7]
```

For a workload of many points, use the native extensions:

```python
import numpy as np
from morton import Morton

m = Morton(dimensions=3, bits=21)
points = np.array([[1, 2, 4], [7, 8, 9]], dtype=np.uint64)
codes = m.pack_array(points)
assert np.array_equal(m.unpack_array(codes), points)
```

## Coverage

Covered upstream API: construction, `split`, `compact`, `shift_sign`, `unshift_sign`, `pack`, `unpack`, `spack`, `sunpack`, `repr`, and equality. The parity suite tests each listed operation against released `morton-py` 1.3, including Python-integer configurations wider than 64 bits.

The accelerated additions are `pack_array`, `unpack_array`, `split_array`, and `compact_array`. They accept integer inputs no wider than `uint64`, reject negative and floating-point inputs, reuse C-contiguous `uint64` input buffers without copying, convert other valid layouts and dtypes to a safe contiguous buffer, and return `uint64`. They support configurations with `dimensions * bits <= 64`. Signed batch packing and arbitrary-precision batch codes are intentionally not provided; use the compatible scalar API for those cases. Upstream has no batch API, spatial range-query helpers, or Hilbert curves, so none are claimed here.

## Benchmark

Measured with `pixi run bench` on this machine (Linux 6.8.0-136-generic, x86_64), using the best of three runs. The comparison is against released `morton-py` 1.3 in its natural equivalent form: a Python comprehension of scalar calls, as upstream has no array API.

| case | Mojo | morton-py 1.3 | ratio | result |
|---|---:|---:|---:|---|
| pack 250,000 x 2 (32-bit) | 25.5 ms | 1150.6 ms | 45.15x | faster |
| pack 250,000 x 3 (21-bit) | 17.9 ms | 1449.1 ms | 80.77x | faster |
| pack 200,000 x 4 (16-bit) | 13.9 ms | 1541.0 ms | 110.55x | faster |
| unpack 250,000 x 3 (21-bit) | 12.6 ms | 1067.6 ms | 85.00x | faster |
| split 250,000 (4D, 16-bit) | 4.0 ms | 323.9 ms | 81.42x | faster |
| compact 250,000 (4D, 16-bit) | 3.2 ms | 349.8 ms | 108.12x | faster |

Run the same guarded benchmark yourself with:

```bash
pixi run bench
```

Morton packing and unpacking are low-arithmetic-intensity bit-permutation kernels, so a GPU path is not provided: host/device transfer would lose to the CPU implementation.

Profiling covers all four native kernels. Each is already more than 5x faster than upstream, so the kernels deliberately remain serial scalar code: SIMD and thread-launch overhead were not added to paths outside the optimization target.

## How it works

`src/capi.mojo` is one compilation unit with four C-ABI kernels. Python passes addresses of contiguous `numpy.uint64` buffers through `ctypes`; Mojo reconstructs mutable pointers internally, writes caller-owned output buffers, and performs no allocation. Input points are row-major `(n, dimensions)` arrays, while packed codes are a contiguous `uint64` vector. Each native kernel interleaves or deinterleaves bits directly, preserving the same least-significant-dimension-first ordering as upstream.

`build/build.sh` emits `dist/libmojo-morton-py.so`. `python/morton/_lib.py` loads it and rebuilds only when the source is newer; set `MOJO_MORTON_LIB` to use a prebuilt library in a deployed environment.
