"""Behavioral parity with the released morton-py 1.3 module."""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path

import numpy as np
import pytest

from morton import Morton


def _upstream_morton():
    path = Path(sysconfig.get_paths()["purelib"]) / "morton.py"
    assert path.exists(), "morton-py must be installed for parity tests"
    spec = importlib.util.spec_from_file_location("upstream_morton", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Morton


UpstreamMorton = _upstream_morton()


@pytest.mark.parametrize("dimensions,bits", [(2, 32), (3, 21), (4, 16), (6, 10), (64, 1), (2, 64)])
def test_scalar_roundtrip_and_api_parity(dimensions, bits):
    ours, upstream = Morton(dimensions, bits), UpstreamMorton(dimensions, bits)
    values = [((index * 17) + 3) & ((1 << bits) - 1) for index in range(dimensions)]
    assert repr(ours) == repr(upstream)
    assert (ours.dimensions, ours.bits, ours.lshifts, ours.rshifts, ours.masks) == (
        upstream.dimensions, upstream.bits, upstream.lshifts, upstream.rshifts, upstream.masks,
    )
    assert ours.pack(*values) == upstream.pack(*values)
    assert ours.unpack(ours.pack(*values)) == upstream.unpack(upstream.pack(*values)) == values
    assert [ours.split(value) for value in values] == [upstream.split(value) for value in values]
    code = ours.pack(*values)
    assert [ours.compact(code >> index) for index in range(dimensions)] == [
        upstream.compact(code >> index) for index in range(dimensions)
    ]


@pytest.mark.parametrize("dimensions,bits", [(2, 32), (4, 16), (3, 21)])
def test_signed_api_parity(dimensions, bits):
    ours, upstream = Morton(dimensions, bits), UpstreamMorton(dimensions, bits)
    limit = (1 << (bits - 1)) - 1
    values = [limit if index % 2 == 0 else -(index + 1) for index in range(dimensions)]
    assert [ours.shift_sign(value) for value in values] == [upstream.shift_sign(value) for value in values]
    assert ours.spack(*values) == upstream.spack(*values)
    assert ours.sunpack(ours.spack(*values)) == upstream.sunpack(upstream.spack(*values)) == values


def test_native_batch_pack_unpack_matches_upstream():
    rng = np.random.default_rng(42)
    ours, upstream = Morton(3, 21), UpstreamMorton(3, 21)
    values = rng.integers(0, 1 << 21, size=(4096, 3), dtype=np.uint64)
    expected = np.array([upstream.pack(*map(int, row)) for row in values], dtype=np.uint64)
    actual = ours.pack_array(values)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(ours.unpack_array(actual), values)
    expected_values = np.array([upstream.unpack(int(code)) for code in actual], dtype=np.uint64)
    np.testing.assert_array_equal(ours.unpack_array(actual), expected_values)


def test_native_batch_split_compact_matches_upstream():
    rng = np.random.default_rng(7)
    ours, upstream = Morton(4, 16), UpstreamMorton(4, 16)
    values = rng.integers(0, 1 << 16, size=8192, dtype=np.uint64)
    split = ours.split_array(values)
    expected_split = np.array([upstream.split(int(value)) for value in values], dtype=np.uint64)
    np.testing.assert_array_equal(split, expected_split)
    np.testing.assert_array_equal(ours.compact_array(split), values)
    np.testing.assert_array_equal(ours.compact_array(split), np.array(
        [upstream.compact(int(code)) for code in split], dtype=np.uint64
    ))


def test_batch_validation_and_wide_fallback():
    native = Morton(2, 32)
    contiguous = np.arange(12, dtype=np.uint64).reshape(6, 2)
    assert native._u64_input(contiguous, "values") is contiguous
    with pytest.raises(ValueError, match="shape"):
        native.pack_array(np.zeros((3, 3), dtype=np.uint64))
    with pytest.raises(ValueError, match="non-negative"):
        native.pack_array(np.array([[-1, 0]], dtype=np.int64))
    with pytest.raises(TypeError, match="integer"):
        native.pack_array(np.zeros((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="fit"):
        native.pack_array(np.array([[1 << 32, 0]], dtype=np.uint64))
    with pytest.raises(TypeError, match="integer"):
        native.unpack_array(np.array([1.5]))
    with pytest.raises(ValueError, match="non-negative"):
        native.unpack_array(np.array([-1], dtype=np.int64))
    with pytest.raises(TypeError, match="integer"):
        native.split_array(np.array([1.5]))
    with pytest.raises(ValueError, match="fit"):
        native.split_array(np.array([1 << 32], dtype=np.uint64))
    with pytest.raises(TypeError, match="integer"):
        native.compact_array(np.array([1.5]))
    with pytest.raises(ValueError, match="non-negative"):
        native.compact_array(np.array([-1], dtype=np.int64))
    # Non-contiguous input is copied into a safe uint64 buffer passed over FFI.
    points = np.arange(24, dtype=np.uint32).reshape(6, 4)[:, ::2]
    converted = native._u64_input(points, "values")
    assert converted.flags.c_contiguous and converted.dtype == np.uint64
    assert not np.shares_memory(converted, points)
    np.testing.assert_array_equal(native.unpack_array(native.pack_array(points)), points)
    empty = native.pack_array(np.empty((0, 2), dtype=np.uint64))
    assert empty.dtype == np.uint64 and empty.shape == (0,)
    wide = Morton(2, 64)
    assert wide.unpack(wide.pack((1 << 64) - 1, 7)) == [(1 << 64) - 1, 7]
    with pytest.raises(ValueError, match="<= 64"):
        wide.pack_array(np.zeros((1, 2), dtype=np.uint64))


def test_constructor_and_assertion_parity():
    for dimensions, bits in [(0, 32), (2, 0)]:
        with pytest.raises(AssertionError):
            Morton(dimensions, bits)
    ours, upstream = Morton(2, 3), UpstreamMorton(2, 3)
    with pytest.raises(AssertionError):
        ours.pack(8, 0)
    with pytest.raises(AssertionError):
        upstream.pack(8, 0)
    assert ours == Morton(2, 3)
    assert not (ours == Morton(3, 3))
