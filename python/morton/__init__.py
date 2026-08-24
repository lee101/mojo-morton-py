"""Morton/Z-order packing with native Mojo batch kernels.

``Morton`` mirrors morton-py 1.3.  The ``*_array`` methods are extensions for
the workloads where crossing into native code is worthwhile.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, build, lib

__version__ = "0.1.0"


class Morton:
    def __init__(self, dimensions=2, bits=32):
        assert dimensions > 0, "dimensions should be greater than zero"
        assert bits > 0, "bits should be greater than zero"

        def flp2(x):
            x |= x >> 1
            x |= x >> 2
            x |= x >> 4
            x |= x >> 8
            x |= x >> 16
            x |= x >> 32
            x -= x >> 1
            return x

        shift = flp2(dimensions * (bits - 1))
        masks, lshifts = [], []
        max_value = (1 << (shift * bits)) - 1
        while shift > 0:
            mask = shifted = 0
            for bit in range(bits):
                distance = (dimensions * bit) - bit
                shifted |= shift & distance
                mask |= 1 << bit << (((shift - 1) ^ max_value) & distance)
            if shifted:
                masks.append(mask)
                lshifts.append(shift)
            shift >>= 1

        self.dimensions = dimensions
        self.bits = bits
        self.lshifts = [0] + lshifts
        self.rshifts = lshifts + [0]
        self.masks = [(1 << bits) - 1] + masks
        self._size = bits * dimensions

    def __repr__(self):
        return "<Morton dimensions={}, bits={}>".format(self.dimensions, self.bits)

    def __eq__(self, other):
        return self.dimensions == other.dimensions and self.bits == other.bits

    def split(self, value):
        for mask, shift in zip(self.masks, self.lshifts):
            value = (value | (value << shift)) & mask
        return value

    def compact(self, code):
        for mask, shift in zip(reversed(self.masks), reversed(self.rshifts)):
            code = (code | (code >> shift)) & mask
        return code

    def shift_sign(self, value):
        assert not (value >= (1 << (self.bits - 1)) or value <= -(1 << (self.bits - 1))), (value, self.bits)
        if value < 0:
            value = -value
            value |= 1 << (self.bits - 1)
        return value

    def unshift_sign(self, value):
        sign = value & (1 << (self.bits - 1))
        value &= (1 << (self.bits - 1)) - 1
        return -value if sign else value

    def pack(self, *args):
        assert len(args) <= self.dimensions
        assert all((value < (1 << self.bits)) and (value >= 0) for value in args)
        code = 0
        for index in range(self.dimensions):
            code |= self.split(args[index]) << index
        return code

    def unpack(self, code):
        return [self.compact(code >> index) for index in range(self.dimensions)]

    def spack(self, *args):
        code = self.pack(*map(self.shift_sign, args))
        return code if code < ((1 << self._size - 1) - 1) else code - (1 << self._size)

    def sunpack(self, code):
        return [self.unshift_sign(value) for value in self.unpack(code)]

    @property
    def _native(self) -> bool:
        return self._size <= 64

    def _check_native(self):
        if not self._native:
            raise ValueError("batch kernels support dimensions * bits <= 64")

    @staticmethod
    def _u64_input(values, name):
        """Validate input and return a C-contiguous uint64 FFI buffer."""
        source = np.asarray(values)
        if source.dtype.kind not in "ui" or source.dtype.itemsize > np.dtype(np.uint64).itemsize:
            raise TypeError(f"{name} must be an integer array with a dtype no wider than uint64")
        if source.dtype.kind == "i" and np.any(source < 0):
            raise ValueError(f"{name} must be non-negative")
        # NumPy returns the original object when it already has the required layout.
        return np.ascontiguousarray(source, dtype=np.uint64)

    def pack_array(self, values):
        """Pack an ``(n, dimensions)`` unsigned-integer array into uint64 codes."""
        self._check_native()
        source = np.asarray(values)
        if source.ndim != 2 or source.shape[1] != self.dimensions:
            raise ValueError("values must have shape (n, dimensions)")
        values = self._u64_input(source, "values")
        if np.any(values >= (np.uint64(1) << np.uint64(self.bits)) if self.bits < 64 else False):
            raise ValueError("values must fit in bits")
        codes = np.empty(values.shape[0], dtype=np.uint64)
        lib().morton_pack_u64(addr(values), values.shape[0], self.dimensions, self.bits, addr(codes))
        return codes

    def unpack_array(self, codes):
        """Unpack uint64 Morton codes into an ``(n, dimensions)`` array."""
        self._check_native()
        codes = self._u64_input(codes, "codes").reshape(-1)
        values = np.empty((codes.size, self.dimensions), dtype=np.uint64)
        lib().morton_unpack_u64(addr(codes), codes.size, self.dimensions, self.bits, addr(values))
        return values

    def split_array(self, values):
        self._check_native()
        values = self._u64_input(values, "values").reshape(-1)
        if np.any(values >= (np.uint64(1) << np.uint64(self.bits)) if self.bits < 64 else False):
            raise ValueError("values must fit in bits")
        result = np.empty_like(values)
        lib().morton_split_u64(addr(values), values.size, self.dimensions, self.bits, addr(result))
        return result

    def compact_array(self, codes):
        self._check_native()
        codes = self._u64_input(codes, "codes").reshape(-1)
        result = np.empty_like(codes)
        lib().morton_compact_u64(addr(codes), codes.size, self.dimensions, self.bits, addr(result))
        return result


__all__ = ["Morton", "build"]
