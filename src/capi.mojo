"""Native batch Morton encode/decode kernels.

The exported ABI deliberately takes addresses as Int: an exported pointer
parameter would make the function parametric in its memory origin.
"""

comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]


@export("morton_pack_u64")
def morton_pack_u64(values_addr: Int, n: Int, dimensions: Int, bits: Int, codes_addr: Int) abi("C"):
    var values = UPtr(unsafe_from_address=values_addr)
    var codes = UPtr(unsafe_from_address=codes_addr)
    for row in range(n):
        var code: UInt64 = 0
        for bit in range(bits):
            for dim in range(dimensions):
                if ((values[row * dimensions + dim] >> UInt64(bit)) & UInt64(1)) != 0:
                    code |= UInt64(1) << UInt64(bit * dimensions + dim)
        codes[row] = code


@export("morton_unpack_u64")
def morton_unpack_u64(codes_addr: Int, n: Int, dimensions: Int, bits: Int, values_addr: Int) abi("C"):
    var codes = UPtr(unsafe_from_address=codes_addr)
    var values = UPtr(unsafe_from_address=values_addr)
    for row in range(n):
        var code = codes[row]
        for dim in range(dimensions):
            var value: UInt64 = 0
            for bit in range(bits):
                if ((code >> UInt64(bit * dimensions + dim)) & UInt64(1)) != 0:
                    value |= UInt64(1) << UInt64(bit)
            values[row * dimensions + dim] = value


@export("morton_split_u64")
def morton_split_u64(values_addr: Int, n: Int, dimensions: Int, bits: Int, result_addr: Int) abi("C"):
    var values = UPtr(unsafe_from_address=values_addr)
    var result = UPtr(unsafe_from_address=result_addr)
    for row in range(n):
        var code: UInt64 = 0
        for bit in range(bits):
            if ((values[row] >> UInt64(bit)) & UInt64(1)) != 0:
                code |= UInt64(1) << UInt64(bit * dimensions)
        result[row] = code


@export("morton_compact_u64")
def morton_compact_u64(codes_addr: Int, n: Int, dimensions: Int, bits: Int, result_addr: Int) abi("C"):
    var codes = UPtr(unsafe_from_address=codes_addr)
    var result = UPtr(unsafe_from_address=result_addr)
    for row in range(n):
        var value: UInt64 = 0
        for bit in range(bits):
            if ((codes[row] >> UInt64(bit * dimensions)) & UInt64(1)) != 0:
                value |= UInt64(1) << UInt64(bit)
        result[row] = value
