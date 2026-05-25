"""Simdgroup 8x8 matrix operations for Apple GPUs (M1–M4).

Provides load, store, and multiply-accumulate for 8x8 simdgroup matrices.
Each thread in the 32-thread simdgroup holds 2 elements as SIMD[T, 2].

Load/store support both device (global) and threadgroup (shared) memory.
Typical usage: software-load a large tile into shared memory, then
hardware-load 8x8 sub-tiles from shared into registers for MMA.

Requirements:
  - Apple GPU target (M1 or later)
  - Build with --target-accelerator apple-<chip>-metal4
"""

from std.sys.info import CompilationTarget, is_apple_gpu
from std.memory import AddressSpace
from std.ffi import external_call
from std.gpu import lane_id


# ---- thread-to-element layout -----------------------------------------------


@always_inline
def _simdgroup_8x8_frag_layout(tid: Int) -> Tuple[Int, Int]:
    """Returns (row, col_base) for the given simdgroup thread.

    Each thread owns elements (row, col_base) and (row, col_base+1).
    The 8x8 tile is split into four 4x4 quadrants:
      t0-7:   rows 0-3, cols 0-3
      t8-15:  rows 0-3, cols 4-7
      t16-23: rows 4-7, cols 0-3
      t24-31: rows 4-7, cols 4-7
    """
    return (
        ((tid & 7) >> 1) + ((tid & 16) >> 2),
        ((tid & 1) << 1) + ((tid & 8) >> 1),
    )


# ---- dtype / addrspace helpers ----------------------------------------------


@always_inline
def _air_dtype_suffix[dtype: DType]() -> String:
    comptime assert dtype in (
        DType.float16,
        DType.bfloat16,
        DType.float32,
    ), "simdgroup_matrix: unsupported dtype (need float16, bfloat16, or float32)"
    if dtype == DType.float16:
        return "f16"
    elif dtype == DType.bfloat16:
        return "bf16"
    else:
        return "f32"


@always_inline
def _air_addrspace_code[addr: AddressSpace]() -> String:
    comptime assert addr in (AddressSpace.GLOBAL, AddressSpace.SHARED), (
        "simdgroup load/store: pointer must be in GLOBAL (device) or SHARED (threadgroup) address space"
    )
    if addr == AddressSpace.GLOBAL:
        return "1"
    else:
        return "3"


# ---- hardware load/store (AIR intrinsics) -----------------------------------


@always_inline
def _sg_hw_load[
    dtype: DType, addr: AddressSpace
](
    ptr: UnsafePointer[Scalar[dtype], address_space=addr, ...],
    elements_per_row: Int,
) -> SIMD[dtype, 64]:
    comptime s = _air_dtype_suffix[dtype]()
    comptime a = _air_addrspace_code[addr]()
    return external_call[
        "air.simdgroup_matrix_8x8_load.v64" + s + ".p" + a + s,
        SIMD[dtype, 64],
    ](
        ptr,
        SIMD[DType.int64, 2](Int64(elements_per_row), Int64(8)),
        SIMD[DType.int64, 2](Int64(1), Int64(elements_per_row)),
        SIMD[DType.int64, 2](Int64(0), Int64(0)),
    )


@always_inline
def _sg_hw_store[
    dtype: DType, addr: AddressSpace
](
    val: SIMD[dtype, 64],
    ptr: UnsafePointer[Scalar[dtype], address_space=addr, ...],
    elements_per_row: Int,
):
    comptime s = _air_dtype_suffix[dtype]()
    comptime a = _air_addrspace_code[addr]()
    external_call[
        "air.simdgroup_matrix_8x8_store.v64" + s + ".p" + a + s,
        NoneType,
    ](
        val,
        ptr,
        SIMD[DType.int64, 2](Int64(elements_per_row), Int64(8)),
        SIMD[DType.int64, 2](Int64(1), Int64(elements_per_row)),
        SIMD[DType.int64, 2](Int64(0), Int64(0)),
    )


# ---- public API: load -------------------------------------------------------


@always_inline
def simdgroup_load[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], ...], elements_per_row: Int) -> SIMD[
    dtype, 2
]:
    """Hardware-load this thread's 2 elements from an 8x8 tile.

    Works with both device (global) and threadgroup (shared) memory.

    Parameters:
        dtype: Element type (float16, bfloat16, or float32).

    Args:
        ptr: Pointer to the top-left element of the 8x8 tile.
        elements_per_row: Stride between consecutive rows (in elements).

    Returns:
        SIMD[dtype, 2] holding this thread's two matrix elements.
    """
    comptime assert is_apple_gpu() and CompilationTarget.is_apple_silicon(), (
        "simdgroup_load requires Apple Silicon GPU"
    )
    comptime resolved = AddressSpace.GLOBAL if ptr.address_space == AddressSpace.GENERIC else ptr.address_space
    var wide = _sg_hw_load(
        ptr.address_space_cast[resolved](), elements_per_row
    )
    return SIMD[dtype, 2](wide[0], wide[1])


# ---- public API: store -------------------------------------------------------


@always_inline
def simdgroup_store[
    dtype: DType
](
    frag: SIMD[dtype, 2],
    ptr: UnsafePointer[Scalar[dtype], ...],
    elements_per_row: Int,
):
    """Hardware-store this thread's 2 elements to an 8x8 tile.

    Works with both device (global) and threadgroup (shared) memory.

    Parameters:
        dtype: Element type (float16, bfloat16, or float32).

    Args:
        frag: SIMD[dtype, 2] holding this thread's two matrix elements.
        ptr: Pointer to the top-left element of the 8x8 tile.
        elements_per_row: Stride between consecutive rows (in elements).
    """
    comptime assert is_apple_gpu() and CompilationTarget.is_apple_silicon(), (
        "simdgroup_store requires Apple Silicon GPU"
    )
    comptime resolved = AddressSpace.GLOBAL if ptr.address_space == AddressSpace.GENERIC else ptr.address_space
    var wide = SIMD[dtype, 64](0)
    wide[0] = frag[0]
    wide[1] = frag[1]
    _sg_hw_store(wide, ptr.address_space_cast[resolved](), elements_per_row)


# ---- public API: multiply-accumulate ----------------------------------------


@always_inline
def simdgroup_multiply_accumulate[
    in_dtype: DType, acc_dtype: DType
](
    a: SIMD[in_dtype, 2], b: SIMD[in_dtype, 2], acc: SIMD[acc_dtype, 2]
) -> SIMD[acc_dtype, 2]:
    """Multiply two 8x8 matrices and accumulate: acc = a * b + acc.

    Each thread provides its 2-element fragment. The simdgroup cooperatively
    performs the full 8x8 matrix multiply across all 32 threads.

    Parameters:
        in_dtype: Input element type (float16, bfloat16, or float32).
        acc_dtype: Accumulator type (float32).

    Args:
        a: This thread's 2 elements of the left matrix.
        b: This thread's 2 elements of the right matrix.
        acc: This thread's 2 elements of the accumulator.

    Returns:
        Updated 2-element accumulator fragment.
    """
    comptime assert is_apple_gpu() and CompilationTarget.is_apple_silicon(), (
        "simdgroup_multiply_accumulate requires Apple Silicon GPU"
    )
    comptime assert acc_dtype == DType.float32, (
        "simdgroup_multiply_accumulate: accumulator must be float32"
    )
    comptime i = _air_dtype_suffix[in_dtype]()
    comptime o = _air_dtype_suffix[acc_dtype]()

    var a64 = SIMD[in_dtype, 64](0)
    var b64 = SIMD[in_dtype, 64](0)
    var c64 = SIMD[acc_dtype, 64](0)
    a64[0] = a[0]
    a64[1] = a[1]
    b64[0] = b[0]
    b64[1] = b[1]
    c64[0] = acc[0]
    c64[1] = acc[1]

    var r = external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate"
        + ".v64" + o + ".v64" + i + ".v64" + i + ".v64" + o,
        SIMD[acc_dtype, 64],
    ](a64, b64, c64)

    return SIMD[acc_dtype, 2](r[0], r[1])
