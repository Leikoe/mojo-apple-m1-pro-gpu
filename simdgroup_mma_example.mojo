"""Simdgroup 8x8 MMA on M1/M2/M3/M4 — working example.

Build with:
    mojo build --target-accelerator apple-m1-metal4 simdgroup_mma_example.mojo -o gemm
    ./gemm

Three workarounds required:
  1. external_call instead of llvm_intrinsic (the backend only lowers 16x16x16)
  2. address_space_cast[AddressSpace.GLOBAL]() for load/store pointers
  3. --target-accelerator apple-m1-metal4 (needs AIR 2.8.0, default emits 2.7.0)
"""

from std.sys import exit, has_accelerator
from std.sys.info import is_apple_gpu
from std.memory import alloc, AddressSpace
from std.ffi import external_call
from std.gpu import block_idx
from std.gpu.globals import WARP_SIZE
from std.gpu.host import DeviceContext


# ---- simdgroup 8x8 wrappers ------------------------------------------------


@always_inline
def simdgroup_load[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], ...], stride: Int) -> SIMD[dtype, 64]:
    comptime s = "f16" if dtype == DType.float16 else "f32"
    return external_call[
        "air.simdgroup_matrix_8x8_load.v64" + s + ".p1" + s,
        SIMD[dtype, 64],
    ](
        ptr.address_space_cast[AddressSpace.GLOBAL](),
        SIMD[DType.int64, 2](Int64(stride), Int64(8)),
        SIMD[DType.int64, 2](Int64(1), Int64(stride)),
        SIMD[DType.int64, 2](Int64(0), Int64(0)),
    )


@always_inline
def simdgroup_store[
    dtype: DType
](val: SIMD[dtype, 64], ptr: UnsafePointer[Scalar[dtype], ...], stride: Int):
    comptime s = "f16" if dtype == DType.float16 else "f32"
    external_call[
        "air.simdgroup_matrix_8x8_store.v64" + s + ".p1" + s,
        NoneType,
    ](
        val,
        ptr.address_space_cast[AddressSpace.GLOBAL](),
        SIMD[DType.int64, 2](Int64(stride), Int64(8)),
        SIMD[DType.int64, 2](Int64(1), Int64(stride)),
        SIMD[DType.int64, 2](Int64(0), Int64(0)),
    )


@always_inline
def simdgroup_multiply_accumulate[
    in_dtype: DType, acc_dtype: DType
](
    a: SIMD[in_dtype, 64], b: SIMD[in_dtype, 64], acc: SIMD[acc_dtype, 64]
) -> SIMD[acc_dtype, 64]:
    comptime i = "f16" if in_dtype == DType.float16 else "f32"
    comptime o = "f16" if acc_dtype == DType.float16 else "f32"
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate"
        + ".v64" + o + ".v64" + i + ".v64" + i + ".v64" + o,
        SIMD[acc_dtype, 64],
    ](a, b, acc)


# ---- 8x8-tiled GEMM kernel -------------------------------------------------

comptime M = 128
comptime N = 128
comptime K = 128

def gemm_kernel(
    a_ptr: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    comptime assert is_apple_gpu()
    var ti = Int(block_idx.x)
    var tj = Int(block_idx.y)
    var acc = SIMD[DType.float32, 64](0)
    for kt in range(0, K, 8):
        var a = simdgroup_load(a_ptr + ti * 8 * K + kt, K)
        var b = simdgroup_load(b_ptr + kt * N + tj * 8, N)
        acc = simdgroup_multiply_accumulate(a, b, acc)
    simdgroup_store(acc, c_ptr + ti * 8 * N + tj * 8, N)


# ---- host -------------------------------------------------------------------

def main() raises:
    if not has_accelerator():
        print("no gpu")
        exit()

    var a = alloc[Scalar[DType.float16]](M * K)
    var b = alloc[Scalar[DType.float16]](K * N)
    for i in range(M * K):
        a[i] = Scalar[DType.float16](Float32(i % 7) - 3.0)
    for i in range(K * N):
        b[i] = Scalar[DType.float16](Float32(i % 5) - 2.0)

    with DeviceContext() as ctx:
        var da = ctx.enqueue_create_buffer[DType.float16](M * K)
        var db = ctx.enqueue_create_buffer[DType.float16](K * N)
        var dc = ctx.enqueue_create_buffer[DType.float32](M * N)
        ctx.enqueue_copy(da, a)
        ctx.enqueue_copy(db, b)

        ctx.enqueue_function[gemm_kernel, gemm_kernel](
            da.unsafe_ptr(), db.unsafe_ptr(), dc.unsafe_ptr(),
            grid_dim=(M // 8, N // 8), block_dim=WARP_SIZE,
        )
        ctx.synchronize()

        with dc.map_to_host() as c:
            for i in range(M):
                for j in range(N):
                    var expected = Float32(0)
                    for k in range(K):
                        expected += Float32(a[i * K + k]) * Float32(
                            b[k * N + j]
                        )
                    var diff = expected - Float32(c[i * N + j])
                    if diff > 1.0 or diff < -1.0:
                        print(
                            "FAIL ({},{}) expected {} got {}".format(
                                i, j, expected, c[i * N + j]
                            )
                        )
                        return
            print("PASS — {} x {} hardware simdgroup MMA GEMM".format(M, N))
