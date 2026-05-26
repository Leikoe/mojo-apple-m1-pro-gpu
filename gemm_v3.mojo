"""Tiled GEMM: MPS-shaped 128x128 tiles, 16 simdgroups.

Run:
    mojo run --target-accelerator apple-m1-metal4 gemm_v3.mojo
Build:
    mojo build --target-accelerator apple-m1-metal4 gemm_v3.mojo -o gemm_v3
"""

from std.sys import exit, has_accelerator
from std.sys.info import is_apple_gpu
from std.memory import alloc, AddressSpace
from std.memory import stack_allocation as raw_stack_alloc
from std.random import randn, seed
from std.time import perf_counter_ns
from std.ffi import external_call

from std.gpu import block_idx, thread_idx, barrier
from std.gpu.globals import WARP_SIZE
from std.gpu.host import DeviceContext

from layout import TileTensor, stack_allocation, row_major

from simdgroup_matrix import (
    simdgroup_load,
    simdgroup_store,
    simdgroup_multiply_accumulate,
)


comptime AB_DTYPE = DType.float16
comptime C_DTYPE = DType.float32

comptime M = 1024
comptime N = 1024
comptime K = 1024

comptime BM = 128
comptime BN = 128
comptime BK = 32
comptime SG_M = 4
comptime SG_N = 4
comptime NUM_SG = SG_M * SG_N
comptime THREADS = NUM_SG * WARP_SIZE
comptime ACC_M = BM // (SG_M * 8)
comptime ACC_N = BN // (SG_N * 8)
comptime NUM_ACC = ACC_M * ACC_N


# ---- async copy helpers ----------------------------------------------------


@always_inline
def _async_copy_2d[
    dtype: DType
](
    dst: UnsafePointer[Scalar[dtype], ...],
    dst_stride: Int,
    src: UnsafePointer[Scalar[dtype], ...],
    src_stride: Int,
    rows: Int,
    cols: Int,
) -> Int64:
    comptime elem_size = 2 if dtype in (DType.float16, DType.bfloat16) else 4
    return external_call[
        "air.simdgroup_async_copy_2d.p3i8.p1i8",
        Int64,
    ](
        Int64(elem_size),
        Int64(elem_size),
        dst.address_space_cast[AddressSpace.SHARED]().bitcast[Int8](),
        Int64(dst_stride),
        Int64(1),
        SIMD[DType.int64, 2](Int64(cols), Int64(rows)),
        src.address_space_cast[AddressSpace.GLOBAL]().bitcast[Int8](),
        Int64(src_stride),
        Int64(1),
        SIMD[DType.int64, 2](Int64(cols), Int64(rows)),
        SIMD[DType.int64, 2](Int64(0), Int64(0)),
        Int32(0),
    )


@always_inline
def _wait_async(event_slot: UnsafePointer[Int64, ...]):
    external_call["air.wait_simdgroup_events", NoneType](Int32(1), event_slot)


# ---- kernel ----------------------------------------------------------------

comptime A_LAYOUT = row_major[M, K]()
comptime B_LAYOUT = row_major[K, N]()
comptime C_LAYOUT = row_major[M, N]()
comptime A_SMEM_LAYOUT = row_major[BM, BK]()
comptime B_SMEM_LAYOUT = row_major[BK, BN]()


def kernel(
    a: TileTensor[AB_DTYPE, type_of(A_LAYOUT), MutAnyOrigin],
    b: TileTensor[AB_DTYPE, type_of(B_LAYOUT), MutAnyOrigin],
    c: TileTensor[C_DTYPE, type_of(C_LAYOUT), MutAnyOrigin],
):
    comptime assert is_apple_gpu()

    var bi = Int(block_idx.x)
    var bj = Int(block_idx.y)
    var sg_id = Int(thread_idx.x) // WARP_SIZE
    var sg_m = sg_id // SG_N
    var sg_n = sg_id % SG_N

    var sa = stack_allocation[AB_DTYPE, address_space=AddressSpace.SHARED](
        A_SMEM_LAYOUT
    )
    var sb = stack_allocation[AB_DTYPE, address_space=AddressSpace.SHARED](
        B_SMEM_LAYOUT
    )

    var event_slots = raw_stack_alloc[2, Int64]()
    event_slots[0] = external_call["air.get_null_simdgroup_event", Int64]()
    event_slots[1] = event_slots[0]

    var acc = InlineArray[SIMD[C_DTYPE, 2], NUM_ACC](fill=SIMD[C_DTYPE, 2](0))

    var a_base = a.ptr + bi * BM * K
    var b_base = b.ptr + bj * BN

    comptime NUM_K_TILES = K // BK

    for kt_idx in range(NUM_K_TILES):
        var kt = kt_idx * BK
        event_slots[0] = _async_copy_2d(sa.ptr, BK, a_base + kt, K, BM, BK)
        event_slots[1] = _async_copy_2d(sb.ptr, BN, b_base + kt * N, N, BK, BN)
        _wait_async(event_slots)
        _wait_async(event_slots + 1)
        barrier()

        comptime for kk in range(BK // 8):
            comptime for mi in range(ACC_M):
                var a_row = (sg_m + mi * SG_M) * 8
                var a_frag = simdgroup_load(sa.ptr + a_row * BK + kk * 8, BK)

                comptime for nj in range(ACC_N):
                    var b_col = (sg_n + nj * SG_N) * 8
                    var b_frag = simdgroup_load(
                        sb.ptr + kk * 8 * BN + b_col, BN
                    )
                    comptime acc_idx = mi * ACC_N + nj
                    acc[acc_idx] = simdgroup_multiply_accumulate(
                        a_frag, b_frag, acc[acc_idx]
                    )
        barrier()

    var c_base = c.ptr + bi * BM * N + bj * BN
    comptime for mi in range(ACC_M):
        var c_row = (sg_m + mi * SG_M) * 8
        comptime for nj in range(ACC_N):
            var c_col = (sg_n + nj * SG_N) * 8
            comptime acc_idx = mi * ACC_N + nj
            simdgroup_store(acc[acc_idx], c_base + c_row * N + c_col, N)


def main() raises:
    seed()
    if not has_accelerator():
        exit()

    var a_host = alloc[Scalar[AB_DTYPE]](M * K)
    var b_host = alloc[Scalar[AB_DTYPE]](K * N)
    randn[AB_DTYPE](a_host, M * K, mean=0.0, standard_deviation=1.0)
    randn[AB_DTYPE](b_host, K * N, mean=0.0, standard_deviation=1.0)

    var a_h = TileTensor(a_host, A_LAYOUT)
    var b_h = TileTensor(b_host, B_LAYOUT)

    comptime FLOPS = 2 * M * N * K

    with DeviceContext() as ctx:
        var ab = ctx.enqueue_create_buffer[AB_DTYPE](M * K)
        var bb = ctx.enqueue_create_buffer[AB_DTYPE](K * N)
        var cb = ctx.enqueue_create_buffer[C_DTYPE](M * N)
        ctx.enqueue_copy(ab, a_host)
        ctx.enqueue_copy(bb, b_host)

        var at = TileTensor(ab, A_LAYOUT)
        var bt = TileTensor(bb, B_LAYOUT)
        var ct = TileTensor(cb, C_LAYOUT)

        ctx.enqueue_function[kernel, kernel](
            at,
            bt,
            ct,
            grid_dim=(M // BM, N // BN),
            block_dim=THREADS,
        )
        ctx.synchronize()

        with cb.map_to_host() as rb:
            var r = TileTensor(rb, C_LAYOUT)
            for i in range(M):
                for j in range(N):
                    var expected: Scalar[C_DTYPE] = 0
                    for k in range(K):
                        expected += (
                            a_h[i, k].cast[C_DTYPE]()
                            * b_h[k, j].cast[C_DTYPE]()
                        )
                    var diff = expected - r[i, j]
                    if diff > 0.1 or diff < -0.1:
                        print(
                            "FAIL ({},{}) {} != {}".format(
                                i, j, expected, r[i, j]
                            )
                        )
                        return

        for _ in range(5):
            ctx.enqueue_function[kernel, kernel](
                at,
                bt,
                ct,
                grid_dim=(M // BM, N // BN),
                block_dim=THREADS,
            )
            ctx.synchronize()

        var t0 = perf_counter_ns()
        for _ in range(100):
            ctx.enqueue_function[kernel, kernel](
                at,
                bt,
                ct,
                grid_dim=(M // BM, N // BN),
                block_dim=THREADS,
            )
            ctx.synchronize()
        var ns = perf_counter_ns() - t0

        var avg = ns / 100
        print(
            "PASS",
            M,
            "x",
            N,
            "|",
            FLOPS / Float64(avg),
            "GFLOPS |",
            Float64(avg) / 1e6,
            "ms",
        )
