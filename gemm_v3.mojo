"""Tiled GEMM: 64x64 tiles, 8 simdgroups, async copy + double buffering.

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

comptime BM = 64
comptime BN = 64
comptime BK = 32
comptime NUM_SG = 8
comptime THREADS = NUM_SG * WARP_SIZE
comptime TILES_N = BN // 8


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
    external_call["air.wait_simdgroup_events", NoneType](
        Int32(1), event_slot
    )


# ---- kernel ----------------------------------------------------------------

comptime A_LAYOUT = row_major[M, K]()
comptime B_LAYOUT = row_major[K, N]()
comptime C_LAYOUT = row_major[M, N]()


def kernel(
    a: TileTensor[AB_DTYPE, type_of(A_LAYOUT), MutAnyOrigin],
    b: TileTensor[AB_DTYPE, type_of(B_LAYOUT), MutAnyOrigin],
    c: TileTensor[C_DTYPE, type_of(C_LAYOUT), MutAnyOrigin],
):
    comptime assert is_apple_gpu()

    var bi = Int(block_idx.x)
    var bj = Int(block_idx.y)
    var sg_id = Int(thread_idx.x) // WARP_SIZE

    # Double-buffered shared memory
    var sa0 = stack_allocation[AB_DTYPE, address_space=AddressSpace.SHARED](
        row_major[BM, BK]()
    )
    var sa1 = stack_allocation[AB_DTYPE, address_space=AddressSpace.SHARED](
        row_major[BM, BK]()
    )
    var sb0 = stack_allocation[AB_DTYPE, address_space=AddressSpace.SHARED](
        row_major[BK, BN]()
    )
    var sb1 = stack_allocation[AB_DTYPE, address_space=AddressSpace.SHARED](
        row_major[BK, BN]()
    )

    # Event slot for async wait
    var event_slot = raw_stack_alloc[1, Int64]()
    event_slot[0] = external_call["air.get_null_simdgroup_event", Int64]()

    # Accumulators
    var acc0 = SIMD[C_DTYPE, 2](0)
    var acc1 = SIMD[C_DTYPE, 2](0)
    var acc2 = SIMD[C_DTYPE, 2](0)
    var acc3 = SIMD[C_DTYPE, 2](0)
    var acc4 = SIMD[C_DTYPE, 2](0)
    var acc5 = SIMD[C_DTYPE, 2](0)
    var acc6 = SIMD[C_DTYPE, 2](0)
    var acc7 = SIMD[C_DTYPE, 2](0)

    var a_base = a.ptr + bi * BM * K
    var b_base = b.ptr + bj * BN

    # Prefetch first tile into buffer 0
    if sg_id == 0:
        event_slot[0] = _async_copy_2d(sa0.ptr, BK, a_base, K, BM, BK)
    if sg_id == 1:
        event_slot[0] = _async_copy_2d(sb0.ptr, BN, b_base, N, BK, BN)
    _wait_async(event_slot)
    barrier()

    comptime NUM_K_TILES = K // BK
    for kt_idx in range(NUM_K_TILES):
        var kt = kt_idx * BK

        # Pick current buffer
        var sa_cur = sa0.ptr if kt_idx % 2 == 0 else sa1.ptr
        var sb_cur = sb0.ptr if kt_idx % 2 == 0 else sb1.ptr

        # Async prefetch NEXT tile into the other buffer
        if kt_idx + 1 < NUM_K_TILES:
            var sa_next = sa1.ptr if kt_idx % 2 == 0 else sa0.ptr
            var sb_next = sb1.ptr if kt_idx % 2 == 0 else sb0.ptr
            var next_kt = kt + BK
            if sg_id == 0:
                event_slot[0] = _async_copy_2d(
                    sa_next, BK, a_base + next_kt, K, BM, BK
                )
            if sg_id == 1:
                event_slot[0] = _async_copy_2d(
                    sb_next, BN, b_base + next_kt * N, N, BK, BN
                )

        # Compute on current tile
        comptime for kk in range(BK // 8):
            var a_frag = simdgroup_load(
                sa_cur + sg_id * 8 * BK + kk * 8, BK
            )

            comptime for tj in range(TILES_N):
                var b_frag = simdgroup_load(
                    sb_cur + kk * 8 * BN + tj * 8, BN
                )
                if tj == 0:
                    acc0 = simdgroup_multiply_accumulate(a_frag, b_frag, acc0)
                elif tj == 1:
                    acc1 = simdgroup_multiply_accumulate(a_frag, b_frag, acc1)
                elif tj == 2:
                    acc2 = simdgroup_multiply_accumulate(a_frag, b_frag, acc2)
                elif tj == 3:
                    acc3 = simdgroup_multiply_accumulate(a_frag, b_frag, acc3)
                elif tj == 4:
                    acc4 = simdgroup_multiply_accumulate(a_frag, b_frag, acc4)
                elif tj == 5:
                    acc5 = simdgroup_multiply_accumulate(a_frag, b_frag, acc5)
                elif tj == 6:
                    acc6 = simdgroup_multiply_accumulate(a_frag, b_frag, acc6)
                else:
                    acc7 = simdgroup_multiply_accumulate(a_frag, b_frag, acc7)

        # Wait for prefetch before swapping buffers
        if kt_idx + 1 < NUM_K_TILES:
            _wait_async(event_slot)
        barrier()

    var c_base = c.ptr + (bi * BM + sg_id * 8) * N + bj * BN
    simdgroup_store(acc0, c_base + 0,  N)
    simdgroup_store(acc1, c_base + 8,  N)
    simdgroup_store(acc2, c_base + 16, N)
    simdgroup_store(acc3, c_base + 24, N)
    simdgroup_store(acc4, c_base + 32, N)
    simdgroup_store(acc5, c_base + 40, N)
    simdgroup_store(acc6, c_base + 48, N)
    simdgroup_store(acc7, c_base + 56, N)


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
            at, bt, ct,
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
                        expected += a_h[i, k].cast[C_DTYPE]() * b_h[
                            k, j
                        ].cast[C_DTYPE]()
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
                at, bt, ct,
                grid_dim=(M // BM, N // BN),
                block_dim=THREADS,
            )
            ctx.synchronize()

        var t0 = perf_counter_ns()
        for _ in range(100):
            ctx.enqueue_function[kernel, kernel](
                at, bt, ct,
                grid_dim=(M // BM, N // BN),
                block_dim=THREADS,
            )
            ctx.synchronize()
        var ns = perf_counter_ns() - t0

        var avg = ns / 100
        print(
            "PASS", M, "x", N, "|",
            FLOPS / Float64(avg), "GFLOPS |",
            Float64(avg) / 1e6, "ms",
        )
