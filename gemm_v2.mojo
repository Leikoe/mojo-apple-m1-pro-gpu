from std.sys import exit, has_accelerator
from std.sys.info import is_apple_gpu
from std.memory import alloc
from std.random import randn, seed
from std.time import perf_counter_ns

from std.gpu import block_idx
from std.gpu.globals import WARP_SIZE
from std.gpu.host import DeviceContext

from layout import TileTensor
from layout.tile_layout import row_major

from simdgroup_matrix import (
    simdgroup_load,
    simdgroup_store,
    simdgroup_multiply_accumulate,
)


comptime AB_DTYPE = DType.float16
comptime C_DTYPE = DType.float32
comptime TILE = 8

comptime M = 1024
comptime N = 1024
comptime K = 1024

comptime A_LAYOUT = row_major[M, K]()
comptime B_LAYOUT = row_major[K, N]()
comptime C_LAYOUT = row_major[M, N]()


def kernel(
    a: TileTensor[AB_DTYPE, type_of(A_LAYOUT), MutAnyOrigin],
    b: TileTensor[AB_DTYPE, type_of(B_LAYOUT), MutAnyOrigin],
    c: TileTensor[C_DTYPE, type_of(C_LAYOUT), MutAnyOrigin],
):
    comptime assert is_apple_gpu()

    var ti = Int(block_idx.x)
    var tj = Int(block_idx.y)

    var acc = SIMD[C_DTYPE, 2](0)

    for kt in range(0, K, TILE):
        var a_frag = simdgroup_load(a.ptr + ti * TILE * K + kt, K)
        var b_frag = simdgroup_load(b.ptr + kt * N + tj * TILE, N)
        acc = simdgroup_multiply_accumulate(a_frag, b_frag, acc)

    simdgroup_store(acc, c.ptr + ti * TILE * N + tj * TILE, N)


def main() raises:
    seed()

    if not has_accelerator():
        print("gpu is required")
        exit()

    var a_host_buf = alloc[Scalar[AB_DTYPE]](M * K)
    var b_host_buf = alloc[Scalar[AB_DTYPE]](K * N)

    randn[AB_DTYPE](a_host_buf, M * K, mean=0.0, standard_deviation=1.0)
    randn[AB_DTYPE](b_host_buf, K * N, mean=0.0, standard_deviation=1.0)

    var a_host = TileTensor(a_host_buf, A_LAYOUT)
    var b_host = TileTensor(b_host_buf, B_LAYOUT)

    comptime FLOPS_PER_GEMM = 2 * M * N * K
    comptime WARMUP = 5
    comptime ITERS = 100

    with DeviceContext() as ctx:
        var a_buf = ctx.enqueue_create_buffer[AB_DTYPE](M * K)
        var b_buf = ctx.enqueue_create_buffer[AB_DTYPE](K * N)
        var c_buf = ctx.enqueue_create_buffer[C_DTYPE](M * N)

        ctx.enqueue_copy(a_buf, a_host_buf)
        ctx.enqueue_copy(b_buf, b_host_buf)

        var a = TileTensor(a_buf, A_LAYOUT)
        var b = TileTensor(b_buf, B_LAYOUT)
        var c = TileTensor(c_buf, C_LAYOUT)

        ctx.enqueue_function[kernel, kernel](
            a, b, c,
            grid_dim=(M // TILE, N // TILE),
            block_dim=WARP_SIZE,
        )
        ctx.synchronize()

        with c_buf.map_to_host() as result_buf:
            var result = TileTensor(result_buf, C_LAYOUT)
            for i in range(M):
                for j in range(N):
                    var expected: Scalar[C_DTYPE] = 0
                    for k in range(K):
                        expected += a_host[i, k].cast[C_DTYPE]() * b_host[
                            k, j
                        ].cast[C_DTYPE]()
                    var diff = expected - result[i, j]
                    if diff > 0.1 or diff < -0.1:
                        print(
                            "FAIL at ({}, {}): expected {} got {}".format(
                                i, j, expected, result[i, j]
                            )
                        )
                        return

        for _ in range(WARMUP):
            ctx.enqueue_function[kernel, kernel](
                a, b, c,
                grid_dim=(M // TILE, N // TILE),
                block_dim=WARP_SIZE,
            )
            ctx.synchronize()

        var t0 = perf_counter_ns()
        for _ in range(ITERS):
            ctx.enqueue_function[kernel, kernel](
                a, b, c,
                grid_dim=(M // TILE, N // TILE),
                block_dim=WARP_SIZE,
            )
            ctx.synchronize()
        var elapsed_ns = perf_counter_ns() - t0

        var avg_ns = elapsed_ns / ITERS
        var gflops = FLOPS_PER_GEMM / Float64(avg_ns)
        var avg_ms = Float64(avg_ns) / 1e6
        print(
            "PASS",
            M, "x", N, "|",
            gflops, "GFLOPS |",
            avg_ms, "ms avg |",
            ITERS, "iters",
        )
