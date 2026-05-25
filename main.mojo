from std.sys import exit, has_accelerator
from std.sys.info import is_apple_gpu
from std.memory import alloc
from std.math import ceildiv
from std.random import randn, seed

from std.gpu import thread_idx, global_idx
from std.gpu.host import DeviceContext

from layout import TileTensor, stack_allocation
from layout.tile_layout import row_major


comptime AB_DTYPE = DType.bfloat16
comptime C_DTYPE = DType.float32


comptime M = 1024
comptime N = 1024
comptime K = 1024

comptime A_LAYOUT = row_major[M, K]()
comptime B_LAYOUT = row_major[N, K]()
comptime C_LAYOUT = row_major[M, N]()


def kernel(
    a: TileTensor[AB_DTYPE, type_of(A_LAYOUT), MutAnyOrigin],
    b: TileTensor[AB_DTYPE, type_of(B_LAYOUT), MutAnyOrigin],
    c: TileTensor[C_DTYPE, type_of(C_LAYOUT), MutAnyOrigin],
):
    comptime assert is_apple_gpu()

    var tidm = global_idx.x
    var tidn = global_idx.y

    var acc: Scalar[C_DTYPE] = 0

    for k in range(K):
        var a_ik: Scalar[C_DTYPE] = a[tidm, k].cast[C_DTYPE]()
        var b_kj: Scalar[C_DTYPE] = b[tidn, k].cast[C_DTYPE]()
        acc += a_ik * b_kj

    c[tidm, tidn] = acc


def main() raises:
    seed()

    if not has_accelerator():
        print("gpu is required here")
        exit()

    var a_host_buff = alloc[Scalar[AB_DTYPE]](A_LAYOUT.size())
    var b_host_buff = alloc[Scalar[AB_DTYPE]](B_LAYOUT.size())
    randn[AB_DTYPE](a_host_buff, M * K, mean=0.0, standard_deviation=1.0)
    randn[AB_DTYPE](b_host_buff, N * K, mean=0.0, standard_deviation=1.0)

    var a_host = TileTensor(a_host_buff, A_LAYOUT)
    var b_host = TileTensor(b_host_buff, B_LAYOUT)

    with DeviceContext() as ctx:
        var a_buff = ctx.enqueue_create_buffer[AB_DTYPE](A_LAYOUT.size())
        var b_buff = ctx.enqueue_create_buffer[AB_DTYPE](B_LAYOUT.size())
        var c_buff = ctx.enqueue_create_buffer[C_DTYPE](C_LAYOUT.size())

        ctx.enqueue_copy(a_buff, a_host_buff)
        ctx.enqueue_copy(b_buff, b_host_buff)

        var a = TileTensor(a_buff, A_LAYOUT)
        var b = TileTensor(b_buff, B_LAYOUT)
        var c = TileTensor(c_buff, C_LAYOUT)

        ctx.enqueue_function[kernel, kernel](
            a, b, c, grid_dim=(M // 16, N // 16), block_dim=(16, 16)
        )
        ctx.synchronize()

        with c_buff.map_to_host() as result_buff:
            var result = TileTensor(result_buff, C_LAYOUT)
            for i in range(M):
                for j in range(N):
                    var expected: Scalar[C_DTYPE] = 0
                    for k in range(K):
                        var a_ik: Scalar[C_DTYPE] = a_host[i, k].cast[C_DTYPE]()
                        var b_kj: Scalar[C_DTYPE] = b_host[j, k].cast[C_DTYPE]()
                        expected += a_ik * b_kj
                    if (expected - result[i, j]) > 0.001:
                        print("failed at ({}, {})".format(i, j))
