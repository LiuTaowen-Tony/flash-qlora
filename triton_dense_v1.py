
import torch
import triton
import triton.language as tl
import common


def get_configs_io_bound():
    configs = []
    for block_n in [256, 128, 64, 32, 16]:
        for block_m in [256, 128, 64, 32]:
            for block_k in [256, 128, 64]:
                for num_stages in [5, 4, 3]:
                    for num_warps in [4, 8]:
                        for num_ctas in [1]:
                            if block_m * block_n * block_k >= 16 * 64 * 64 and block_m * block_n * block_k <= 128 * 128 * 256:
                                configs.append(
                                    triton.Config({'block_M': block_m, 'block_N': block_n, 'block_K': block_k, 'R': 16, 'GROUP_SIZE_M': 8},
                                                    num_stages=num_stages, num_warps=num_warps, num_ctas=num_ctas))
                        # for split_k in [2, 4, 8, 16]:
                        #     configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                        #                                     num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs

@triton.autotune(
    configs=common.get_autotune_config(),
    # configs=get_configs_io_bound(),
    key=["M", "N", "K"],
)
@triton.jit
def triton_dense_forward_kernel(
    x_ptr,
    w_ptr,
    c_ptr,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    M: int,
    N: int,
    K: int,
    R: tl.constexpr,
    block_M: tl.constexpr,
    block_N: tl.constexpr,
    block_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_m, pid_n = common.reorder_pid(pid_m, pid_n, M, N, block_M, block_N, GROUP_SIZE_M)

    # Block starting positions
    offs_m = pid_m * block_M
    offs_n = pid_n * block_N

    # Initialize fp and int accumulators
    fp_acc = tl.zeros((block_M, block_N), dtype=tl.float32)

    # R: 16 block_N: 256 block_K: 32 block_M: 64

    for i in range(0, K, block_K):
        # Load blocks of X, W, and U
        w_blk = tl.load(
            w_ptr
            + (i + tl.arange(0, block_K))[:, None] * stride_wk
            + tl.arange(0, block_N)
        )
        x_blk = tl.load(
            x_ptr
            + (offs_m + tl.arange(0, block_M))[:, None] * stride_xm
            + (i + tl.arange(0, block_K))
        )
        fp_acc = tl.dot(x_blk, w_blk, fp_acc)

    tl.store(
        c_ptr + (offs_m + tl.arange(0, block_M))[:, None] * N + tl.arange(0, block_N),
        fp_acc,
    )


def triton_dense_forward(
    x: torch.Tensor, w: torch.Tensor,
) -> torch.Tensor:
    # Allocate result tensor on the GPU
    m, k = x.shape
    _, n = w.shape

    assert w.shape[0] == k

    c = torch.empty((m, n), dtype=x.dtype, device="cuda")
    grid = lambda opt: (triton.cdiv(m, opt["block_M"]), triton.cdiv(n, opt["block_N"]))

    # Launch the Triton kernel with auto-tuned configurations
    triton_dense_forward_kernel[grid](
        x,
        w,
        c,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        M=m,
        N=n,
        K=k,
    )

    return c

