import torch
import triton
import triton.language as tl
import common


@triton.autotune(
    configs=common.get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def svd_weight_kernel(
    x_ptr,
    u_ptr,
    v_ptr,
    c_ptr,
    stride_xm,
    stride_xk,
    stride_uk,
    stride_ur,
    stride_vr,
    stride_vn,
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
    # pid_m, pid_n = reorder_pid(pid_m, pid_n, M, N, block_M, block_N, GROUP_SIZE_M)

    # Block starting positions
    offs_m = pid_m * block_M
    offs_n = pid_n * block_N

    # Initialize fp and int accumulators
    fp_acc = tl.zeros((block_M, block_N), dtype=tl.float32)

    # Load block of V
    v_blk = tl.load(
        v_ptr + tl.arange(0, R)[:, None] * stride_vr + tl.arange(0, block_N)
    )
    # R: 16 block_N: 256 block_K: 32 block_M: 64

    for i in range(0, K, block_K):
        # Load blocks of X, W, and U
        x_blk = tl.load(
            x_ptr
            + (offs_m + tl.arange(0, block_M))[:, None] * stride_xm
            + (i + tl.arange(0, block_K))
        )
        u_blk = tl.load(
            u_ptr + (i + tl.arange(0, block_K))[:, None] * stride_uk + tl.arange(0, R)
        )

        # Compute X * U
        # xu_blk = tl.dot(x_blk, u_blk)
        # xu_blk2 = tl.cast(xu_blk, dtype=tl.bfloat16)
        # xuv_blk = tl.dot(xu_blk2, v_blk)
        # (32 * 16) (16 256) -> 32 256 32 * 16 * 256
        uv_blk = tl.dot(u_blk, v_blk)
        # (64 32) (32 256) -> 64 256 64 * 32 * 256
        xuv_blk = tl.dot(x_blk, tl.cast(uv_blk, dtype=tl.bfloat16))
        # xu (32 16 64) 
        # xuv (16 256 64)

        # Accumulate fp result
        fp_acc += xuv_blk
    # Store result to C
    c_out = tl.cast(fp_acc, dtype=tl.bfloat16)
    tl.store(
        c_ptr + (offs_m + tl.arange(0, block_M))[:, None] * N + tl.arange(0, block_N),
        c_out,
    )


def svd_weight_matmul(
    x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # Allocate result tensor on the GPU
    m, k = x.shape
    r, n = v.shape

    assert k == u.shape[0]
    assert u.shape[1] == r

    c = torch.empty((m, n), dtype=x.dtype, device="cuda")
    grid = lambda opt: (triton.cdiv(m, opt["block_M"]), triton.cdiv(n, opt["block_N"]))

    # Launch the Triton kernel with auto-tuned configurations
    svd_weight_kernel[grid](
        x,
        u,
        v,
        c,
        x.stride(0),
        x.stride(1),
        u.stride(0),
        u.stride(1),
        v.stride(0),
        v.stride(1),
        M=m,
        N=n,
        K=k,
    )

    return c

