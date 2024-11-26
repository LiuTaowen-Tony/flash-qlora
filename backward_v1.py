import torch
import triton
import triton.language as tl
def get_autotune_config():
    return [
        triton.Config({'block_M': 128, 'block_N': 256, 'block_K': 64, 'R': 16, 'GROUP_SIZE_M': 8},),
        triton.Config({'block_M': 64, 'block_N': 256, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, ),
        triton.Config({'block_M': 128, 'block_N': 128, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, ),
        triton.Config({'block_M': 128, 'block_N': 64, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, ),
        triton.Config({'block_M': 64, 'block_N': 128, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, ),
        triton.Config({'block_M': 128, 'block_N': 32, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, ),
        triton.Config({'block_M': 64, 'block_N': 32, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, ),
        triton.Config({'block_M': 32, 'block_N': 64, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, ),
        # Good config for fp8 inputs.
        # triton.Config({'block_M': 128, 'block_N': 256, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'block_M': 256, 'block_N': 128, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'block_M': 256, 'block_N': 64, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'block_M': 64, 'block_N': 256, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'block_M': 128, 'block_N': 128, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'block_M': 128, 'block_N': 64, 'block_K': 64, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'block_M': 64, 'block_N': 128, 'block_K': 64, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'block_M': 128, 'block_N': 32, 'block_K': 64, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4)
    ]

# @triton.autotune(
#     configs=get_autotune_config(),
#     key=["M", "N", "K"],
# )
@triton.jit
def lora_backward_gA_gB_kernel(
    gO_ptr, 
    At_ptr, 
    Bt_ptr, 
    Xt_ptr,
    gA_ptr,
    gB_ptr,
    XtgO_ptr,
    stride_gok,
    stride_gon,
    stride_atr,
    stride_atm,
    stride_btn,
    stride_btr,
    stride_xtm,
    stride_xtk,
    M: int,
    N: int,
    K: int,
    R: tl.constexpr,
    block_M: tl.constexpr,
    block_N: tl.constexpr,
    block_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ):
    # XT [m x k]
    # gO [k x n]
    # AT [r x m]
    # BT [n x r]
    # gA [m x r]
    # gB [r x n]

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * block_M
    offs_n = pid_n * block_N


    acc = tl.zeros((block_M, block_N), dtype=tl.float32)

    offs_XTm = (pid_m * block_M + tl.arange(0, block_M)) % M
    offs_gOn = (pid_n * block_N + tl.arange(0, block_N)) % N
    offs_k = tl.arange(0, block_K)
    Xt_ptrs = Xt_ptr + (offs_XTm[:, None] * K + offs_k[None, :])
    gO_ptrs = gO_ptr + (offs_k[:, None] * N + offs_gOn[None, :])


    for i in range(0, K, block_K):
        XT_blk = tl.load(Xt_ptrs)
        gO_blk = tl.load(gO_ptrs)
        acc = tl.dot(XT_blk, gO_blk, acc)
        Xt_ptrs += block_K
        gO_ptrs += block_K * N

    tl.store(XtgO_ptr 
             + (offs_m + tl.arange(0, block_M))[:, None] * N 
             + offs_n + tl.arange(0, block_N), 
            acc)
    accbf16 = tl.cast(acc, dtype=tl.bfloat16)

    BT_block = tl.load(
        Bt_ptr 
        + (offs_n + tl.arange(0, block_N))[:, None] * R
        + tl.arange(0, R)[None, :]
    )
    gA_blk = tl.dot(accbf16, BT_block)
    ga_store_ptrs = (gA_ptr 
                     + (offs_m + tl.arange(0, block_M))[:, None] * R 
                     + tl.arange(0, R)[None, :])
    tl.atomic_add(ga_store_ptrs, gA_blk)

    AT_block = tl.load(
        At_ptr 
        + tl.arange(0, R)[:, None] * M
        + (offs_m + tl.arange(0, block_M))
    )
    gB_blk = tl.dot(AT_block, accbf16)
    gB_store_ptrs = (gB_ptr 
                     + tl.arange(0, R)[:, None] * N 
                     + (offs_n + tl.arange(0, block_N)))
    tl.atomic_add(gB_store_ptrs, gB_blk)


def lora_backward_gA_gB(gO, A, B, X):
    # gO [k x n]
    # A [m x r]
    # B [r x n]
    # X [k x m]

    AT = A.T.contiguous()
    BT = B.T.contiguous()
    XT = X.T.contiguous()

    M, R = A.shape
    N = B.shape[1]
    K = X.shape[0]

    assert X.shape[1] == M
    assert gO.shape[0] == K
    assert gO.shape[1] == N


    gA = torch.zeros((M, R), dtype=torch.float32, device="cuda")
    gB = torch.zeros((R, N), dtype=torch.float32, device="cuda")
    assert gO.stride(0) == N
    assert gO.stride(1) == 1

    assert XT.stride(0) == K
    assert XT.stride(1) == 1

    assert AT.stride(0) == M
    assert AT.stride(1) == 1

    assert BT.stride(0) == R
    assert BT.stride(1) == 1

    grid = lambda opt: (triton.cdiv(M, opt["block_M"]), triton.cdiv(N, opt["block_N"]))

    XtgO = torch.empty((M, N), dtype=torch.float32, device="cuda")

    lora_backward_gA_gB_kernel[grid](
        gO,
        AT,
        BT,
        XT,
        gA,
        gB,
        XtgO,
        gO.stride(0),
        gO.stride(1),
        AT.stride(0),
        AT.stride(1),
        BT.stride(0),
        BT.stride(1),
        XT.stride(0),
        XT.stride(1),
        M=M,
        N=N,
        K=K,
        R=16,
        block_M=32,
        block_N=32,
        block_K=16,
        GROUP_SIZE_M=1
    )
    gA = gA.to(torch.bfloat16)
    gB = gB.to(torch.bfloat16)
    return gA, gB, XtgO



