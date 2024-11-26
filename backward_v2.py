import torch
import triton
import triton.language as tl
import common

@triton.autotune(
    configs=common.get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def lora_backward_gA_gB_kernel(
    gO_ptr, 
    At_ptr, 
    Bt_ptr, 
    Xt_ptr,
    gA_ptr,
    gB_ptr,
    stride_gok,
    stride_gon,
    stride_xtm,
    stride_xtk,
    stride_atr,
    stride_atm,
    stride_btn,
    stride_btr,
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
    AT_block = tl.load(
        At_ptr 
        + tl.arange(0, R)[:, None] * stride_atr
        + (offs_m + tl.arange(0, block_M)) * stride_atm
    )
    BT_block = tl.load(
        Bt_ptr 
        + (offs_n + tl.arange(0, block_N))[:, None] * stride_btn
        + tl.arange(0, R) * stride_btr
    )

    for i in range(0, K, block_K):
        XT_blk = tl.load(
            Xt_ptr 
            + (offs_m + tl.arange(0, block_M))[:, None] * stride_xtm
            + tl.arange(0, block_K) * stride_xtk
        )
        gO_blk = tl.load(
            gO_ptr 
            + tl.arange(0, block_K)[:, None] * stride_gok 
            + (offs_n + tl.arange(0,  block_N)) * stride_gon
        )

        acc += tl.dot(XT_blk, gO_blk)

    accbf16 = tl.cast(acc, dtype=tl.bfloat16)
    gA_blk = tl.dot(accbf16, BT_block)

    ga_store_ptrs = (gA_ptr 
                     + (offs_m + tl.arange(0, block_M))[:, None] * stride_atr 
                     + tl.arange(0, R))
    tl.atomic_add(ga_store_ptrs, gA_blk)

    gB_blk = tl.dot(AT_block, accbf16)
    gB_store_ptrs = (gB_ptr 
                     + tl.arange(0, R)[:, None] * stride_btn 
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

    grid = lambda opt: (triton.cdiv(M, opt["block_M"]), triton.cdiv(N, opt["block_N"]))

    lora_backward_gA_gB_kernel[grid](
        gO,
        AT,
        BT,
        XT,
        gA,
        gB,
        gO.stride(0),
        gO.stride(1),
        XT.stride(0),
        XT.stride(1),
        AT.stride(0),
        AT.stride(1),
        BT.stride(0),
        BT.stride(1),
        M=M,
        N=N,
        K=K,
    )
    return gA, gB



