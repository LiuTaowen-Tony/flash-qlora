import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config({'block_M': 128, 'block_N': 256, 'block_K': 64, }, num_stages=3,
                      num_warps=8),
        triton.Config({'block_M': 64, 'block_N': 256, 'block_K': 32, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 128, 'block_K': 32, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 64, 'block_K': 32, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 64, 'block_N': 128, 'block_K': 32, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 32, 'block_K': 32, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 64, 'block_N': 32, 'block_K': 32, }, num_stages=5,
                      num_warps=2),
        triton.Config({'block_M': 32, 'block_N': 64, 'block_K': 32, }, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'block_M': 128, 'block_N': 256, 'block_K': 128, }, num_stages=3,
                      num_warps=8),
        triton.Config({'block_M': 256, 'block_N': 128, 'block_K': 128, }, num_stages=3,
                      num_warps=8),
        triton.Config({'block_M': 256, 'block_N': 64, 'block_K': 128, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 64, 'block_N': 256, 'block_K': 128, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 128, 'block_K': 128, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 64, 'block_K': 64, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 64, 'block_N': 128, 'block_K': 64, }, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 32, 'block_K': 64, }, num_stages=4,
                      num_warps=4)
    ]

# @triton.autotune(
#     configs=get_autotune_config(),
#     key=['M', 'N', 'K'],
# )
@triton.jit
def fast_qlora_kernel(
    w_ptr, x_ptr, u_ptr, v_ptr, c_ptr, x_int_ptr,
    block_M: tl.constexpr, block_N: tl.constexpr, block_K: tl.constexpr, 
    M: int, N: int, K: int, r: int
):
    pid_m = tl.program_id(0)  # block row index
    pid_n = tl.program_id(1)  # block column index
    
    # Block starting positions
    offs_m = pid_m * block_M
    offs_n = pid_n * block_N
    
    # Initialize fp and int accumulators
    fp_acc = tl.zeros((block_M, block_N), dtype=tl.float32)
    int_acc = tl.zeros((block_M, block_N), dtype=tl.int32)

    # Load block of V
    v_blk = tl.load(v_ptr + tl.arange(0, r)[:, None] * N + offs_n)

    for i in range(0, K, block_K):
        # Load blocks of X, W, and U
        x_blk = tl.load(x_ptr + (offs_m + tl.arange(0, block_M))[:, None] * K + (i + tl.arange(0, block_K)))
        w_blk = tl.load(w_ptr + (i + tl.arange(0, block_K))[:, None] * N + offs_n)
        u_blk = tl.load(u_ptr + (i + tl.arange(0, block_K))[:, None] * r + tl.arange(0, r))

        # Convert X to int8
        x_int_blk = tl.load(x_int_ptr + (offs_m + tl.arange(0, block_M))[:, None] * K + (i + tl.arange(0, block_K)))

        # Compute X * U
        xu_blk = tl.dot(x_blk, u_blk)
        xuv_blk = tl.dot(xu_blk, v_blk)

        # Accumulate fp result
        fp_acc += xuv_blk

        # Compute X * W
        xw_blk = tl.dot(x_int_blk.to(tl.int32), w_blk.to(tl.int32))
        int_acc += xw_blk

    # Store result to C
    c_out = int_acc + fp_acc
    tl.store(c_ptr + (offs_m + tl.arange(0, block_M))[:, None] * N + offs_n, c_out)


def fast_qlora(w, x, u, v, x_int, m, n, k, r):
    # Allocate result tensor on the GPU
    c = torch.empty((m, n), dtype=torch.float32, device='cuda')

    # Define the grid for the kernel launch based on problem size and block size
    # grid = lambda meta: (
    #     (m + meta['block_M'] - 1) // meta['block_M'],
    #     (n + meta['block_N'] - 1) // meta['block_N']
    # )
    grid = lambda meta: ()

    # Launch the Triton kernel with auto-tuned configurations
    fast_qlora_kernel[grid](
        w, x, u, v, c, x_int,
        block_M=2, block_N=2, block_K=2,
        M=m, N=n, K=k, r=r
    )

    return c

def test():
    # Small matrices for easy manual verification
    m, n, k, r = 4, 4, 4, 2

    # Initialize test matrices with known values
    w = torch.randint(-128, 128, (k, n), dtype=torch.int8, device='cuda')
    x = torch.randint(-128, 128, (m, k), dtype=torch.float32, device='cuda')
    u = torch.randint(-128, 128, (k, r), dtype=torch.float32, device='cuda')
    v = torch.randint(-128, 128, (r, n), dtype=torch.float32, device='cuda')
    x_int = x.to(torch.int8)

    # Direct computation for expected result
    expected_c = x_int.to(torch.float32) @ w.to(torch.float32) + x @ u @ v

    # Triton kernel computation
    c_triton = fast_qlora(w, x, u, v, x_int, m, n, k, r)

    # Check if the results match within a tolerance
    if torch.allclose(c_triton, expected_c, atol=1e-5):
        print("Test passed for small-scale test case.")
    else:
        print("Test failed for small-scale test case.")
        print("Expected:", expected_c)
        print("Triton Output:", c_triton)


if __name__ == '__main__':
    test()