import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config({'block_M': 128, 'block_N': 256, 'block_K': 64, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'block_M': 64, 'block_N': 256, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 128, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 64, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 64, 'block_N': 128, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 32, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 64, 'block_N': 32, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'block_M': 32, 'block_N': 64, 'block_K': 32, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'block_M': 128, 'block_N': 256, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'block_M': 256, 'block_N': 128, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'block_M': 256, 'block_N': 64, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 64, 'block_N': 256, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 128, 'block_K': 128, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 64, 'block_K': 64, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 64, 'block_N': 128, 'block_K': 64, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'block_M': 128, 'block_N': 32, 'block_K': 64, 'R': 16, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def svd_weight_kernel(
    x_ptr, u_ptr, v_ptr, c_ptr,
    stride_xm, stride_xk,
    stride_uk, stride_ur,
    stride_vr, stride_vn,
    M: int, N: int, K: int, R: tl.constexpr,
    block_M: tl.constexpr, block_N: tl.constexpr, block_K: tl.constexpr, 
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
    v_blk = tl.load(v_ptr + tl.arange(0, R)[:, None] * stride_vr + tl.arange(0, block_N))

    for i in range(0, K, block_K):
        # Load blocks of X, W, and U
        x_blk = tl.load(x_ptr + (offs_m + tl.arange(0, block_M))[:, None] * stride_xm + (i + tl.arange(0, block_K)))
        u_blk = tl.load(u_ptr + (i + tl.arange(0, block_K))[:, None] * stride_uk + tl.arange(0, R))

        # Compute X * U
        # xu_blk = tl.dot(x_blk, u_blk)
        # xu_blk2 = tl.cast(xu_blk, dtype=tl.bfloat16)
        # xuv_blk = tl.dot(xu_blk2, v_blk)
        uv_blk = tl.dot(u_blk, v_blk)
        xuv_blk = tl.dot(x_blk, tl.cast(uv_blk, dtype=tl.bfloat16))

        # Accumulate fp result
        fp_acc += xuv_blk
    # Store result to C
    c_out =  fp_acc
    tl.store(c_ptr + (offs_m + tl.arange(0, block_M))[:, None] * N + tl.arange(0, block_N), c_out)

@triton.jit
def reorder_pid(pid_m, pid_n, M, N, 
                block_M: tl.constexpr, block_N: tl.constexpr, 
                GROUP_SIZE_M: tl.constexpr):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, block_M)
    num_pid_n = tl.cdiv(N, block_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def svd_weight_matmul(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Allocate result tensor on the GPU
    m, k = x.shape
    r, n = v.shape

    assert k == u.shape[0]
    assert u.shape[1] == r

    c = torch.empty((m, n), dtype=x.dtype, device='cuda')
    grid = lambda opt: (triton.cdiv(m, opt["block_M"]), triton.cdiv(n, opt["block_N"]))

    # Launch the Triton kernel with auto-tuned configurations
    svd_weight_kernel[grid](
        x, u, v, c,
        x.stride(0), x.stride(1),
        u.stride(0), u.stride(1),
        v.stride(0), v.stride(1),
        M=m, N=n, K=k, 
    )

    return c

def test():
    # Small matrices for easy manual verification
    m, n, k, r = 128, 128, 128, 16

    # Initialize test matrices with known values
    x = torch.randn( (m, k), dtype=torch.bfloat16, device='cuda') 
    u = torch.randn( (k, r), dtype=torch.bfloat16, device='cuda') 
    v = torch.randn( (r, n), dtype=torch.bfloat16, device='cuda') 

    # Triton kernel computation
    c_triton = svd_weight_matmul( x, u, v)
    expected_c = x @ u @ v

    # Check if the results match within a tolerance
    if torch.allclose(c_triton, expected_c, atol=1e-5):
        print("Test passed for small-scale test case.")
    else:
        print("Test failed for small-scale test case.")
        print("Expected:", expected_c)
        print("Triton Output:", c_triton)


configs = []
for fp8_inputs in [False, True]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["torch", "triton"],  # Label name for the lines
            line_names=["torch", "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-",
            args={},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    x = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    u = torch.randn((K, 16), device='cuda', dtype=torch.bfloat16)
    v = torch.randn((16, N), device='cuda', dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x @ u @ v, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: svd_weight_matmul(x, u, v), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

test()
benchmark.run(show_plots=True, print_data=True)
