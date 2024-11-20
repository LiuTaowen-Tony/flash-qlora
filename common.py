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

