import torch
import triton
import dataclasses
from ml_utils.args import DataClassArgumentParser


@dataclasses.dataclass
class BenchmarkArgs:
    pass


def test(svd_matmul_fn):
    # Small matrices for easy manual verification
    m, n, k, r = 128, 128, 128, 16

    # Initialize test matrices with known values
    x = torch.randn((m, k), dtype=torch.float32, device="cuda")
    u = torch.randn((k, r), dtype=torch.float32, device="cuda")
    v = torch.randn((r, n), dtype=torch.float32, device="cuda")

    # Triton kernel computation
    triton_out = svd_matmul_fn(
        x.to(torch.bfloat16), u.to(torch.bfloat16), v.to(torch.bfloat16)
    )
    bf16_out = x.to(torch.bfloat16) @ u.to(torch.bfloat16) @ v.to(torch.bfloat16)
    expected_out = (x @ u @ v).to(torch.bfloat16)

    triton_error_norm = torch.linalg.norm(triton_out - expected_out)
    bf16_error_norm = torch.linalg.norm(bf16_out - expected_out)
    print(f"Triton error: {triton_error_norm}")
    print(f"BF16 error: {bf16_error_norm}")
    return triton_error_norm < 1.5 * bf16_error_norm


def get_benchmark_configs():
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[
                128 * i for i in range(2, 33)
            ],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=["torch" ,"triton"],  # Label name for the lines
            line_names=["torch", "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-"),],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="svd-matmul-performance-",
            args={},
        )
    )
    return configs


if __name__ == "__main__":
    for i in range(1, 3):
        version = f"forward_v{i}"
        svd_matmul_fn = getattr(__import__(version), "svd_weight_matmul")
        test(svd_matmul_fn)
        configs = get_benchmark_configs()
        @triton.testing.perf_report(configs)
        def benchmark(M, N, K, provider):
            x = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
            u = torch.randn((K, 16), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((16, N), device='cuda', dtype=torch.bfloat16)
            quantiles = [0.5, 0.2, 0.8]
            if provider == 'torch':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: x @ u @ v, quantiles=quantiles)
            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: svd_matmul_fn(x, u, v), quantiles=quantiles)
            perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
            return perf(ms), perf(max_ms), perf(min_ms)
        result = benchmark.run(print_data=True, save_path=f"benchmarks/{version}")
