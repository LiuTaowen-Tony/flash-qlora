import torch
import triton
import dataclasses
from ml_utils.args import DataClassArgumentParser


@dataclasses.dataclass
class BenchmarkArgs:
    pass


def reference_impl(x, w, u, v):
    return x @ u @ v + x @ w

def test(merged_qlora_forward):
    # Small matrices for easy manual verification
    m, n, k, r = 128, 128, 128, 16

    # Initialize test matrices with known values
    x = torch.randn((m, k), dtype=torch.float32, device="cuda")
    w = torch.randn((k, n), dtype=torch.float32, device="cuda")
    u = torch.randn((k, r), dtype=torch.float32, device="cuda")
    v = torch.randn((r, n), dtype=torch.float32, device="cuda")

    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = w.to(torch.bfloat16)
    u_bf16 = u.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)

    # Triton kernel computation
    triton_out = merged_qlora_forward(
        x_bf16, w_bf16, u_bf16, v_bf16
    )
    bf16_out = reference_impl(x_bf16, w_bf16, u_bf16, v_bf16)
    expected_out = reference_impl(x, w, u, v).to(torch.bfloat16)

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
                256 * i for i in range(14, 16)
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
    for i in range(1, 2):
        version = f"merged_forward_v{i}"
        merged_qlora_forward = getattr(__import__(version), "merged_qlora_forward")
        test(merged_qlora_forward)
        configs = get_benchmark_configs()
        @triton.testing.perf_report(configs)
        def benchmark(M, N, K, provider):
            x = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
            w = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)
            u = torch.randn((K, 16), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((16, N), device='cuda', dtype=torch.bfloat16)
            quantiles = [0.5, 0.2, 0.8]
            if provider == 'torch':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: reference_impl(x,w,u,v), quantiles=quantiles)
            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: merged_qlora_forward(x, w, u, v), quantiles=quantiles)
            perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
            return perf(ms), perf(max_ms), perf(min_ms)
        result = benchmark.run(print_data=True, save_path=f"benchmarks/{version}")
