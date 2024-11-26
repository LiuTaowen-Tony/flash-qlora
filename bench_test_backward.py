import torch
import triton
import dataclasses
from ml_utils.args import DataClassArgumentParser


@dataclasses.dataclass
class BenchmarkArgs:
    pass


def test(backward_fn):
    # Small matrices for easy manual verification
    m, n, k, r = 512, 512, 512, 16

    # Initialize test matrices with known values
    x = torch.randn((m, k), dtype=torch.float32, device="cuda")
    u = torch.randn((k, r), dtype=torch.float32, device="cuda")
    v = torch.randn((r, n), dtype=torch.float32, device="cuda")
    go = torch.randn((m, n), dtype=torch.float32, device="cuda")

    go16 = go.to(torch.bfloat16)
    x16 = x.to(torch.bfloat16)
    u16 = u.to(torch.bfloat16)
    v16 = v.to(torch.bfloat16)
    # Triton kernel computation
    triton_gA, triton_gB, triton_xtgo = backward_fn(
        go16,  u16, v16, x16
    )


    gA16 = x16.T @ (go16 @ v16.T)
    gB16 = (u16.T @ x16.T) @ go16

    gA = x.T @ (go @ v.T)
    gB = (u.T @ x.T) @ go
    xtgo = x.T @ go

    # print(triton_xtgo)
    # print(xtgo)

    print(triton_gA)
    print(gA)

    triton_error_norm = torch.linalg.norm(triton_gA - gA) + torch.linalg.norm(triton_gB - gB)
    bf16_error_norm = torch.linalg.norm(gA16 - gA) + torch.linalg.norm(gB16 - gB)
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
        version = f"backward_v{i}"
        triton_back = getattr(__import__(version), "lora_backward_gA_gB")
        assert test(triton_back), f"Test failed for {version}"
        configs = get_benchmark_configs()
        @triton.testing.perf_report(configs)
        def benchmark(M, N, K, provider):
            x = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
            u = torch.randn((K, 16), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((16, N), device='cuda', dtype=torch.bfloat16)
            go = torch.randn((M, N), device='cuda', dtype=torch.bfloat16)
            quantiles = [0.5, 0.2, 0.8]
            if provider == 'torch':
                def torch_back(go, x, u, v):
                    gA = x.T @ (go @ v.T)
                    gB = (u.T @ x.T) @ go
                    return gA, gB
                ms, min_ms, max_ms = triton.testing.do_bench(lambda : torch_back(go, u, v, x), quantiles=quantiles)
            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_back(go, u, v, x), quantiles=quantiles)
            perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
            return perf(ms), perf(max_ms), perf(min_ms)
        result = benchmark.run(print_data=True, save_path=f"benchmarks/{version}")
