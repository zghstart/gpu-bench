#!/usr/bin/env python3
# 脚本：02a_gemm_single_gpu.py
# 用途：测试单卡 GEMM 算力，支持 H100/H200/B200/B300
# 自动检测 GPU 类型并使用对应理论峰值

import torch
import time
import sys
import os

# 添加 scripts 目录到路径以导入 gpu_config
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from gpu_config import detect_gpu_type, get_gpu_peak_info

# 测试配置
MATRIX_SIZE = 8192
WARMUP_ITERS = 5
TEST_ITERS = 20
GPU_ID = 0


def benchmark_gemm(dtype, size, device, label, peak_tflops, has_sparsity=False):
    M = N = K = size
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(WARMUP_ITERS):
        C = torch.mm(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(TEST_ITERS):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time = elapsed / TEST_ITERS
    # FLOPS = 2 * M * N * K
    flops = 2 * M * N * K
    tflops = flops / avg_time / 1e12
    efficiency = tflops / peak_tflops * 100

    status = "✅ 优秀" if efficiency >= 80 else ("✅ 良好" if efficiency >= 70 else "⚠️ 偏低")
    print(f"  {label:<25} {tflops:>8.1f} TFLOPS  峰值={peak_tflops} TFLOPS  达标率={efficiency:.1f}%  {status}")
    return tflops


def main():
    device = torch.device(f'cuda:{GPU_ID}')
    gpu_name = torch.cuda.get_device_name(GPU_ID)
    gpu_info = detect_gpu_type(GPU_ID)

    # 根据是否支持稀疏，显示对应的峰值
    has_sparsity = gpu_info.get('has_sparsity', True)
    sparsity_note = " (稀疏算力)" if has_sparsity else " (稠密算力)"

    print(f"\n{'='*65}")
    print(f"  单卡 GEMM 基准测试 — {gpu_name}")
    print(f"  GPU 类型: {gpu_info['name']}{sparsity_note}")
    print(f"  矩阵尺寸: {MATRIX_SIZE}×{MATRIX_SIZE}  迭代次数: {TEST_ITERS}")
    print(f"{'='*65}")

    # TF32 (torch.float32 with TF32 enabled)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    benchmark_gemm(torch.float32, MATRIX_SIZE, device,
                   'FP32 (TF32 enabled)', gpu_info['fp32_tf32'], has_sparsity)

    # FP16
    torch.backends.cuda.matmul.allow_tf32 = False
    benchmark_gemm(torch.float16, MATRIX_SIZE, device,
                   'FP16 (Tensor Core)', gpu_info['fp16'], has_sparsity)

    # BF16
    benchmark_gemm(torch.bfloat16, MATRIX_SIZE, device,
                   'BF16 (Tensor Core)', gpu_info['bf16'], has_sparsity)

    # FP8 (如果支持且 GPU 支持 float8)
    if gpu_info.get('fp8'):
        try:
            # 检查是否支持 float8
            if hasattr(torch, 'float8_e4m3fn'):
                benchmark_gemm(torch.float8_e4m3fn, MATRIX_SIZE, device,
                               'FP8 (Tensor Core)', gpu_info['fp8'], has_sparsity)
        except Exception:
            pass  # 跳过 FP8 测试

    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
