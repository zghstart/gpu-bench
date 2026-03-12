#!/usr/bin/env python3
# 脚本：02a_gemm_single_gpu_dense.py
# 用途：测试单卡 GEMM 的稠密算力（不启用稀疏）
# 对应：GPU 峰值为稠密算力（H200 约 989 TFLOPS TF32)

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


def benchmark_gemm_dense(dtype, size, device, label, peak_tflops, dense=True):
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
    flops = 2 * M * N * K  # GEMM: 2 * M * N * K
    tflops = flops / avg_time / 1e12
    efficiency = tflops / peak_tflops * 100

    # 对于稠密算力（H200: 989 TFLOPS）
    if dense:
        sparsity_note = " (稠密算力)"
    else:
        sparsity_note = " (稀疏算力)"

    status = "✅ 优秀" if efficiency >= 80 else ("✅ 良好" if efficiency >= 70 else "⚠️ 偏低")
    print(f"  {label:<25} {tflops:>8.1f} TFLOPS  峰值={peak_tflops} TFLOPS  达标率={efficiency:.1f}%  {status}")
    return tflops


def main():
    device = torch.device(f'cuda:{GPU_ID}')
    gpu_name = torch.cuda.get_device_name(GPU_ID)
    gpu_info = detect_gpu_type(GPU_ID)

    print(f"\n{'='*65}")
    print(f"  单卡 GEMM 基准测试 — {gpu_name}")
    print(f"  GPU 类型: {gpu_info['name']}")
    print(f"  矩阵尺寸: {MATRIX_SIZE}×{MATRIX_SIZE}  迭代次数: {TEST_ITERS}")
    print(f"  注意: 此版本测试稠密算力，理论峰值={gpu_info['fp32']} TFLOPS")
    print(f"{'='*65}")

    # FP32 (dense, no sparse)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    benchmark_gemm_dense(torch.float32, MATRIX_SIZE, device,
                         'FP32 (dense)', gpu_info['fp32'], dense=True)

    # TF32 (dense)
    benchmark_gemm_dense(torch.float32, MATRIX_SIZE, device,
                        'TF32 (dense)', gpu_info['fp32_tf32'], dense=True)

    # FP16 (dense - Tensor Core 不启用稀疏)
    torch.backends.cuda.matmul.allow_tf32 = False
    benchmark_gemm_dense(torch.float16, MATRIX_SIZE, device,
                        'FP16 (Tensor Core, dense)', gpu_info['fp16'], dense=True)

    # BF16 (dense - Tensor Core 不启用稀疏)
    benchmark_gemm_dense(torch.bfloat16, MATRIX_SIZE, device,
                        'BF16 (Tensor Core, dense)', gpu_info['bf16'], dense=True)

    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
