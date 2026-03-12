#!/usr/bin/env python3
# 脚本：02b_gemm_multi_gpu.py
# 用途：测试多卡并行 GEMM 算力，支持 H100/H200/B200/B300
# 自动检测 GPU 类型并使用对应理论峰值

import torch
import threading
import time
import sys
import os

# 添加 scripts 目录到路径以导入 gpu_config
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from gpu_config import detect_gpu_type


MATRIX_SIZE = 4096
WARMUP_ITERS = 3
TEST_ITERS = 10
DTYPE = torch.float16


def benchmark_gpu(gpu_id, results, gpu_peaks):
    device = torch.device(f'cuda:{gpu_id}')
    M = N = K = MATRIX_SIZE
    A = torch.randn(M, K, dtype=DTYPE, device=device)
    B = torch.randn(K, N, dtype=DTYPE, device=device)

    for _ in range(WARMUP_ITERS):
        C = torch.mm(A, B)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(TEST_ITERS):
        C = torch.mm(A, B)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    flops = 2 * M * N * K
    tflops = flops / (elapsed / TEST_ITERS) / 1e12
    peak = gpu_peaks.get(gpu_id, 1979)  # 默认使用 H100 峰值
    results[gpu_id] = {'tflops': tflops, 'peak': peak}


def main():
    num_gpus = torch.cuda.device_count()

    # 为每个 GPU 检测类型并获取 FP16 峰值
    gpu_peaks = {}
    gpu_names = {}
    for i in range(num_gpus):
        info = detect_gpu_type(i)
        gpu_peaks[i] = info['fp16']
        gpu_names[i] = info['name']

    print(f"\n{'='*65}")
    print(f"  多卡并行 GEMM — FP16  {MATRIX_SIZE}×{MATRIX_SIZE}  共 {num_gpus} 卡")
    print(f"{'='*65}")

    results = {}
    threads = []
    for i in range(num_gpus):
        t = threading.Thread(target=benchmark_gpu, args=(i, results, gpu_peaks))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = 0
    print(f"\n  {'GPU':<6} {'类型':<20} {'TFLOPS':>10} {'效率':>8}  状态")
    print(f"  {'─'*60}")

    for i in range(num_gpus):
        data = results[i]
        tflops = data['tflops']
        peak = data['peak']
        efficiency = tflops / peak * 100
        gpu_name = gpu_names[i][:18]  # 限制长度

        if tflops < 100:
            status = "⚠️ 严重异常"
        elif efficiency < 70:
            status = "⚠️ 偏低"
        else:
            status = "✅ 正常"

        print(f"  GPU {i:<2} {gpu_name:<20} {tflops:>10.1f} {efficiency:>6.1f}%  {status}")
        total += tflops

    print(f"  {'─'*60}")
    print(f"  {'合计':<28} {total:>10.1f} TFLOPS")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
