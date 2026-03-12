#!/usr/bin/env python3
# 脚本：03_memory_bandwidth.py
# 用途：测试 HBM 显存带宽，支持 H100/H200/B200/B300
# 自动检测 GPU 类型并使用对应理论带宽

import torch
import time
import sys
import os

# 添加 scripts 目录到路径以导入 gpu_config
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from gpu_config import detect_gpu_type

WARMUP_ITERS = 10
TEST_ITERS = 30

# 测试规模：H200 的 141GB 显存需要测试更大规模
TEST_SIZES_MB = [256, 1024, 4096, 16384]


def benchmark_bandwidth(size_mb, device, gpu_id, gpu_info):
    num_elements = size_mb * 1024 * 1024 // 4  # float32: 4 bytes
    src = torch.randn(num_elements, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)

    # Warmup
    for _ in range(WARMUP_ITERS):
        dst.copy_(src)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(TEST_ITERS):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_transferred = num_elements * 4 * 2  # read + write
    bw_gbs = bytes_transferred * TEST_ITERS / elapsed / 1e9
    efficiency = bw_gbs / gpu_info['hbm_bw'] * 100
    status = "✅ 优秀" if efficiency >= 80 else ("✅ 良好" if efficiency >= 70 else "⚠️ 偏低")

    size_label = f"{size_mb}MB" if size_mb < 1024 else f"{size_mb//1024}GB"
    print(f"  {size_label} copy:  {bw_gbs:>8.1f} GB/s  "
          f"理论={gpu_info['hbm_bw']} GB/s  达标率={efficiency:.1f}%  {status}")
    return bw_gbs


def main():
    num_gpus = torch.cuda.device_count()

    print(f"\n{'='*65}")
    print(f"  HBM 显存带宽测试  (迭代: {TEST_ITERS} 次)")
    print(f"{'='*65}")

    for gpu_id in range(num_gpus):
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_info = detect_gpu_type(gpu_id)

        print(f"\n  GPU {gpu_id}: {gpu_name}")
        print(f"    类型: {gpu_info['name']}")
        print(f"    理论带宽: {gpu_info['hbm_bw']} GB/s")
        for size_mb in TEST_SIZES_MB:
            benchmark_bandwidth(size_mb, device, gpu_id, gpu_info)

    print(f"\n{'='*65}\n")


if __name__ == '__main__':
    main()
