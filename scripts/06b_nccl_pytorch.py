#!/usr/bin/env python3
# 脚本：06b_nccl_pytorch.py
# 用途：使用 PyTorch dist 测试 NCCL AllReduce/AllGather
# 支持任意数量的 GPU

import torch
import torch.distributed as dist
import time
import os


def benchmark_allreduce(tensor_size_bytes, warmup=5, iters=20):
    rank = dist.get_rank()
    num_elements = tensor_size_bytes // 4  # float32
    tensor = torch.ones(num_elements, dtype=torch.float32, device=f'cuda:{rank}')

    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / iters * 1e6
    world_size = dist.get_world_size()
    # AllReduce 算法带宽 = 2 * (N-1)/N * size / time
    algo_bw = 2 * (world_size - 1) / world_size * tensor_size_bytes / (elapsed / iters) / 1e9
    bus_bw = algo_bw  # nccl-tests 定义的 bus_bw
    return avg_us, algo_bw, bus_bw


def benchmark_allgather(tensor_size_bytes, warmup=5, iters=20):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 每个 rank 持有 size/world_size 的数据
    num_elements = tensor_size_bytes // 4 // world_size
    tensor = torch.ones(num_elements, dtype=torch.float32, device=f'cuda:{rank}')
    output = [torch.empty_like(tensor) for _ in range(world_size)]

    for _ in range(warmup):
        dist.all_gather(output, tensor)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dist.all_gather(output, tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / iters * 1e6
    algo_bw = tensor_size_bytes / (elapsed / iters) / 1e9
    bus_bw = algo_bw * (world_size - 1) / world_size
    return avg_us, algo_bw, bus_bw


def main():
    # 检查是否通过torchrun运行
    if not os.environ.get('RANK'):
        print("错误: 此脚本需要通过 torchrun 运行，例如:")
        print("  torchrun --nproc_per_node=<GPU数量> 06b_nccl_pytorch.py")
        print("或者在 run_all_tests.sh 脚本中运行")
        exit(1)
    
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    TEST_SIZES = [8, 1*1024**2, 64*1024**2, 256*1024**2, 1024**3, 2*1024**3]
    SIZE_LABELS = ['8B', '1MB', '64MB', '256MB', '1GB', '2GB']

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  NCCL AllReduce 性能测试  ({world_size} GPU)")
        print(f"  {'大小':<8} {'延迟(μs)':>10} {'算法BW(GB/s)':>14} {'总线BW(GB/s)':>14}")
        print(f"  {'─'*60}")

    for size, label in zip(TEST_SIZES, SIZE_LABELS):
        try:
            us, algo_bw, bus_bw = benchmark_allreduce(size)
            if rank == 0:
                status = "✅" if bus_bw > 100 or size <= 8 else "⚠️"
                print(f"  {label:<8} {us:>10.0f} {algo_bw:>14.1f} {bus_bw:>14.1f}  {status}")
        except Exception as e:
            if rank == 0:
                print(f"  {label:<8} 测试失败: {e}")

    if rank == 0:
        print(f"\n  {'─'*60}")
        print(f"  NCCL AllGather 性能测试  ({world_size} GPU)")
        print(f"  {'大小':<8} {'延迟(μs)':>10} {'算法BW(GB/s)':>14} {'总线BW(GB/s)':>14}")
        print(f"  {'─'*60}")

    for size, label in zip(TEST_SIZES[2:], SIZE_LABELS[2:]):  # 从 64MB 开始
        try:
            us, algo_bw, bus_bw = benchmark_allgather(size)
            if rank == 0:
                status = "✅" if algo_bw > 100 else "⚠️"
                print(f"  {label:<8} {us:>10.0f} {algo_bw:>14.1f} {bus_bw:>14.1f}  {status}")
        except Exception as e:
            if rank == 0:
                print(f"  {label:<8} 测试失败: {e}")

    if rank == 0:
        print(f"\n  参考值（8×H100 SXM5）：")
        print(f"    AllReduce 2GB 总线带宽: ~472 GB/s")
        print(f"    AllGather 2GB 总线带宽: ~350 GB/s")
        print(f"    AllReduce 8B 延迟:      ~40 μs")
        print(f"{'='*70}\n")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
