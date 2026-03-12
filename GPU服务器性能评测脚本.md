# GPU 服务器性能评测脚本

> 基于九章云极云 GPU 服务器性能评测报告还原的完整测试脚本集
> 测试环境：8× NVIDIA H100 80GB SXM5 | Kubernetes 容器 | Ubuntu 22.04 | PyTorch 2.8.0+cu128

---

## 目录

1. [环境检查与硬件信息采集](#1-环境检查与硬件信息采集)
2. [GEMM 矩阵乘法基准测试](#2-gemm-矩阵乘法基准测试)
3. [HBM3 显存带宽测试](#3-hbm3-显存带宽测试)
4. [磁盘 I/O 性能测试](#4-磁盘-io-性能测试)
5. [GPU 拓扑与 NVLink 检查](#5-gpu-拓扑与-nvlink-检查)
6. [NCCL 多卡通信性能测试](#6-nccl-多卡通信性能测试)
7. [GPU-Burn 稳定性压力测试](#7-gpu-burn-稳定性压力测试)
8. [GPU 时钟频率与温度监控](#8-gpu-时钟频率与温度监控)
9. [Transformer 推理吞吐量测试](#9-transformer-推理吞吐量测试)
10. [一键运行全部测试](#10-一键运行全部测试)

---

## 1. 环境检查与硬件信息采集

```bash
#!/bin/bash
# 脚本：01_env_check.sh
# 用途：采集系统硬件配置、驱动版本、CUDA 版本等基础信息

echo "========================================"
echo "  GPU 服务器环境信息采集"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

echo ""
echo "--- 操作系统 ---"
cat /etc/os-release | grep -E "^(NAME|VERSION)="

echo ""
echo "--- CPU 信息 ---"
lscpu | grep -E "(Model name|Socket|Core|Thread|NUMA)"

echo ""
echo "--- 内存信息 ---"
free -h

echo ""
echo "--- GPU 基础信息 ---"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.current \
    --format=csv,noheader

echo ""
echo "--- GPU 驱动与 CUDA 版本 ---"
nvidia-smi | head -4
nvcc --version 2>/dev/null || echo "nvcc not found"

echo ""
echo "--- PyTorch 版本 ---"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "--- 磁盘挂载信息 ---"
df -h | grep -E "(Filesystem|overlay|nvme|nfs)"

echo ""
echo "--- 网络接口 ---"
ibstat 2>/dev/null | grep -E "(CA|Port|State|Physical)" | head -40 || \
    ip link show | grep -E "^[0-9]+:|mlx5|ib"
```

---

## 2. GEMM 矩阵乘法基准测试

### 2.1 单卡 GEMM（8192×8192）

```python
#!/usr/bin/env python3
# 脚本：02a_gemm_single_gpu.py
# 用途：测试单卡 GEMM 算力，对应报告第 2.1 节
# 理论峰值：TF32=495 TFLOPS, FP16/BF16=989 TFLOPS

import torch
import time

# 测试配置
MATRIX_SIZE = 8192
WARMUP_ITERS = 5
TEST_ITERS = 20
GPU_ID = 0

# H100 SXM5 理论峰值（非稀疏）
THEORETICAL_PEAKS = {
    'fp32_tf32': 495,   # TF32 Tensor Core
    'fp16':      989,   # FP16 Tensor Core
    'bf16':      989,   # BF16 Tensor Core
}

def benchmark_gemm(dtype, size, device, label, peak_tflops):
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
    print(f"\n{'='*65}")
    print(f"  单卡 GEMM 基准测试 — {gpu_name} (GPU {GPU_ID})")
    print(f"  矩阵尺寸: {MATRIX_SIZE}×{MATRIX_SIZE}  迭代次数: {TEST_ITERS}")
    print(f"{'='*65}")

    # TF32 (torch.float32 with TF32 enabled)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    benchmark_gemm(torch.float32, MATRIX_SIZE, device,
                   'FP32 (TF32 enabled)', THEORETICAL_PEAKS['fp32_tf32'])

    # FP16
    torch.backends.cuda.matmul.allow_tf32 = False
    benchmark_gemm(torch.float16, MATRIX_SIZE, device,
                   'FP16 (Tensor Core)', THEORETICAL_PEAKS['fp16'])

    # BF16
    benchmark_gemm(torch.bfloat16, MATRIX_SIZE, device,
                   'BF16 (Tensor Core)', THEORETICAL_PEAKS['bf16'])

    print(f"{'='*65}\n")

if __name__ == '__main__':
    main()
```

### 2.2 多卡并行 GEMM（4096×4096，全部 8 卡）

```python
#!/usr/bin/env python3
# 脚本：02b_gemm_multi_gpu.py
# 用途：测试 8 卡并行 GEMM 算力，对应报告第 2.2 节
# 注意：检测 GPU 0 偶发性算力异常和 GPU 2/3 偏低问题

import torch
import threading
import time

MATRIX_SIZE = 4096
WARMUP_ITERS = 3
TEST_ITERS = 10
DTYPE = torch.float16
PEAK_TFLOPS = 989  # FP16 Tensor Core 非稀疏峰值

def benchmark_gpu(gpu_id, results):
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
    results[gpu_id] = tflops

def main():
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*65}")
    print(f"  多卡并行 GEMM — FP16  {MATRIX_SIZE}×{MATRIX_SIZE}  共 {num_gpus} 卡")
    print(f"{'='*65}")

    results = {}
    threads = []
    for i in range(num_gpus):
        t = threading.Thread(target=benchmark_gpu, args=(i, results))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = 0
    for i in range(num_gpus):
        tflops = results[i]
        efficiency = tflops / PEAK_TFLOPS * 100
        if tflops < 100:
            status = "⚠️ 严重异常"
        elif efficiency < 70:
            status = "⚠️ 偏低"
        else:
            status = "✅ 正常"
        print(f"  GPU {i}:  {tflops:>8.1f} TFLOPS  效率={efficiency:>5.1f}%  {status}")
        total += tflops

    print(f"  {'─'*50}")
    print(f"  合计:   {total:>8.1f} TFLOPS")
    print(f"{'='*65}\n")

if __name__ == '__main__':
    main()
```

---

## 3. HBM3 显存带宽测试

```python
#!/usr/bin/env python3
# 脚本：03_memory_bandwidth.py
# 用途：测试 HBM3 显存带宽，对应报告第 3.1 节
# 理论峰值：H100 SXM5 = 3350 GB/s

import torch
import time

THEORETICAL_BW_GBS = 3350  # H100 SXM5 HBM3 理论带宽 GB/s
WARMUP_ITERS = 10
TEST_ITERS = 30

TEST_SIZES_MB = [256, 1024]  # 测试规模：256MB 和 1GB

def benchmark_bandwidth(size_mb, device):
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
    efficiency = bw_gbs / THEORETICAL_BW_GBS * 100
    status = "✅ 优秀" if efficiency >= 80 else ("✅ 良好" if efficiency >= 70 else "⚠️ 偏低")

    size_label = f"{size_mb}MB" if size_mb < 1024 else f"{size_mb//1024}GB"
    print(f"  {size_label} copy:  {bw_gbs:>8.1f} GB/s  "
          f"理论={THEORETICAL_BW_GBS} GB/s  达标率={efficiency:.1f}%  {status}")
    return bw_gbs

def main():
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*65}")
    print(f"  HBM3 显存带宽测试  (迭代: {TEST_ITERS} 次)")
    print(f"{'='*65}")

    for gpu_id in range(num_gpus):
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"\n  GPU {gpu_id}: {gpu_name}")
        for size_mb in TEST_SIZES_MB:
            benchmark_bandwidth(size_mb, device)

    print(f"\n{'='*65}\n")

if __name__ == '__main__':
    main()
```

---

## 4. 磁盘 I/O 性能测试

```bash
#!/bin/bash
# 脚本：04_disk_io.sh
# 用途：测试磁盘顺序读写速度，对应报告第 3.2 节
# 说明：测试 overlay 层和 NVMe 路径，4GB 文件，Direct IO

echo "========================================"
echo "  磁盘 I/O 性能测试 (Direct IO, 4GB)"
echo "========================================"

TEST_FILE_SIZE="4G"
BLOCK_SIZE="1G"

test_path() {
    local path=$1
    local label=$2

    if [ ! -d "$path" ]; then
        echo "  [$label] 路径不存在，跳过: $path"
        return
    fi

    echo ""
    echo "  [$label] 路径: $path"

    # 顺序写测试
    echo -n "    顺序写 ($TEST_FILE_SIZE): "
    WRITE_RESULT=$(dd if=/dev/zero of="${path}/test_write.tmp" \
        bs=$BLOCK_SIZE count=4 oflag=direct conv=fdatasync 2>&1 | \
        grep -oP '\d+(\.\d+)? [GM]B/s' | tail -1)
    echo "${WRITE_RESULT:-测试失败}"

    sync

    # 顺序读测试（先清缓存）
    echo -n "    顺序读 ($TEST_FILE_SIZE): "
    # 清页缓存（需 root）
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    READ_RESULT=$(dd if="${path}/test_write.tmp" of=/dev/null \
        bs=$BLOCK_SIZE iflag=direct 2>&1 | \
        grep -oP '\d+(\.\d+)? [GM]B/s' | tail -1)
    echo "${READ_RESULT:-测试失败}"

    # 清理
    rm -f "${path}/test_write.tmp"
}

# 测试 overlay 文件系统（容器层）
test_path "/tmp" "overlay/容器层 (/tmp)"

# 测试 NVMe 路径（挂载点，按实际路径调整）
test_path "/anc-init" "NVMe 分区 (/anc-init)"

# 测试 NFS 共享（按实际挂载点调整）
test_path "/root/public" "NFS 共享 (/root/public)"

echo ""
echo "参考值："
echo "  overlay 写: ~1.1 GB/s | 读: ~1.9 GB/s（容器层受限）"
echo "  NVMe 写:    ~3-7 GB/s | 读: ~3-7 GB/s（裸 NVMe）"
echo "========================================"
```

---

## 5. GPU 拓扑与 NVLink 检查

```bash
#!/bin/bash
# 脚本：05_gpu_topology.sh
# 用途：检查 GPU 拓扑、NVLink 互联状态，对应报告第 4.1 节
# 期望：所有 GPU 对之间为 NV18（NVLink 4.0 × 18 全互联）

echo "========================================"
echo "  GPU 拓扑与 NVLink 互联检查"
echo "========================================"

echo ""
echo "--- GPU 拓扑矩阵 ---"
nvidia-smi topo -m

echo ""
echo "--- NVLink 状态（逐卡）---"
for i in $(seq 0 7); do
    echo "  GPU $i NVLink 链路状态:"
    nvidia-smi nvlink --status -i $i 2>/dev/null | grep -E "(Link|Active|Inactive)" | head -5
done

echo ""
echo "--- NVLink 带宽统计 ---"
nvidia-smi nvlink --capabilities -i 0 2>/dev/null | head -10

echo ""
echo "--- NUMA 亲和性 ---"
for i in $(seq 0 7); do
    numa=$(cat /sys/bus/pci/devices/$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $i | \
        tr '[:upper:]' '[:lower:]' | sed 's/00000000://')/numa_node 2>/dev/null || echo "N/A")
    echo "  GPU $i -> NUMA node: $numa"
done

echo ""
echo "期望拓扑："
echo "  GPU 0-7 两两之间：NV18（NVLink 4.0 全互联）"
echo "  GPU 0-3 -> NUMA 0，GPU 4-7 -> NUMA 1"
echo "========================================"
```

---

## 6. NCCL 多卡通信性能测试

### 6.1 使用 nccl-tests 工具（推荐）

```bash
#!/bin/bash
# 脚本：06a_nccl_tests.sh
# 用途：使用 nccl-tests 测试 AllReduce 和 AllGather 带宽，对应报告第 4.2/4.3 节
# 前置：需编译 nccl-tests (https://github.com/NVIDIA/nccl-tests)

NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-"./nccl-tests/build"}
NUM_GPUS=8

# 检查工具是否存在
if [ ! -f "${NCCL_TESTS_DIR}/all_reduce_perf" ]; then
    echo "nccl-tests 未找到，请先编译："
    echo "  git clone https://github.com/NVIDIA/nccl-tests"
    echo "  cd nccl-tests && make -j"
    exit 1
fi

echo "========================================"
echo "  NCCL 多卡通信性能测试（${NUM_GPUS} GPU）"
echo "========================================"

echo ""
echo "--- AllReduce 性能 ---"
echo "  期望（2GB）: 总线带宽 ~472 GB/s"
${NCCL_TESTS_DIR}/all_reduce_perf \
    -b 8 -e 2G -f 2 \
    -g ${NUM_GPUS} \
    -c 1 -w 5 -n 20

echo ""
echo "--- AllGather 性能 ---"
echo "  期望（2GB）: 总线带宽 ~350 GB/s"
${NCCL_TESTS_DIR}/all_gather_perf \
    -b 8 -e 2G -f 2 \
    -g ${NUM_GPUS} \
    -c 1 -w 5 -n 20

echo ""
echo "--- ReduceScatter 性能 ---"
${NCCL_TESTS_DIR}/reduce_scatter_perf \
    -b 8 -e 2G -f 2 \
    -g ${NUM_GPUS} \
    -c 1 -w 5 -n 20
```

### 6.2 纯 PyTorch NCCL 测试（无需额外编译）

```python
#!/usr/bin/env python3
# 脚本：06b_nccl_pytorch.py
# 用途：使用 PyTorch dist 测试 NCCL AllReduce/AllGather，对应报告第 4.2/4.3 节
# 运行：torchrun --nproc_per_node=8 06b_nccl_pytorch.py

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
    bus_bw = algo_bw * world_size / (2 * (world_size - 1)) * 2 * (world_size - 1) / world_size
    # 简化：bus_bw ≈ algo_bw * world_size / (2*(world_size-1)) * (world_size-1)
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
```

**运行命令：**
```bash
torchrun --nproc_per_node=8 06b_nccl_pytorch.py
```

---

## 7. GPU-Burn 稳定性压力测试

### 7.1 使用 gpu-burn 工具

```bash
#!/bin/bash
# 脚本：07a_gpu_burn.sh
# 用途：120 秒满载压力测试，检测稳定性和错误，对应报告第 5.1 节
# 前置：需编译 gpu-burn (https://github.com/wilicc/gpu-burn)

GPU_BURN_DIR=${GPU_BURN_DIR:-"./gpu-burn"}
DURATION=120

if [ ! -f "${GPU_BURN_DIR}/gpu_burn" ]; then
    echo "gpu-burn 未找到，请先编译："
    echo "  git clone https://github.com/wilicc/gpu-burn"
    echo "  cd gpu-burn && make"
    exit 1
fi

echo "========================================"
echo "  GPU-Burn 压力测试（${DURATION}秒）"
echo "  期望：各卡 ~51,000+ Gflop/s，错误数=0"
echo "========================================"

# 同时监控温度（后台）
monitor_temps() {
    while true; do
        echo "[$(date '+%H:%M:%S')] 温度: $(nvidia-smi \
            --query-gpu=index,temperature.gpu,power.draw,clocks.sm \
            --format=csv,noheader | \
            awk -F', ' '{printf "GPU%s=%s°C/%sW/%sMHz  ", $1,$2,$3,$4}')"
        sleep 10
    done
}

monitor_temps &
MONITOR_PID=$!

# 运行 gpu-burn
cd "${GPU_BURN_DIR}"
./gpu_burn ${DURATION}
BURN_EXIT=$?

kill $MONITOR_PID 2>/dev/null

echo ""
echo "--- 测试结束温度 ---"
nvidia-smi --query-gpu=index,temperature.gpu,power.draw \
    --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %s°C  %s\n", $1, $2, $3}'

echo ""
echo "评估标准："
echo "  算力: 各卡 >50,000 Gflop/s ✅"
echo "  温度: <83°C（热节流阈值）✅"
echo "  错误: 0 ✅"
echo "========================================"
exit $BURN_EXIT
```

### 7.2 PyTorch 替代压力测试（无需 gpu-burn）

```python
#!/usr/bin/env python3
# 脚本：07b_stress_test_pytorch.py
# 用途：用 PyTorch 实现等效压力测试，测试 120 秒持续算力和稳定性

import torch
import threading
import time
import subprocess

DURATION_SECONDS = 120
MATRIX_SIZE = 8192
DTYPE = torch.float16
NUM_GPUS = torch.cuda.device_count()

def stress_gpu(gpu_id, results, stop_event):
    device = torch.device(f'cuda:{gpu_id}')
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=DTYPE, device=device)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=DTYPE, device=device)

    iters = 0
    start = time.perf_counter()

    while not stop_event.is_set():
        C = torch.mm(A, B)
        iters += 1

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    flops_per_iter = 2 * MATRIX_SIZE ** 3
    gflops = flops_per_iter * iters / elapsed / 1e9
    results[gpu_id] = gflops

def get_gpu_temps():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,temperature.gpu,power.draw',
         '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def main():
    print(f"\n{'='*65}")
    print(f"  PyTorch GPU 压力测试（{DURATION_SECONDS}秒，{NUM_GPUS} 卡）")
    print(f"  矩阵规模: {MATRIX_SIZE}×{MATRIX_SIZE}  精度: FP16")
    print(f"{'='*65}")

    results = {}
    stop_event = threading.Event()
    threads = [threading.Thread(target=stress_gpu, args=(i, results, stop_event))
               for i in range(NUM_GPUS)]

    for t in threads:
        t.start()

    # 定时打印温度
    start_time = time.perf_counter()
    print(f"\n  时间      GPU温度与功耗")
    while time.perf_counter() - start_time < DURATION_SECONDS:
        elapsed = int(time.perf_counter() - start_time)
        temps = get_gpu_temps()
        temp_summary = "  ".join([
            f"GPU{line.split(',')[0].strip()}={line.split(',')[1].strip()}°C"
            for line in temps.split('\n') if line
        ])
        print(f"  [{elapsed:>3}s] {temp_summary}")
        time.sleep(10)

    stop_event.set()
    for t in threads:
        t.join()

    print(f"\n  {'─'*55}")
    print(f"  GPU    持续算力 (Gflop/s)    最终温度    评价")
    print(f"  {'─'*55}")

    total = 0
    for i in range(NUM_GPUS):
        gflops = results.get(i, 0)
        total += gflops
        temps_raw = subprocess.run(
            ['nvidia-smi', f'--id={i}', '--query-gpu=temperature.gpu',
             '--format=csv,noheader'], capture_output=True, text=True
        ).stdout.strip()
        status = "✅ OK" if gflops > 48000 else "⚠️ 偏低"
        print(f"  GPU {i}  {gflops:>12,.0f}           {temps_raw:>4}°C      {status}")

    print(f"  {'─'*55}")
    print(f"  合计   {total:>12,.0f} Gflop/s (~{total/1000:.0f} TFLOPS)")
    print(f"\n  期望: 各卡 >50,000 Gflop/s，合计 >400,000 Gflop/s")
    print(f"{'='*65}\n")

if __name__ == '__main__':
    main()
```

---

## 8. GPU 时钟频率与温度监控

```bash
#!/bin/bash
# 脚本：08_clock_temp_monitor.sh
# 用途：监控 GPU 时钟频率、温度、功耗，对应报告第 5.2/5.3 节
# 期望：负载态核心频率 1980 MHz，显存 2619 MHz，温度 <83°C

echo "========================================"
echo "  GPU 时钟频率与温度监控"
echo "========================================"

echo ""
echo "--- 空载状态 ---"
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate \
    --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: 核心=%s  显存=%s  温度=%s  功耗=%s  状态=%s\n",
                 $1,$2,$3,$4,$5,$6}'

echo ""
echo "--- 最大允许频率 ---"
nvidia-smi --query-gpu=index,clocks.max.sm,clocks.max.mem \
    --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: 最大核心=%s  最大显存=%s\n", $1,$2,$3}'

echo ""
echo "--- 连续监控（Ctrl+C 停止）---"
echo "  时间        GPU  核心(MHz)  显存(MHz)  温度(°C)  功耗(W)   P-State"
echo "  ─────────────────────────────────────────────────────────────────"

while true; do
    timestamp=$(date '+%H:%M:%S')
    nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate \
        --format=csv,noheader | \
        while IFS=',' read -r gpu_id sm_clk mem_clk temp power pstate; do
            # 检查是否有降频
            sm_mhz=$(echo $sm_clk | grep -oP '\d+')
            flag=""
            [ "$sm_mhz" -lt 1900 ] 2>/dev/null && flag="⚠️ 降频"
            printf "  %s  GPU%-2s  %-9s  %-9s  %-8s  %-8s  %s %s\n" \
                "$timestamp" "$gpu_id" "$sm_clk" "$mem_clk" "$temp" "$power" "$pstate" "$flag"
        done
    sleep 5
done
```

---

## 9. Transformer 推理吞吐量测试

```python
#!/usr/bin/env python3
# 脚本：09_inference_throughput.py
# 用途：测试 Transformer 推理吞吐量，对应报告第 6.1 节
# 配置：12层, d_model=1024, seq_len=256, FP16
# 期望：batch=16 时约 1,481,521 tokens/s

import torch
import torch.nn as nn
import time

# 模型配置（与报告一致）
NUM_LAYERS    = 12
D_MODEL       = 1024
NHEAD         = 16
DIM_FEEDFORWARD = 4096
SEQ_LEN       = 256
DTYPE         = torch.float16
DEVICE        = torch.device('cuda:0')

WARMUP_ITERS  = 10
TEST_ITERS    = 50

BATCH_SIZES   = [1, 4, 16, 32]

def build_model():
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL,
        nhead=NHEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        batch_first=True,
        dtype=DTYPE,
        device=DEVICE,
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
    model.eval()
    return model

@torch.no_grad()
def benchmark_inference(model, batch_size):
    x = torch.randn(batch_size, SEQ_LEN, D_MODEL, dtype=DTYPE, device=DEVICE)

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(TEST_ITERS):
        out = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_latency_ms = elapsed / TEST_ITERS * 1000
    tokens_per_sec = batch_size * SEQ_LEN * TEST_ITERS / elapsed

    return tokens_per_sec, avg_latency_ms

def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n{'='*70}")
    print(f"  Transformer 推理吞吐量测试")
    print(f"  模型: {NUM_LAYERS}层 TransformerEncoder | d={D_MODEL} | seq={SEQ_LEN} | FP16")
    print(f"  设备: {gpu_name}")
    print(f"{'='*70}")
    print(f"\n  {'Batch':>8}  {'吞吐量(tokens/s)':>20}  {'延迟(ms)':>10}  评价")
    print(f"  {'─'*58}")

    model = build_model()

    prev_tps = None
    for bs in BATCH_SIZES:
        tps, latency = benchmark_inference(model, bs)
        if prev_tps is not None:
            growth = (tps - prev_tps) / prev_tps * 100
            if growth < 5:
                status = "✅ 接近饱和"
            elif tps > 1_000_000:
                status = "✅ 优秀"
            else:
                status = "✅ 良好"
        else:
            status = "✅ 低延迟基准"

        print(f"  {bs:>8}  {tps:>20,.0f}  {latency:>10.1f}  {status}")
        prev_tps = tps

    print(f"\n  参考值（H100 SXM5, 单卡）：")
    print(f"    batch=1:   ~134,563 tokens/s  延迟 ~1.9ms")
    print(f"    batch=4:   ~531,977 tokens/s  延迟 ~1.9ms")
    print(f"    batch=16: ~1,481,521 tokens/s  延迟 ~2.8ms")
    print(f"    batch=32: ~1,515,143 tokens/s  延迟 ~5.4ms")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
```

---

## 10. 一键运行全部测试

```bash
#!/bin/bash
# 脚本：run_all_tests.sh
# 用途：按顺序执行全部测试，生成综合报告
# 运行：bash run_all_tests.sh 2>&1 | tee gpu_benchmark_result_$(date +%Y%m%d_%H%M%S).log

set -e

REPORT_FILE="gpu_benchmark_result_$(date +%Y%m%d_%H%M%S).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
section() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════════════════"
}

section "GPU 服务器性能评测 — 开始"
log "测试机器: $(hostname)"
log "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "GPU 数量: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"

# ── 1. 环境检查 ──────────────────────────────────────────
section "1/7  环境检查与硬件信息"
bash "${SCRIPT_DIR}/01_env_check.sh"

# ── 2. 单卡 GEMM ─────────────────────────────────────────
section "2/7  GEMM 矩阵乘法基准"
python3 "${SCRIPT_DIR}/02a_gemm_single_gpu.py"
python3 "${SCRIPT_DIR}/02b_gemm_multi_gpu.py"

# ── 3. 显存带宽 ──────────────────────────────────────────
section "3/7  HBM3 显存带宽"
python3 "${SCRIPT_DIR}/03_memory_bandwidth.py"

# ── 4. 磁盘 I/O ──────────────────────────────────────────
section "4/7  磁盘 I/O 性能"
bash "${SCRIPT_DIR}/04_disk_io.sh"

# ── 5. GPU 拓扑 ──────────────────────────────────────────
section "5/7  GPU 拓扑与 NVLink"
bash "${SCRIPT_DIR}/05_gpu_topology.sh"

# ── 6. NCCL 通信 ─────────────────────────────────────────
section "6/7  NCCL 多卡通信"
if command -v torchrun &>/dev/null; then
    torchrun --nproc_per_node=8 "${SCRIPT_DIR}/06b_nccl_pytorch.py"
else
    log "torchrun 未找到，跳过 NCCL 测试"
fi

# ── 7. 推理吞吐 ──────────────────────────────────────────
section "7/7  Transformer 推理吞吐量"
python3 "${SCRIPT_DIR}/09_inference_throughput.py"

section "全部测试完成"
log "结果已输出，请与报告期望值对照"
echo ""
echo "  关键参考值（8×H100 SXM5）："
echo "    FP16 GEMM 单卡:     ~767 TFLOPS  (峰值 78%)"
echo "    HBM3 带宽:          ~2826 GB/s   (峰值 84%)"
echo "    NCCL AllReduce 2GB: ~472 GB/s 总线带宽"
echo "    GPU-Burn 持续算力:  ~51,500 Gflop/s / 卡"
echo "    推理 batch=16:      ~1,481,521 tokens/s"
echo ""
```

---

## 附录：依赖安装

```bash
# Python 依赖（PyTorch 已预装）
pip install torch  # 已有 2.8.0+cu128，无需重装

# nccl-tests（用于 AllReduce/AllGather 精确测试）
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests
make -j MPI=0 NCCL_HOME=/usr/local/nccl
export NCCL_TESTS_DIR=$(pwd)/build

# gpu-burn（稳定性压力测试）
git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
make
export GPU_BURN_DIR=$(pwd)
```

## 附录：期望结果汇总

| 测试项 | 期望值 | 评级 |
|--------|--------|------|
| FP16 GEMM 单卡 | ≥ 750 TFLOPS | ⭐⭐⭐⭐ 良好 |
| BF16/TF32 GEMM 单卡 | ≥ 780 TFLOPS | ⭐⭐⭐⭐ 良好 |
| HBM3 带宽 | ≥ 2700 GB/s | ⭐⭐⭐⭐⭐ 优秀 |
| NCCL AllReduce 2GB | ≥ 450 GB/s 总线 | ⭐⭐⭐⭐⭐ 优秀 |
| NCCL AllGather 2GB | ≥ 340 GB/s 总线 | ⭐⭐⭐⭐⭐ 优秀 |
| GPU-Burn 各卡算力 | ≥ 50,000 Gflop/s | ⭐⭐⭐⭐⭐ 优秀 |
| 满载温度 | < 83°C | ⭐⭐⭐⭐⭐ 优秀 |
| 推理 batch=16 | ≥ 1,400,000 tokens/s | ⭐⭐⭐⭐⭐ 优秀 |
| 磁盘顺序读（NVMe） | ≥ 3 GB/s | ⭐⭐⭐ 需用 NVMe 路径 |
