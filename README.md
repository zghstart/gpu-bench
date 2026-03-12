# GPU 服务器性能基准测试工具

本项目是一个全面的 GPU 服务器性能基准测试工具集，用于评估 NVIDIA GPU（如 H100、H200、B200、B300 等）的性能表现。

## 项目结构

```
gpu-bench/
├── scripts/              # 测试脚本目录
│   ├── 01_env_check.sh          # 环境检查脚本
│   ├── 02a_gemm_single_gpu.py   # 单 GPU GEMM 性能测试
│   ├── 02a_gemm_single_gpu_dense.py # 单 GPU 密集 GEMM 测试
│   ├── 02b_gemm_multi_gpu.py    # 多 GPU GEMM 性能测试
│   ├── 03_memory_bandwidth.py    # HBM 内存带宽测试
│   ├── 04_disk_io.sh             # 磁盘 I/O 性能测试
│   ├── 05_gpu_topology.sh        # GPU 拓扑和 NVLink 测试
│   ├── 06b_nccl_pytorch.py       # NCCL 通信性能测试
│   ├── 07_network_performance.sh # 网络性能测试
│   ├── 07b_stress_test_pytorch.py # GPU 压力测试
│   ├── 08_clock_temp_monitor.sh  # 时钟和温度监控
│   ├── 09_inference_throughput.py # 推理吞吐量测试
│   ├── generate_report.py        # 生成测试报告
│   ├── gpu_config.py             # GPU 配置文件
│   └── run_all_tests.sh          # 运行所有测试的脚本
├── .gitignore
├── .mcp.json
├── CLAUDE.md
├── GPU服务器性能评测脚本.md
├── 九章云极云 GPU 服务器性能评测报告.docx
└── README.md                    # 本文档
```

## 测试内容

### 1. 环境检查 (`01_env_check.sh`)
- 检查系统基本信息（CPU、内存、内核版本）
- 检查 GPU 信息（型号、驱动、CUDA 版本）
- 检查磁盘空间和文件系统
- 检查 NUMA 节点内存信息
- 检查网络接口和 InfiniBand 状态

**用法**：
```bash
bash 01_env_check.sh
```

### 2. 单 GPU GEMM 性能测试 (`02a_gemm_single_gpu.py`)
- 测试单 GPU 的矩阵乘法性能
- 支持 FP32、FP16、BF16 精度
- 自动检测 GPU 类型并使用相应的理论峰值进行评估

**用法**：
```bash
python 02a_gemm_single_gpu.py <gpu_id>
```

**参数**：
- `gpu_id`：要测试的 GPU ID（从 0 开始）

### 3. 单 GPU 密集 GEMM 测试 (`02a_gemm_single_gpu_dense.py`)
- 测试单 GPU 的密集矩阵乘法性能
- 与标准 GEMM 测试类似，但使用不同的矩阵尺寸

**用法**：
```bash
python 02a_gemm_single_gpu_dense.py <gpu_id>
```

**参数**：
- `gpu_id`：要测试的 GPU ID（从 0 开始）

### 4. 多 GPU GEMM 性能测试 (`02b_gemm_multi_gpu.py`)
- 测试多 GPU 的并行矩阵乘法性能
- 自动检测所有可用 GPU
- 显示每个 GPU 的性能和整体效率

**用法**：
```bash
python 02b_gemm_multi_gpu.py
```

### 5. 内存带宽测试 (`03_memory_bandwidth.py`)
- 测试 GPU HBM 内存带宽
- 使用不同大小的内存拷贝测试（256MB、1GB、4GB、16GB）

**用法**：
```bash
python 03_memory_bandwidth.py <gpu_id>
```

**参数**：
- `gpu_id`：要测试的 GPU ID（从 0 开始）

### 6. 磁盘 I/O 性能测试 (`04_disk_io.sh`)
- 测试磁盘顺序读写性能
- 测试不同路径的 I/O 性能（临时目录、NVMe 分区、NFS 共享）

**用法**：
```bash
bash 04_disk_io.sh
```

### 7. GPU 拓扑和 NVLink 测试 (`05_gpu_topology.sh`)
- 显示 GPU 拓扑矩阵
- 检查 NVLink 状态和带宽
- 显示 NUMA 亲和性

**用法**：
```bash
bash 05_gpu_topology.sh
```

### 8. NCCL 通信性能测试 (`06b_nccl_pytorch.py`)
- 测试 GPU 间 NCCL 通信性能
- 支持 AllReduce 操作的延迟和带宽测试

**用法**：
```bash
torchrun --nproc_per_node=<gpu_count> 06b_nccl_pytorch.py
```

**参数**：
- `gpu_count`：要使用的 GPU 数量

### 9. 网络性能测试 (`07_network_performance.sh`)
- 测试以太网性能（延迟和带宽）
- 测试 InfiniBand 性能

**用法**：
```bash
bash 07_network_performance.sh
```

### 10. GPU 压力测试 (`07b_stress_test_pytorch.py`)
- 测试 GPU 在 120 秒持续负载下的稳定性
- 监控温度和功耗
- 支持指定要测试的 GPU

**用法**：
```bash
python 07b_stress_test_pytorch.py <gpu_ids>
```

**参数**：
- `gpu_ids`：要测试的 GPU ID，多个 ID 用空格分隔（如 "0 3 6"）

### 11. 时钟和温度监控 (`08_clock_temp_monitor.sh`)
- 监控 GPU 时钟频率和温度
- 定期输出监控数据

**用法**：
```bash
bash 08_clock_temp_monitor.sh
```

### 12. 推理吞吐量测试 (`09_inference_throughput.py`)
- 测试 GPU 的模型推理吞吐量
- 支持不同批量大小的测试

**用法**：
```bash
python 09_inference_throughput.py <gpu_id>
```

**参数**：
- `gpu_id`：要测试的 GPU ID（从 0 开始）

### 13. 生成测试报告 (`generate_report.py`)
- 汇总所有测试结果生成报告
- 支持不同格式的报告输出

**用法**：
```bash
python generate_report.py
```

### 14. 运行所有测试 (`run_all_tests.sh`)
- 按顺序运行所有测试脚本
- 自动处理测试参数
- 生成综合测试报告

**用法**：
```bash
bash run_all_tests.sh
```

## 系统要求

- NVIDIA GPU（H100、H200、B200、B300 等）
- CUDA 11.0 或更高版本
- Python 3.8 或更高版本
- PyTorch 1.10 或更高版本
- NCCL 2.0 或更高版本
- 必要的系统工具：nvidia-smi、ibstat、iperf3 等

## 安装步骤

1. 克隆项目到本地：
   ```bash
   git clone <repository_url>
   cd gpu-bench
   ```

2. 安装依赖：
   ```bash
   # 安装 PyTorch 和相关依赖
   pip install torch torchvision torchaudio
   
   # 安装其他必要的包
   pip install numpy psutil
   ```

3. 赋予脚本执行权限：
   ```bash
   chmod +x scripts/*.sh
   chmod +x scripts/*.py
   ```

## 测试流程

1. **环境检查**：首先运行 `01_env_check.sh` 确认系统环境
2. **基础性能测试**：运行单 GPU 和多 GPU GEMM 测试
3. **内存带宽测试**：运行 `03_memory_bandwidth.py` 测试内存性能
4. **其他测试**：根据需要运行磁盘 I/O、网络、拓扑等测试
5. **压力测试**：运行 `07b_stress_test_pytorch.py` 测试稳定性
6. **生成报告**：运行 `generate_report.py` 汇总结果

或者直接运行 `run_all_tests.sh` 执行完整测试流程。

## 注意事项

1. 测试前请确保 GPU 没有被其他进程占用，避免测试结果不准确
2. 部分测试可能需要较长时间，请耐心等待
3. 对于 NCCL 测试，需要使用 `torchrun` 启动
4. 压力测试会使 GPU 长时间高负载，请确保散热良好
5. 测试结果会受到系统负载、温度等因素影响，请在相同条件下进行对比

## 结果解读

- **GEMM 性能**：以 TFLOPS 为单位，越接近理论峰值越好
- **内存带宽**：以 GB/s 为单位，越接近理论带宽越好
- **NCCL 性能**：延迟越低、带宽越高越好
- **压力测试**：持续性能稳定、温度在合理范围内为正常

## 支持的 GPU 型号

- NVIDIA H100
- NVIDIA H200
- NVIDIA B200
- NVIDIA B300
- 其他支持 CUDA 的 NVIDIA GPU

## 故障排除

1. **CUDA 错误**：检查 CUDA 版本和驱动是否匹配
2. **内存不足**：调整测试中的矩阵大小或批量大小
3. **NCCL 错误**：检查网络连接和 NCCL 版本
4. **权限问题**：确保有足够的权限运行测试脚本

## 联系方式

如有问题或建议，请联系项目维护者。

---

**版本**：1.0.0
**更新日期**：2026-03-12
**适用平台**：Linux
**支持 GPU**：NVIDIA H100/H200/B200/B300 等