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

echo ""
echo "--- nvcc 编译器版本 ---"
# 尝试多种常见路径
if command -v nvcc &>/dev/null; then
    nvcc --version
else
    echo "nvcc not in PATH, searching common locations..."
    CUDA_PATHS=("/usr/local/cuda" "/usr/local/cuda-12" "/usr/local/cuda-11" "/usr/local/cuda-11.8" "/usr/local/cuda-12.0" "/usr/local/cuda-12.1" "/usr/local/cuda-12.2" "/usr/local/cuda-12.3" "/usr/local/cuda-12.4" "/usr/local/cuda-12.5" "/usr/local/cuda-12.6" "/usr/local/cuda-12.7" "/usr/local/cuda-12.8" "/usr/local/cuda-12.9")
    for cuda_path in "${CUDA_PATHS[@]}"; do
        if [ -f "$cuda_path/bin/nvcc" ]; then
            echo "Found: $cuda_path/bin/nvcc"
            $cuda_path/bin/nvcc --version
            break
        fi
    done
fi

echo ""
echo "--- PyTorch 版本 ---"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "--- 磁盘挂载信息 ---"
df -h

echo ""
echo "--- NUMA 节点内存信息 ---"
# 检查并安装 numactl
if ! command -v numactl &>/dev/null; then
    echo "numactl 未安装，尝试安装..."
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        apt-get update -qq && apt-get install -y -qq numactl 2>/dev/null
    elif [ -f /etc/redhat-release ]; then
        # RHEL/CentOS
        yum install -y numactl 2>/dev/null
    elif [ -f /etc/SuSE-release ]; then
        # SUSE
        zypper install -y numactl 2>/dev/null
    else
        echo "无法确定包管理器，请手动安装: apt-get install numactl / yum install numactl"
    fi
fi

if command -v numactl &>/dev/null; then
    numactl -H | grep -E "(hardware|available|node)"
else
    echo "numactl 安装失败，请手动安装"
fi

echo ""
echo "--- 网络接口 ---"
echo "=== InfiniBand 统计信息 ==="
ibstat 2>/dev/null || echo "ibstat not found"

echo ""
echo "=== RDMA 链路状态 ==="
if command -v rdma &>/dev/null; then
    rdma link show 2>/dev/null || ip link show | grep -E "^[0-9]+:|mlx5|ib"
else
    ip link show | grep -E "^[0-9]+:|mlx5|ib|rdma"
fi

echo ""
echo "=== 网络接口详细信息 ==="
ip link show 2>/dev/null | head -100
