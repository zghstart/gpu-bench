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
# 尝试使用numactl命令获取NUMA亲和性
if command -v numactl &>/dev/null; then
    echo "  使用 numactl 检查 NUMA 亲和性:"
    for i in $(seq 0 7); do
        # 获取GPU的PCI ID
        pci_id=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $i | tr '[:upper:]' '[:lower:]')
        # 提取PCI总线号和设备号
        bus=$(echo $pci_id | cut -d ':' -f 2)
        device=$(echo $pci_id | cut -d ':' -f 3 | cut -d '.' -f 1)
        # 构建PCI设备路径
        pci_path="0000:$bus:$device.0"
        # 读取NUMA节点
        numa=$(cat /sys/bus/pci/devices/$pci_path/numa_node 2>/dev/null || echo "N/A")
        echo "  GPU $i -> NUMA node: $numa"
    done
else
    echo "  numactl 未安装，尝试直接读取 sysfs:"
    for i in $(seq 0 7); do
        # 获取GPU的PCI ID
        pci_id=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $i)
        # 转换为小写
        pci_path=$(echo $pci_id | tr '[:upper:]' '[:lower:]')
        # 读取NUMA节点
        numa=$(cat /sys/bus/pci/devices/$pci_path/numa_node 2>/dev/null || echo "N/A")
        echo "  GPU $i -> NUMA node: $numa"
    done
fi

echo ""
echo "期望拓扑："
echo "  GPU 0-7 两两之间：NV18（NVLink 4.0 全互联）"
echo "  GPU 0-3 -> NUMA 0，GPU 4-7 -> NUMA 1"
echo "========================================"
