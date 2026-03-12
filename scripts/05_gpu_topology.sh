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
