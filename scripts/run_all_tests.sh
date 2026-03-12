#!/bin/bash
# 脚本：run_all_tests.sh
# 用途：按顺序执行全部测试，生成综合报告
# 运行：bash run_all_tests.sh 2>&1 | tee gpu_benchmark_result_$(date +%Y%m%d_%H%M%S).log

set -e

REPORT_FILE="gpu_benchmark_result_$(date +%Y%m%d_%H%M%S).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 获取 GPU 数量
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 2>/dev/null || echo 0)

log() { 
    local timestamp=$(date '+%H:%M:%S')
    echo "[${timestamp}] $*"
    echo "[${timestamp}] $*" >> "${LOG_DIR}/test_run_$(date +%Y%m%d).log"
}

section() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════════════════"
    log "开始测试: $*"
}

error_handler() {
    local error_code=$1
    local command=$2
    log "错误: 命令执行失败，错误码: ${error_code}"
    log "失败的命令: ${command}"
    echo "\n⚠️  测试执行过程中出现错误，请查看日志获取详细信息\n"
}

# 设置错误处理
trap 'error_handler $? "$BASH_COMMAND"' ERR

section "GPU 服务器性能评测 — 开始"
log "测试机器: $(hostname)"
log "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "GPU 数量: $NUM_GPUS"

# ── 1. 环境检查 ──────────────────────────────────────────
section "1/9  环境检查与硬件信息"
log "执行环境检查脚本..."
bash "${SCRIPT_DIR}/01_env_check.sh"
log "环境检查完成"

# ── 2. 单卡 GEMM ─────────────────────────────────────────
section "2/9  GEMM 矩阵乘法基准"
log "执行单卡GEMM测试..."
python3 "${SCRIPT_DIR}/02a_gemm_single_gpu.py"
log "单卡GEMM测试完成"
log "执行多卡GEMM测试..."
python3 "${SCRIPT_DIR}/02b_gemm_multi_gpu.py"
log "多卡GEMM测试完成"

# ── 3. 显存带宽 ──────────────────────────────────────────
section "3/9  HBM 显存带宽"
log "执行显存带宽测试..."
python3 "${SCRIPT_DIR}/03_memory_bandwidth.py"
log "显存带宽测试完成"

# ── 4. 磁盘 I/O ──────────────────────────────────────────
section "4/9  磁盘 I/O 性能"
log "执行磁盘I/O测试..."
bash "${SCRIPT_DIR}/04_disk_io.sh"
log "磁盘I/O测试完成"

# ── 5. GPU 拓扑 ──────────────────────────────────────────
section "5/9  GPU 拓扑与 NVLink"
log "执行GPU拓扑测试..."
bash "${SCRIPT_DIR}/05_gpu_topology.sh"
log "GPU拓扑测试完成"

# ── 6. NCCL 通信 ──────────────────────────────────────────
section "6/9  NCCL 多卡通信"
if command -v torchrun &>/dev/null; then
    log "执行NCCL多卡通信测试..."
    torchrun --nproc_per_node=${NUM_GPUS} "${SCRIPT_DIR}/06b_nccl_pytorch.py"
    log "NCCL多卡通信测试完成"
else
    log "torchrun 未找到，跳过 NCCL 测试"
fi

# ── 7. 网络性能 ──────────────────────────────────────────
section "7/9  网络性能测试"
log "执行网络性能测试..."
bash "${SCRIPT_DIR}/07_network_performance.sh"
log "网络性能测试完成"

# ── 8. 压力测试 ──────────────────────────────────────────
section "8/9  GPU 压力测试"
log "执行GPU压力测试..."
python3 "${SCRIPT_DIR}/07b_stress_test_pytorch.py"
log "GPU压力测试完成"

# ── 9. 推理吞吐 ──────────────────────────────────────────
section "9/9  Transformer 推理吞吐量"
log "执行推理吞吐量测试..."
python3 "${SCRIPT_DIR}/09_inference_throughput.py"
log "推理吞吐量测试完成"

section "全部测试完成"
log "结果已输出，请与报告期望值对照"
echo ""
echo "  说明: 参考值因 GPU 型号而异 (H100/H200/B200/B300)"
echo "  使用 gpu_config.py 可查看各型号理论峰值"
echo ""

# 生成标准化报告
section "生成测试报告"
log "正在生成标准化测试报告..."
python3 "${SCRIPT_DIR}/generate_report.py"
log "报告生成完成，请查看生成的HTML和JSON格式报告"
log "测试流程全部完成"

# 总结测试结果
log "测试总结:"
log "- 测试机器: $(hostname)"
log "- 测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "- GPU 数量: $NUM_GPUS"
log "- 测试结果已保存到: ${REPORT_FILE}"
log "- 详细日志已保存到: ${LOG_DIR}/test_run_$(date +%Y%m%d).log"

