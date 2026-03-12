#!/bin/bash
# 脚本：07_network_performance.sh
# 用途：测试网络性能，包括带宽和延迟
# 支持：以太网和InfiniBand网络

set -e

log() { echo "[$(date '+%H:%M:%S')] $*"; }
section() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════════════════"
}

section "网络性能测试"

# 检查是否安装了必要的工具
check_dependencies() {
    local missing=0
    
    for cmd in iperf3 ping ibstat; do
        if ! command -v $cmd &>/dev/null; then
            echo "⚠️  缺少工具: $cmd"
            missing=1
        fi
    done
    
    if [ $missing -eq 1 ]; then
        echo "建议安装: apt-get install iperf3 iputils-ping infiniband-diags"
        echo "继续执行可用的测试..."
    fi
}

# 测试以太网性能
test_ethernet() {
    section "以太网性能测试"
    
    # 显示网络接口信息
    echo "--- 网络接口信息 ---"
    ip link show | grep -E "^[0-9]+:"
    
    # 测试本地环回延迟
    echo "\n--- 本地环回延迟测试 ---"
    if command -v ping &>/dev/null; then
        ping -c 10 localhost | tail -3
    fi
    
    # 测试网络带宽（如果有iperf3）
    if command -v iperf3 &>/dev/null; then
        echo "\n--- 网络带宽测试 ---"
        echo "启动iperf3服务器..."
        iperf3 -s -D
        sleep 2
        
        echo "测试TCP带宽..."
        iperf3 -c localhost -t 10
        
        echo "测试UDP带宽..."
        iperf3 -c localhost -u -b 1G -t 10
        
        # 停止iperf3服务器
        pkill -f "iperf3 -s"
    fi
}

# 测试InfiniBand性能
test_infiniband() {
    section "InfiniBand性能测试"
    
    if command -v ibstat &>/dev/null; then
        echo "--- InfiniBand状态 ---"
        ibstat
        
        # 检查是否有活跃的InfiniBand接口
        if ibstat | grep -q "State: Active"; then
            echo "\n--- InfiniBand带宽测试 ---"
            # 尝试使用ib_write_bw测试InfiniBand带宽
            if command -v ib_write_bw &>/dev/null; then
                echo "启动ib_write_bw服务器..."
                ib_write_bw -d auto -F --report_gbits &
                BW_PID=$!
                sleep 2
                
                echo "测试InfiniBand带宽..."
                ib_write_bw -d auto -F --report_gbits localhost
                
                # 停止ib_write_bw服务器
                kill $BW_PID 2>/dev/null
            else
                echo "⚠️  ib_write_bw 未找到，无法测试InfiniBand带宽"
                echo "建议安装: apt-get install perftest"
            fi
        else
            echo "⚠️  未检测到活跃的InfiniBand接口"
        fi
    else
        echo "⚠️  ibstat 未找到，无法测试InfiniBand"
    fi
}

# 主函数
main() {
    check_dependencies
    test_ethernet
    test_infiniband
    
    section "网络性能测试完成"
    echo "网络性能测试结果已输出，请与预期值对照"
    echo "说明: 网络性能受硬件配置、驱动版本和网络拓扑影响"
}

main