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
