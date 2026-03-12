#!/usr/bin/env python3
# 脚本：07b_stress_test_pytorch.py
# 用途：用 PyTorch 实现等效压力测试，测试 120 秒持续算力和稳定性
# 支持 H100/H200/B200/B300，自动检测 GPU 类型

import torch
import threading
import time
import subprocess
import sys
import os

# 添加 scripts 目录到路径以导入 gpu_config
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from gpu_config import detect_gpu_type

DURATION_SECONDS = 120
MATRIX_SIZE = 8192
DTYPE = torch.float16


def stress_gpu(gpu_id, results, stop_event, gpu_peaks):
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
    results[gpu_id] = {'gflops': gflops, 'peak': gpu_peaks.get(gpu_id, 989000)}


def get_gpu_temps():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,temperature.gpu,power.draw',
         '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def check_gpu_health():
    """检查GPU温度和功耗是否异常"""
    temps_output = get_gpu_temps()
    issues = []
    
    for line in temps_output.split('\n'):
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 3:
            continue
        
        gpu_id = parts[0].strip()
        temp = float(parts[1].strip())
        power = float(parts[2].strip())
        
        # 设置温度和功耗阈值
        MAX_TEMP = 90.0  # 摄氏度
        MAX_POWER = 700.0  # 瓦特
        
        if temp > MAX_TEMP:
            issues.append(f"GPU {gpu_id} 温度过高: {temp}°C")
        if power > MAX_POWER:
            issues.append(f"GPU {gpu_id} 功耗过高: {power}W")
    
    return issues


def main():
    num_gpus = torch.cuda.device_count()

    # 为每个 GPU 检测类型并获取 FP16 峰值 (转换为 Gflop/s)
    gpu_peaks = {}
    gpu_names = {}
    for i in range(num_gpus):
        info = detect_gpu_type(i)
        gpu_peaks[i] = info['fp16'] * 1000  # 转换为 Gflop/s
        gpu_names[i] = info['name']

    print(f"\n{'='*65}")
    print(f"  PyTorch GPU 压力测试（{DURATION_SECONDS}秒，{num_gpus} 卡）")
    print(f"  矩阵规模: {MATRIX_SIZE}×{MATRIX_SIZE}  精度: FP16")
    print(f"{'='*65}")
    
    # 检查初始GPU健康状态
    initial_issues = check_gpu_health()
    if initial_issues:
        print("\n  ⚠️  初始GPU状态异常:")
        for issue in initial_issues:
            print(f"    - {issue}")
        print("  继续测试，但请注意监控")

    results = {}
    stop_event = threading.Event()
    threads = [threading.Thread(target=stress_gpu, args=(i, results, stop_event, gpu_peaks))
               for i in range(num_gpus)]

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

    print(f"\n  {'─'*65}")
    print(f"  GPU    持续算力 (Gflop/s)    理论峰值    评价")
    print(f"  {'─'*65}")

    total = 0
    for i in range(num_gpus):
        data = results.get(i, {'gflops': 0, 'peak': 989000})
        gflops = data['gflops']
        peak = data['peak']
        total += gflops
        temps_raw = subprocess.run(
            ['nvidia-smi', f'--id={i}', '--query-gpu=temperature.gpu',
             '--format=csv,noheader'], capture_output=True, text=True
        ).stdout.strip()
        efficiency = gflops / peak * 100
        status = "✅ OK" if efficiency > 80 else ("⚠️ 良好" if efficiency > 70 else "⚠️ 偏低")
        print(f"  GPU {i}  {gflops:>12,.0f}        {peak:>6,.0f}   {status}")

    print(f"  {'─'*65}")
    print(f"  合计   {total:>12,.0f} Gflop/s (~{total/1000:.0f} TFLOPS)")
    
    # 检查测试后GPU健康状态
    final_issues = check_gpu_health()
    if final_issues:
        print("\n  ⚠️  测试后GPU状态异常:")
        for issue in final_issues:
            print(f"    - {issue}")
        print("  请检查GPU是否存在硬件问题")
    
    print(f"\n  说明: 根据检测到的 GPU 类型使用对应的理论峰值进行评估")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
