#!/usr/bin/env python3
# 脚本：09_inference_throughput.py
# 用途：测试 Transformer 推理吞吐量
# 支持 H100/H200/B200/B300，自动检测 GPU 类型

import torch
import torch.nn as nn
import time
import sys
import os

# 添加 scripts 目录到路径以导入 gpu_config
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from gpu_config import detect_gpu_type

# 模型配置
NUM_LAYERS    = 12
D_MODEL       = 1024
NHEAD         = 16
DIM_FEEDFORWARD = 4096
SEQ_LEN       = 256
DTYPE         = torch.float16

WARMUP_ITERS  = 10
TEST_ITERS    = 50

BATCH_SIZES   = [1, 4, 16, 32]


def build_model(device):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL,
        nhead=NHEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        batch_first=True,
        dtype=DTYPE,
        device=device,
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
    model.eval()
    return model


@torch.no_grad()
def benchmark_inference(model, batch_size, device):
    x = torch.randn(batch_size, SEQ_LEN, D_MODEL, dtype=DTYPE, device=device)

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
    num_gpus = torch.cuda.device_count()

    for gpu_id in range(num_gpus):
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_info = detect_gpu_type(gpu_id)

        print(f"\n{'='*70}")
        print(f"  Transformer 推理吞吐量测试 (GPU {gpu_id})")
        print(f"  模型: {NUM_LAYERS}层 TransformerEncoder | d={D_MODEL} | seq={SEQ_LEN} | FP16")
        print(f"  设备: {gpu_name}")
        print(f"  GPU 类型: {gpu_info['name']}")
        print(f"{'='*70}")
        print(f"\n  {'Batch':>8}  {'吞吐量(tokens/s)':>20}  {'延迟(ms)':>10}  评价")
        print(f"  {'─'*58}")

        model = build_model(device)

        prev_tps = None
        for bs in BATCH_SIZES:
            tps, latency = benchmark_inference(model, bs, device)
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

        print(f"\n  说明: 吞吐量受 GPU 算力和显存带宽影响")
        print(f"  H100 参考: batch=16 时约 1,481,521 tokens/s")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
