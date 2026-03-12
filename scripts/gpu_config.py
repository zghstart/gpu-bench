#!/usr/bin/env python3
# 脚本：gpu_config.py
# 用途：定义不同 GPU 型号的理论峰值参数
# 支持: H100, H200, B200, B300
# 来源: NVIDIA 官方网站 https://www.nvidia.com/en-us/data-center/

import torch


def get_gpu_peak_info(gpu_name):
    """
    根据 GPU 名称返回理论峰值参数
    返回: {'fp32': X, 'fp32_tf32': X, 'fp16': X, 'bf16': X, 'fp8': X, 'hbm_bw': X}

    注意：
    1. NVIDIA 官网标注的 Tensor Core 算力均为 "With sparsity" (稀疏算力)
    2. 稠密算力约为稀疏算力的一半
    3. 不同显存版本（PCIe/SXM/NVL）性能不同
    """
    gpu_name_lower = gpu_name.lower()

    # ====================
    # H100 系列
    # ====================
    if 'h100' in gpu_name_lower:
        # H100 SXM5 - Dense vs Sparse compute power
        # - Dense FP32: 67 TFLOPS
        # - Dense TF32 (without sparsity): 495 TFLOPS
        # - Sparse TF32 (with sparsity): 989 TFLOPS
        # - Dense FP16/BF16 (without sparsity): 989 TFLOPS
        # - Sparse FP16/BF16 (with sparsity): 1979 TFLOPS
        if 'sxm' in gpu_name_lower or 'nvl' not in gpu_name_lower:
            return {
                'name': 'H100 SXM5',
                'fp32': 67,           # Dense FP32 (dense FP32)
                'fp32_tf32': 495,      # Dense TF32 (no sparsity)
                'fp16': 989,          # Dense FP16 (no sparsity)
                'bf16': 989,          # Dense BF16 (no sparsity)
                'fp8': 1958,          # Dense FP8 (no sparsity)
                'hbm_bw': 3350,       # GB/s
                'has_sparsity': True, # supports sparsity
            }
        # H100 NVL - PCIe version
        elif 'nvl' in gpu_name_lower or 'pcie' in gpu_name_lower:
            return {
                'name': 'H100 NVL (PCIe)',
                'fp32': 60,
                'fp32_tf32': 417,     # Dense TF32 (no sparsity)
                'fp16': 835,         # Dense FP16 (no sparsity)
                'bf16': 835,         # Dense BF16 (no sparsity)
                'fp8': 1670,         # Dense FP8 (no sparsity)
                'hbm_bw': 3900,       # 3.9 TB/s
                'has_sparsity': True,
            }
        else:
            # 默认 H100 SXM5
            return {
                'name': 'H100 SXM5 (默认)',
                'fp32': 67,
                'fp32_tf32': 495,
                'fp16': 989,
                'bf16': 989,
                'fp8': 1958,
                'hbm_bw': 3350,
                'has_sparsity': True,
            }

    # ====================
    # H200 系列
    # ====================
    elif 'h200' in gpu_name_lower:
        # H200 SXM - 根据官网，Tensor Core 算力与 H100 SXM 相同
        # 注意：
        # - fp32 (稠密): 67 TFLOPS (稠密 FP32 算力)
        # - fp32_tf32 (稠密 TF32): 989 TFLOPS (稠密 TF32 算力)
        # - fp16/bf16 (稠密): 1979 TFLOPS (稠密 Tensor Core 算力)
        if 'nvl' in gpu_name_lower:
            return {
                'name': 'H200 NVL',
                'fp32': 60,       # 稠密 FP32 算力
                'fp32_tf32': 835, # 稠密 TF32 算力
                'fp16': 1671,     # 稠密 FP16 算力
                'bf16': 1671,     # 稠密 BF16 算力
                'fp8': 3341,      # 稠密 FP8 算力
                'hbm_bw': 4800,   # HBM3，141GB
                'has_sparsity': True,
            }
        else:
            # H200 SXM - Dense vs Sparse compute power
            # - Dense FP32: 67 TFLOPS
            # - Dense TF32 (without sparsity): 495 TFLOPS
            # - Sparse TF32 (with sparsity): 989 TFLOPS
            # - Dense FP16/BF16 (without sparsity): 989 TFLOPS
            # - Sparse FP16/BF16 (with sparsity): 1979 TFLOPS
            return {
                'name': 'H200 SXM',
                'fp32': 67,            # Dense FP32 (dense FP32)
                'fp32_tf32': 495,       # Dense TF32 (no sparsity)
                'fp16': 989,           # Dense FP16 (no sparsity)
                'bf16': 989,           # Dense BF16 (no sparsity)
                'fp8': 1958,            # Dense FP8 (no sparsity)
                'hbm_bw': 4800,         # 4.8 TB/s HBM3, 141GB
                'has_sparsity': True,
            }

    # ====================
    # B200 系列
    # ====================
    elif 'b200' in gpu_name_lower:
        # B200 - 根据公开资料
        # 注意：稠密算力 (Dense)，不是稀疏算力
        # 如果有稀疏加速，稠密算力约为稀疏算力的一半
        return {
            'name': 'B200',
            'fp32': 67,             # Dense FP32
            'fp32_tf32': 2048,      # Dense TF32 (no sparsity)
            'fp16': 2700,           # Dense FP16
            'bf16': 2700,           # Dense BF16
            'fp8': 5400,            # Dense FP8
            'hbm_bw': 5200,         # HBM3E
            'has_sparsity': True,   # may support sparsity
        }

    # ====================
    # B300 系列
    # ====================
    elif 'b300' in gpu_name_lower:
        # B300 - 根据公开资料
        return {
            'name': 'B300',
            'fp32': 67,             # Dense FP32
            'fp32_tf32': 3072,      # Dense TF32
            'fp16': 3800,           # Dense FP16
            'bf16': 3800,           # Dense BF16
            'fp8': 7600,            # Dense FP8
            'hbm_bw': 6000,         # HBM3E
            'has_sparsity': True,   # may support sparsity
        }

    # 默认返回 H100 SXM5 参数
    else:
        return {
            'name': 'Unknown (assumed H100 SXM5)',
            'fp32': 67,
            'fp32_tf32': 495,
            'fp16': 989,
            'bf16': 989,
            'fp8': 1958,
            'hbm_bw': 3350,
            'has_sparsity': True,
        }


def detect_gpu_type(gpu_id=0):
    """检测当前 GPU 类型"""
    gpu_name = torch.cuda.get_device_name(gpu_id)
    return get_gpu_peak_info(gpu_name)


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个 GPU:")
    for i in range(num_gpus):
        info = detect_gpu_type(i)
        print(f"\n  GPU {i}: {info['name']}")
        print(f"    FP32 (稠密): {info['fp32']} TFLOPS")
        print(f"    TF32 Tensor Core: {info['fp32_tf32']} TFLOPS {'(稠密)' if info['has_sparsity'] else ''}")
        print(f"    FP16 Tensor Core: {info['fp16']} TFLOPS {'(稠密)' if info['has_sparsity'] else ''}")
        print(f"    BF16 Tensor Core: {info['bf16']} TFLOPS {'(稠密)' if info['has_sparsity'] else ''}")
        if info.get('fp8'):
            print(f"    FP8 Tensor Core: {info['fp8']} TFLOPS {'(稠密)' if info['has_sparsity'] else ''}")
        print(f"    HBM 带宽: {info['hbm_bw']} GB/s")
