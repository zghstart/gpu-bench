#!/usr/bin/env python3
# 脚本：generate_report.py
# 用途：生成标准化的GPU测试报告
# 支持：收集所有测试结果，生成HTML和JSON格式的报告

import os
import sys
import json
import datetime
import subprocess
import re

# 创建日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 日志记录函数
def log(message):
    """记录日志"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(os.path.join(LOG_DIR, f'report_generation_{datetime.datetime.now().strftime("%Y%m%d")}.log'), 'a') as f:
        f.write(log_message + '\n')

# 添加scripts目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from gpu_config import detect_gpu_type

def run_test(command, cwd=None):
    """运行测试命令并返回输出"""
    log(f"执行命令: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd
        )
        log(f"命令执行完成，返回码: {result.returncode}")
        if result.stderr:
            log(f"命令错误输出: {result.stderr[:200]}...")
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except Exception as e:
        error_message = str(e)
        log(f"命令执行异常: {error_message}")
        return {
            'stdout': '',
            'stderr': error_message,
            'returncode': 1
        }

def collect_env_info():
    """收集环境信息"""
    log("收集环境信息...")
    result = run_test(f"bash {script_dir}/01_env_check.sh")
    log("环境信息收集完成")
    return result

def collect_gemm_info():
    """收集GEMM测试信息"""
    log("收集GEMM测试信息...")
    single_result = run_test(f"python3 {script_dir}/02a_gemm_single_gpu.py")
    multi_result = run_test(f"python3 {script_dir}/02b_gemm_multi_gpu.py")
    log("GEMM测试信息收集完成")
    return {
        'single': single_result,
        'multi': multi_result
    }

def collect_memory_bandwidth():
    """收集内存带宽测试信息"""
    log("收集内存带宽测试信息...")
    result = run_test(f"python3 {script_dir}/03_memory_bandwidth.py")
    log("内存带宽测试信息收集完成")
    return result

def collect_disk_io():
    """收集磁盘I/O测试信息"""
    log("收集磁盘I/O测试信息...")
    result = run_test(f"bash {script_dir}/04_disk_io.sh")
    log("磁盘I/O测试信息收集完成")
    return result

def collect_gpu_topology():
    """收集GPU拓扑测试信息"""
    log("收集GPU拓扑测试信息...")
    result = run_test(f"bash {script_dir}/05_gpu_topology.sh")
    log("GPU拓扑测试信息收集完成")
    return result

def collect_nccl_info():
    """收集NCCL测试信息"""
    log("收集NCCL测试信息...")
    num_gpus = subprocess.run(
        "nvidia-smi --query-gpu=count --format=csv,noheader | head -1",
        shell=True,
        capture_output=True,
        text=True
    ).stdout.strip()
    result = run_test(f"torchrun --nproc_per_node={num_gpus} {script_dir}/06b_nccl_pytorch.py")
    log("NCCL测试信息收集完成")
    return result

def collect_network_info():
    """收集网络性能测试信息"""
    log("收集网络性能测试信息...")
    result = run_test(f"bash {script_dir}/07_network_performance.sh")
    log("网络性能测试信息收集完成")
    return result

def collect_stress_test():
    """收集压力测试信息"""
    log("收集压力测试信息...")
    result = run_test(f"python3 {script_dir}/07b_stress_test_pytorch.py")
    log("压力测试信息收集完成")
    return result

def collect_inference_info():
    """收集推理吞吐量测试信息"""
    log("收集推理吞吐量测试信息...")
    result = run_test(f"python3 {script_dir}/09_inference_throughput.py")
    log("推理吞吐量测试信息收集完成")
    return result

def parse_gpu_info():
    """解析GPU信息"""
    log("解析GPU信息...")
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        log(f"检测到 {num_gpus} 个GPU")
        gpus = []
        for i in range(num_gpus):
            info = detect_gpu_type(i)
            gpu_name = torch.cuda.get_device_name(i)
            gpus.append({
                'id': i,
                'name': gpu_name,
                'info': info
            })
            log(f"GPU {i}: {gpu_name}")
        log("GPU信息解析完成")
        return gpus
    except Exception as e:
        error_message = str(e)
        log(f"解析GPU信息时出错: {error_message}")
        return []


def load_history_reports():
    """加载历史测试报告"""
    log("加载历史测试报告...")
    history_reports = []
    for file in os.listdir('.'):
        if file.endswith('.json') and 'gpu_benchmark_report' in file:
            try:
                with open(file, 'r') as f:
                    report = json.load(f)
                    history_reports.append(report)
                log(f"加载历史报告: {file}")
            except Exception as e:
                error_message = str(e)
                log(f"加载历史报告 {file} 时出错: {error_message}")
                pass
    # 按时间戳排序，最新的在前
    history_reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    log(f"共加载 {len(history_reports)} 个历史测试报告")
    return history_reports


def compare_with_theoretical(gpu_info, test_name, test_value):
    """与理论值对比"""
    if not gpu_info:
        return None
    
    # 根据测试名称获取理论值
    if test_name == 'memory_bandwidth':
        return gpu_info.get('hbm_bw', 0)
    elif test_name == 'fp16_performance':
        return gpu_info.get('fp16', 0) * 1000  # 转换为Gflop/s
    elif test_name == 'fp32_performance':
        return gpu_info.get('fp32', 0) * 1000  # 转换为Gflop/s
    else:
        return None

def generate_json_report(data):
    """生成JSON格式的报告"""
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'hostname': subprocess.run('hostname', shell=True, capture_output=True, text=True).stdout.strip(),
        'gpu_info': parse_gpu_info(),
        'tests': data
    }
    
    report_file = f"gpu_benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_file

def generate_html_report(data):
    """生成HTML格式的报告"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = subprocess.run('hostname', shell=True, capture_output=True, text=True).stdout.strip()
    gpus = parse_gpu_info()
    history_reports = load_history_reports()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPU 服务器性能测试报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
            .warning {{ background-color: #fff3cd; border-color: #ffeeba; }}
            .error {{ background-color: #f8d7da; border-color: #f5c6cb; }}
            pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>GPU 服务器性能测试报告</h1>
        <p>测试时间: {timestamp}</p>
        <p>测试机器: {hostname}</p>
        
        <h2>GPU 信息</h2>
        <div class="section">
            <table>
                <tr>
                    <th>GPU ID</th>
                    <th>型号</th>
                    <th>理论FP16性能 (TFLOPS)</th>
                    <th>理论HBM带宽 (GB/s)</th>
                </tr>
        """
    
    for gpu in gpus:
        html += f"""
                <tr>
                    <td>{gpu['id']}</td>
                    <td>{gpu['name']}</td>
                    <td>{gpu['info']['fp16']}</td>
                    <td>{gpu['info']['hbm_bw']}</td>
                </tr>
        """
    
    html += f"""
            </table>
        </div>
        
        <h2>测试结果对比</h2>
        <div class="section">
            <h3>与历史测试对比</h3>
            {f'<p>找到 {len(history_reports)} 个历史测试报告</p>' if history_reports else '<p>无历史测试报告</p>'}
            
            {f"""
            <h4>最近的历史测试</h4>
            <table>
                <tr>
                    <th>测试时间</th>
                    <th>测试机器</th>
                    <th>GPU数量</th>
                </tr>
                {''.join([f"<tr><td>{report.get('timestamp', '')}</td><td>{report.get('hostname', '')}</td><td>{len(report.get('gpu_info', []))}</td></tr>" for report in history_reports[:5]])}
            </table>
            """ if history_reports else ''}
        </div>
    """
    
    test_sections = [
        ('环境信息', 'env', data.get('env', {})),
        ('GEMM 矩阵乘法', 'gemm', data.get('gemm', {})),
        ('内存带宽', 'memory', data.get('memory', {})),
        ('磁盘 I/O', 'disk', data.get('disk', {})),
        ('GPU 拓扑', 'topology', data.get('topology', {})),
        ('NCCL 通信', 'nccl', data.get('nccl', {})),
        ('网络性能', 'network', data.get('network', {})),
        ('压力测试', 'stress', data.get('stress', {})),
        ('推理吞吐量', 'inference', data.get('inference', {}))
    ]
    
    for section_name, section_key, section_data in test_sections:
        html += f"""
        <h2>{section_name}</h2>
        <div class="section">
        """
        
        if isinstance(section_data, dict) and 'stdout' in section_data:
            # 单个测试结果
            status = 'success' if section_data['returncode'] == 0 else 'error'
            html += f"""
            <div class="{status}">
                <h3>测试结果</h3>
                <pre>{section_data['stdout']}</pre>
                {f'<pre style="color: red;">{section_data["stderr"]}</pre>' if section_data['stderr'] else ''}
            </div>
            """
        elif isinstance(section_data, dict):
            # 多个测试结果（如GEMM）
            for sub_test, sub_data in section_data.items():
                status = 'success' if sub_data['returncode'] == 0 else 'error'
                html += f"""
                <h3>{sub_test} 测试</h3>
                <div class="{status}">
                    <pre>{sub_data['stdout']}</pre>
                    {f'<pre style="color: red;">{sub_data["stderr"]}</pre>' if sub_data['stderr'] else ''}
                </div>
                """
        
        html += f"""
        </div>
        """
    
    html += f"""
    </body>
    </html>
    """
    
    report_file = f"gpu_benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_file, 'w') as f:
        f.write(html)
    
    return report_file

def main():
    """主函数"""
    log("开始收集测试数据...")
    
    try:
        data = {
            'env': collect_env_info(),
            'gemm': collect_gemm_info(),
            'memory': collect_memory_bandwidth(),
            'disk': collect_disk_io(),
            'topology': collect_gpu_topology(),
            'nccl': collect_nccl_info(),
            'network': collect_network_info(),
            'stress': collect_stress_test(),
            'inference': collect_inference_info()
        }
        
        log("生成报告...")
        json_report = generate_json_report(data)
        html_report = generate_html_report(data)
        
        log(f"报告生成完成:")
        log(f"  JSON 报告: {json_report}")
        log(f"  HTML 报告: {html_report}")
    except Exception as e:
        error_message = str(e)
        log(f"生成报告时出错: {error_message}")
        import traceback
        log(f"错误堆栈: {traceback.format_exc()}")

if __name__ == '__main__':
    main()
