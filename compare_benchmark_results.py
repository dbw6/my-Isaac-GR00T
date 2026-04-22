#!/usr/bin/env python3
"""
对比两个benchmark文件夹的结果并绘制柱状图
"""
import os
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def extract_task_name_from_final_results(line):
    """从final_results.txt中提取任务名称"""
    # 格式: | `robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env` | 34.00% |
    match = re.search(r'`([^`]+)`', line)
    if match:
        full_name = match.group(1)
        # 提取任务名称部分（去掉前缀）
        task_name = full_name.split('/')[-1] if '/' in full_name else full_name
        return task_name
    return None

def extract_success_rate_from_final_results(line):
    """从final_results.txt中提取成功率"""
    # 格式: | `robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env` | 34.00% |
    match = re.search(r'\|\s*([\d.]+)%\s*\|', line)
    if match:
        return float(match.group(1))
    return None

def extract_success_rate_from_task_file(file_path):
    """从任务文件中提取success rate"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 查找 "success rate: 0.31" 这样的行
            match = re.search(r'success rate:\s*([\d.]+)', content)
            if match:
                return float(match.group(1)) * 100  # 转换为百分比
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def get_task_results_from_folder1(folder_path):
    """从第一个文件夹（有final_results.txt）读取结果"""
    results = {}
    final_results_path = os.path.join(folder_path, 'final_results.txt')
    
    if not os.path.exists(final_results_path):
        print(f"Warning: {final_results_path} not found")
        return results
    
    with open(final_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line and '`' in line and '%' in line:
                task_name = extract_task_name_from_final_results(line)
                success_rate = extract_success_rate_from_final_results(line)
                if task_name and success_rate is not None:
                    results[task_name] = success_rate
    
    return results

def get_task_results_from_folder2(folder_path, task_names):
    """从第二个文件夹（没有final_results.txt）读取结果"""
    results = {}
    incomplete_tasks = []
    
    for task_name in task_names:
        # 构建任务文件名
        task_file = f"{task_name}.txt"
        task_file_path = os.path.join(folder_path, task_file)
        
        if os.path.exists(task_file_path):
            success_rate = extract_success_rate_from_task_file(task_file_path)
            if success_rate is not None:
                results[task_name] = success_rate
            else:
                # 文件存在但无法提取结果，视为未完成
                results[task_name] = 0.0
                incomplete_tasks.append(task_name)
        else:
            # 文件不存在，视为未完成
            results[task_name] = 0.0
            incomplete_tasks.append(task_name)
    
    return results, incomplete_tasks

def plot_comparison(folder1_results, folder2_results, folder1_name, folder2_name):
    """绘制对比柱状图"""
    # 获取所有任务名称，按第一个文件夹的顺序
    task_names = list(folder1_results.keys())
    
    # 计算平均成功率
    avg1 = np.mean(list(folder1_results.values()))
    avg2 = np.mean(list(folder2_results.values()))
    
    # 准备数据（包括平均值）
    rates1 = [folder1_results.get(task, 0.0) for task in task_names] + [avg1]
    rates2 = [folder2_results.get(task, 0.0) for task in task_names] + [avg2]
    
    # 简化任务名称用于显示（去掉_PandaOmron_Env后缀）
    display_names = [name.replace('_PandaOmron_Env', '') for name in task_names] + ['Average']
    
    # 创建图形
    x = np.arange(len(task_names) + 1)  # +1 for Average
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, rates1, width, label=folder1_name, alpha=0.8)
    bars2 = ax.bar(x + width/2, rates2, width, label=folder2_name, alpha=0.8)
    
    # 在Average柱子前添加分隔线
    ax.axvline(x=len(task_names) - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 设置标签和标题
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Benchmark Results Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
    # Average标签加粗
    ax.get_xticklabels()[-1].set_fontweight('bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    # 在柱子上添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=7)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = 'benchmark_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 显示图片
    plt.show()

def main():
    # 文件夹路径
    folder1 = '/home/bd199/my-Isaac-GR00T/robocasa_benchmark_20260107_162517'
    folder2 = '/home/bd199/my-Isaac-GR00T/robocasa_benchmark_20260121_123908'
    
    folder1_name = 'Original Model'
    folder2_name = 'Apply Focus'
    
    print("正在读取第一个文件夹的结果...")
    folder1_results = get_task_results_from_folder1(folder1)
    print(f"从第一个文件夹读取到 {len(folder1_results)} 个任务的结果")
    
    print("\n正在读取第二个文件夹的结果...")
    folder2_results, incomplete_tasks = get_task_results_from_folder2(folder2, list(folder1_results.keys()))
    print(f"从第二个文件夹读取到 {len(folder2_results)} 个任务的结果")
    
    # 统计未完成的任务（文件不存在或无法提取结果）
    if incomplete_tasks:
        print(f"\n第二个文件夹中未完成的任务 ({len(incomplete_tasks)} 个):")
        for task in incomplete_tasks:
            print(f"  - {task}")
    
    print("\n正在生成对比图表...")
    plot_comparison(folder1_results, folder2_results, folder1_name, folder2_name)
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"{folder1_name} 平均成功率: {np.mean(list(folder1_results.values())):.2f}%")
    print(f"{folder2_name} 平均成功率: {np.mean(list(folder2_results.values())):.2f}%")
    
    # 计算改进情况
    improvements = []
    for task in folder1_results.keys():
        rate1 = folder1_results[task]
        rate2 = folder2_results.get(task, 0.0)
        if rate2 > rate1:
            improvements.append((task, rate2 - rate1))
    
    if improvements:
        print(f"\n{folder2_name} 改进的任务 ({len(improvements)} 个):")
        for task, improvement in sorted(improvements, key=lambda x: x[1], reverse=True):
            print(f"  - {task.replace('_PandaOmron_Env', '')}: +{improvement:.2f}%")

if __name__ == '__main__':
    main()
