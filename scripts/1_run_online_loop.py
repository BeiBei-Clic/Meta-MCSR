#!/usr/bin/env python3
"""
阶段一：基于真实解的专家微调

此脚本实现第二阶段的在线微调，使用MCTS+三元组损失
对预训练的编码器进行精细打磨。
"""

import os
import sys
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from symbolic_regression.models.expression_encoder import ExpressionEncoder
from symbolic_regression.models.data_encoder import DataEncoder
from symbolic_regression.training.finetune_loop import OnlineFinetuneLoop
from symbolic_regression.core.reward_calculator import RewardCalculator
from symbolic_regression.utils.config_utils import load_config





def load_pysr_datasets(data_dir: str = "data/pysr_datasets") -> Optional[List[Dict[str, Any]]]:
    """加载PySR格式的数据集"""
    if not os.path.exists(data_dir):
        print(f"错误：数据路径 {data_dir} 不存在")
        return None

    # 读取所有txt文件
    datasets = []
    txt_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
    
    print(f"在目录 {data_dir} 中找到 {len(txt_files)} 个数据文件")
    
    if len(txt_files) == 0:
        print("没有找到数据文件")
        return None
    
    # 处理所有文件
    for file_name in txt_files:
        file_path = os.path.join(data_dir, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            print(f"  文件 {file_name} 内容不足，跳过")
            continue
        
        # 解析第一行的表达式
        first_line = lines[0].strip()
        if first_line.startswith('表达式: '):
            expression = first_line.replace('表达式: ', '').strip()
        else:
            print(f"  文件 {file_name} 第一行格式错误: {first_line[:50]}...")
            continue
        
        # 解析数据行
        samples = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            try:
                # 解析任意维度数据格式: x1,x2,x3,...,y
                parts = line.split(',')
                if len(parts) >= 2:  # 至少要有x变量和y值
                    # 最后一个值是y，前面的都是x变量
                    x_values = [float(part) for part in parts[:-1]]
                    y_value = float(parts[-1])
                    
                    samples.append((x_values, y_value))
                else:
                    print(f"    数据格式错误: {line}")
            except (ValueError, IndexError) as e:
                print(f"    解析数据失败: {line} (错误: {e})")
                continue
        
        if expression and samples:
            datasets.append({
                'expression': expression,
                'samples': samples
            })
            # print(f"  加载表达式: {expression[:50]}... 样本数: {len(samples)}")
        else:
            print(f"  文件 {file_name} 无有效数据")
    
    print(f"成功加载 {len(datasets)} 个表达式数据\n")
    return datasets
    
def load_pretrained_models(
    pretrained_dir: str,
    device: str = 'cpu'
) -> Tuple[ExpressionEncoder, DataEncoder]:
    """加载预训练模型"""
    expr_path = os.path.join(pretrained_dir, 'expression_encoder')
    data_path = os.path.join(pretrained_dir, 'data_encoder')

    print(f"加载预训练模型从: {pretrained_dir}")

    expression_encoder = ExpressionEncoder.from_pretrained(expr_path)
    data_encoder = DataEncoder.from_pretrained(data_path)

    expression_encoder.to(device)
    data_encoder.to(device)

    print("预训练模型加载完成")

    return expression_encoder, data_encoder


def main():
    """主函数"""
    # 加载配置
    config = load_config()

    # 设置设备
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}\n")

    # 创建输出目录
    os.makedirs('models_weights/finetuned', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/logs/online_finetune.log')
        ]
    )

    # 检查是否有预训练模型
    pretrained_dir = config['training']['pretrain']['output_dir']

    if not os.path.exists(os.path.join(pretrained_dir, 'expression_encoder')):
        print(f"错误: 找不到预训练模型，请先运行预训练脚本")
        print(f"  运行: python scripts/0_run_pretraining.py")
        sys.exit(1)

    # 加载预训练模型
    expression_encoder, data_encoder = load_pretrained_models(pretrained_dir, device)

    # 加载PySR数据集
    datasets = load_pysr_datasets("data/pysr_datasets")
    
    if datasets is None:
        print("错误：无法加载数据集，程序退出")
        sys.exit(1)
    
    # 转换数据格式以适配现有逻辑
    benchmark_tasks = []
    for dataset in datasets:
        expression = dataset['expression']
        samples = dataset['samples']
        
        # 转换为numpy数组格式
        X = np.array([sample[0] for sample in samples])
        y = np.array([sample[1] for sample in samples])
        
        benchmark_tasks.append((expression, X, y))
    
    print(f"加载了 {len(benchmark_tasks)} 个任务用于在线微调")

    # 创建在线微调循环
    finetune_loop = OnlineFinetuneLoop(
        expression_encoder=expression_encoder,
        data_encoder=data_encoder,
        config=config['training']['online_finetune'],
        device=device
    )

    # 执行在线微调
    print("=" * 60)
    print("开始在线微调阶段")
    print("=" * 60)

    train_history = finetune_loop.finetune(
        benchmark_tasks=benchmark_tasks,
        mcts_epochs=config['training']['online_finetune'].get('mcts_epochs', 50),
        eval_steps=config['training']['online_finetune'].get('eval_steps', 10),
        save_steps=config['training']['online_finetune'].get('save_steps', 10),
        output_dir=config['training']['online_finetune']['output_dir'],
        verbose=True
    )

    # 保存最终模型
    final_model_path = config['training']['online_finetune']['output_dir']
    finetune_loop.save_pretrained(final_model_path)

    # 打印最终统计信息
    stats = finetune_loop.get_statistics()
    print("\n" + "=" * 60)
    print("在线微调完成")
    print(f"模型已保存到: {final_model_path}")

    # 保存训练历史
    import json
    history_file = os.path.join(final_model_path, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump({
            'train_history': train_history,
            'final_statistics': stats
        }, f, indent=2)

    print(f"训练历史已保存到: {history_file}\n")


if __name__ == "__main__":
    main()
