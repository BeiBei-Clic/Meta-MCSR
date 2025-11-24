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
from symbolic_regression.utils.data_loader import load_pysr_data
from symbolic_regression.utils.model_utils import load_pretrained_models, save_models, check_model_exists
from symbolic_regression.utils.logging_utils import setup_logger

def main():
    """主函数"""
    # 加载配置
    config = load_config()

    # 设置设备
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}\n")

    # 设置日志
    setup_logger('online_finetune.log')

    # 检查是否有预训练模型
    model_dir = config['training']['model_dir']

    if not check_model_exists(model_dir):
        print(f"错误: 找不到预训练模型，请先运行预训练脚本")
        print(f"  运行: python scripts/0_run_pretraining.py")
        sys.exit(1)

    # 加载预训练模型
    expression_encoder, data_encoder, _ = load_pretrained_models(model_dir, device)

    # 加载PySR数据集
    datasets = load_pysr_data("data/pysr_datasets")

    if datasets is None:
        print("错误：无法加载数据集，程序退出")
        sys.exit(1)

    # 转换数据格式以适配现有逻辑
    benchmark_tasks = [(d['expression'], np.array([s[0] for s in d['samples']]), np.array([s[1] for s in d['samples']])) for d in datasets]

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
        output_dir=model_dir,  # 使用统一的模型目录
        verbose=True
    )

    # 保存最终模型（使用统一的save_models函数）
    save_models(expression_encoder, data_encoder, model_dir)

    # 打印最终统计信息
    stats = finetune_loop.get_statistics()
    print("\n" + "=" * 60)
    print("在线微调完成")
    print(f"模型已保存到: {model_dir}")

    # 保存训练历史
    import json
    history_file = os.path.join(model_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump({'train_history': train_history, 'final_statistics': stats}, f, indent=2)

    print(f"训练历史已保存到: {history_file}\n")


if __name__ == "__main__":
    main()
