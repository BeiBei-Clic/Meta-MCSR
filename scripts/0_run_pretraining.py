#!/usr/bin/env python3
"""
阶段零：基于对比学习的预训练

此脚本实现第一阶段的预训练，使用对称性InfoNCE损失
对表达式编码器和数据编码器进行联合训练。
"""

import os
import sys
import torch
import numpy as np
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from symbolic_regression.models.expression_encoder import ExpressionEncoder
from symbolic_regression.models.data_encoder import DataEncoder
from symbolic_regression.training.pretrain_pipeline import PretrainPipeline
from symbolic_regression.utils.data_loader import DataLoader, load_pysr_data
from symbolic_regression.utils.config_utils import load_config
from symbolic_regression.utils.model_utils import load_pretrained_models, save_models, check_model_exists





def setup_logging():
    """设置日志"""
    # 确保日志目录存在
    os.makedirs('results/logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/logs/pretrain.log')
        ]
    )




def convert_pysr_data_to_pretrain_format(pysr_data: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """将PySR格式数据转换为预训练格式"""
    expressions = []
    datasets = []
    
    for item in pysr_data:
        samples = item['samples']
        if len(samples) == 0:
            continue
            
        # samples是(x_values, y_value)的元组列表
        X = np.array([sample[0] for sample in samples]).astype(np.float32)
        y = np.array([sample[1] for sample in samples]).astype(np.float32)
        
        expressions.append(item['expression'])
        datasets.append((X, y))
    
    return expressions, datasets





def main():
    """主函数"""
    print("=" * 60)
    print("基于对比学习的预训练 - 阶段零")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 加载配置和设置环境
    config = load_config()
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    device = config['device']
    
    # 模型目录
    model_dir = config['training']['model_dir']
    
    # 检查和加载预训练模型
    expression_encoder, data_encoder, loaded_existing = load_pretrained_models(
        model_dir, device, auto_create=True
    )
    
    # 如果没有加载到现有模型，创建新模型
    if expression_encoder is None or data_encoder is None:
        print("创建新模型")
        expression_encoder = ExpressionEncoder(**config['model']['expression_encoder'])
        data_encoder = DataEncoder(**config['model']['data_encoder'])
        expression_encoder.to(device)
        data_encoder.to(device)
    
    pretrain_pipeline = PretrainPipeline(
        expression_encoder=expression_encoder,
        data_encoder=data_encoder,
        config=config['training']['pretrain'],
        device=device
    )
    
    # 加载和处理数据
    pysr_data = load_pysr_data("data/pysr_datasets", auto_generate=True)
    if not pysr_data:
        print("无法加载预训练数据")
        return 1
    
    expressions, datasets = convert_pysr_data_to_pretrain_format(pysr_data)
    
    # 开始预训练
    training_history = pretrain_pipeline.fit(expressions=expressions, datasets=datasets)
    
    # 保存模型
    save_models(expression_encoder, data_encoder, model_dir)

    # 打印结果
    print("\n" + "=" * 60)
    print("预训练完成！")
    print("=" * 60)
    print(f"最终训练损失: {training_history['train_loss']['loss'][-1]:.4f}")
    print(f"最终验证损失: {training_history['val_loss']['loss'][-1]:.4f}")
    print(f"模型保存路径: {model_dir}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
