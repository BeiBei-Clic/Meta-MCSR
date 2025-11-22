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
from symbolic_regression.utils.data_loader import DataLoader





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


def load_pretrain_data(pretrain_data_path: str = "data/pysr_datasets"):
    """加载PySR格式的预训练数据"""
    if not os.path.exists(pretrain_data_path):
        print(f"预训练数据路径 {pretrain_data_path} 不存在，开始自动生成数据...")
        pysr_generate_script_path = os.path.join(project_root, 'scripts', 'generate_pretrain_data_PySR.py')
        result = subprocess.run([sys.executable, pysr_generate_script_path], 
                               capture_output=True, text=True, cwd=str(project_root))
        if result.returncode != 0:
            print(f"PySR数据生成失败: {result.stderr}")
            return None
    
    # 目录处理
    if os.path.isdir(pretrain_data_path):
        pretrain_data = []
        txt_files = [f for f in os.listdir(pretrain_data_path) if f.endswith('.txt')]
        
        for file_name in txt_files:
            file_path = os.path.join(pretrain_data_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                continue
            
            current_expression = lines[0].replace('表达式: ', '').strip()
            samples = []
            
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 2:
                    x_values = [float(part) for part in parts[:-1]]
                    y_value = float(parts[-1])
                    
                    sample = {'y': y_value}
                    for i, x_val in enumerate(x_values, 1):
                        sample[f'x{i}'] = x_val
                    samples.append(sample)
            
            if samples:
                variables = [key for key in sorted(samples[0].keys()) if key.startswith('x')]
                pretrain_data.append({
                    'expression': current_expression,
                    'samples': samples,
                    'variables': variables
                })
        
        return pretrain_data if pretrain_data else None
    
    # 单文件处理
    else:
        with open(pretrain_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
        
        expression = lines[0].replace('表达式: ', '').strip()
        samples = []
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                x_values = [float(part) for part in parts[:-1]]
                y_value = float(parts[-1])
                
                sample = {'y': y_value}
                for i, x_val in enumerate(x_values, 1):
                    sample[f'x{i}'] = x_val
                samples.append(sample)
        
        if samples:
            variables = [f'x{i}' for i in range(1, len(samples[0]))]
            return [{'expression': expression, 'samples': samples, 'variables': variables}]
        
        return None

def convert_pysr_data_to_pretrain_format(pysr_data: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """将PySR格式数据转换为预训练格式"""
    expressions = []
    datasets = []
    
    for item in pysr_data:
        samples = item['samples']
        if len(samples) == 0:
            continue
            
        # 简单处理x变量
        x_variables = [f'x{i}' for i in range(1, len(samples[0]))]
        X = np.array([[sample[var] for var in x_variables] for sample in samples]).astype(np.float32)
        y = np.array([sample['y'] for sample in samples]).astype(np.float32)
        
        expressions.append(item['expression'])
        datasets.append((X, y))
    
    return expressions, datasets


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        # 返回默认配置
        return {
            'model': {
                'expression_encoder': {
                    'embedding_dim': 512,
                    'n_heads': 8,
                    'n_layers': 6,
                    'dropout': 0.1,
                    'max_seq_length': 128
                },
                'data_encoder': {
                    'embedding_dim': 512,
                    'n_heads': 8,
                    'n_layers': 6,
                    'dropout': 0.1,
                    'max_features': 100
                }
            },
            'training': {
                'pretrain': {
                    'n_expressions': 10000,
                    'n_samples_per_expr': 100,
                    'variables_range': [-5, 5],
                    'noise_level': 0.01,
                    'batch_size': 32,
                    'learning_rate': 0.0001,
                    'num_epochs': 10,
                    'weight_decay': 0.0001,
                    'warmup_steps': 1000,
                    'output_dir': 'models_weights/pretrained/'
                }
            },
            'data': {
                'pretrain': {
                    'output_path': 'data/pretrain/'
                }
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'random_seed': 42
        }


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
    
    # 创建模型和预训练管道
    expression_encoder = ExpressionEncoder(**config['model']['expression_encoder'])
    data_encoder = DataEncoder(**config['model']['data_encoder'])
    pretrain_pipeline = PretrainPipeline(
        expression_encoder=expression_encoder,
        data_encoder=data_encoder,
        config=config['training']['pretrain'],
        device=device
    )
    
    # 加载和处理数据
    pysr_data = load_pretrain_data("data/pysr_datasets")
    if not pysr_data:
        print("无法加载预训练数据")
        return 1
    
    expressions, datasets = convert_pysr_data_to_pretrain_format(pysr_data)
    
    # 开始预训练
    training_history = pretrain_pipeline.fit(expressions=expressions, datasets=datasets)
    pretrain_pipeline.save_pretrained()

    # 打印结果
    print("\n" + "=" * 60)
    print("预训练完成！")
    print("=" * 60)
    print(f"最终训练损失: {training_history['train_loss']['loss'][-1]:.4f}")
    print(f"最终验证损失: {training_history['val_loss']['loss'][-1]:.4f}")
    print(f"模型保存路径: {config['training']['pretrain']['output_dir']}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
