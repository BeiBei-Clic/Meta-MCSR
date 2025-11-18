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
from typing import Dict, Any, Optional
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/logs/pretrain.log')
        ]
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        print("请安装pyyaml: pip install pyyaml")
        raise
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
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
            },
            'contrastive': {
                'temperature': 0.07
            }
        },
        'training': {
            'pretrain': {
                'n_expressions': 10000,  # 减少数量以便快速测试
                'n_samples_per_expr': 100,
                'variables_range': [-5, 5],
                'noise_level': 0.01,
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 10,  # 减少epochs以便快速测试
                'weight_decay': 1e-4,
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
    
    # 加载配置
    try:
        config = load_config()
        logger.info("配置加载成功")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return 1
    
    # 设置随机种子
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    logger.info(f"随机种子设置为: {config['random_seed']}")
    
    # 设置设备
    device = config['device']
    logger.info(f"使用设备: {device}")
    
    try:
        # 创建模型
        logger.info("初始化模型...")
        expression_encoder = ExpressionEncoder(
            **config['model']['expression_encoder']
        )
        data_encoder = DataEncoder(
            **config['model']['data_encoder']
        )
        
        # 创建预训练管道
        logger.info("创建预训练管道...")
        pretrain_pipeline = PretrainPipeline(
            expression_encoder=expression_encoder,
            data_encoder=data_encoder,
            config=config['training']['pretrain'],
            device=device
        )
        
        # 开始预训练
        logger.info("开始预训练...")
        training_history = pretrain_pipeline.fit(
            generate_data=True
        )
        
        # 保存最终结果
        pretrain_pipeline.save_pretrained()
        
        # 打印结果
        print("\n" + "=" * 60)
        print("预训练完成！")
        print("=" * 60)
        print(f"最终训练损失: {training_history['train_loss']['loss'][-1]:.4f}")
        print(f"最终验证损失: {training_history['val_loss']['loss'][-1]:.4f}")
        print(f"模型保存路径: {config['training']['pretrain']['output_dir']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"预训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
