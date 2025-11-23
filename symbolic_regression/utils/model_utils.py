"""
模型工具函数

提供统一的模型加载和保存功能
"""

import os
import torch
import logging
from typing import Tuple
from pathlib import Path

from ..models.expression_encoder import ExpressionEncoder
from ..models.data_encoder import DataEncoder

# 创建logger
logger = logging.getLogger(__name__)


def load_pretrained_models(
    model_dir: str,
    device: str = 'cpu',
    auto_create: bool = True
) -> Tuple[ExpressionEncoder, DataEncoder, bool]:
    """
    加载预训练模型
    
    Args:
        model_dir: 模型存储目录
        device: 设备
        auto_create: 如果模型不存在是否创建新模型
    
    Returns:
        Tuple[表达式编码器, 数据编码器, 是否成功加载了已有权重]
    """
    expr_path = os.path.join(model_dir, 'expression_encoder')
    data_path = os.path.join(model_dir, 'data_encoder')
    
    # 检查模型文件是否存在
    has_pretrained = (
        os.path.exists(expr_path) and os.path.exists(os.path.join(expr_path, 'pytorch_model.bin')) and
        os.path.exists(data_path) and os.path.exists(os.path.join(data_path, 'pytorch_model.bin'))
    )
    
    if has_pretrained:
        print(f"发现已有模型权重，从 {model_dir} 加载...")
        
        try:
            expression_encoder = ExpressionEncoder.from_pretrained(expr_path)
            data_encoder = DataEncoder.from_pretrained(data_path)
            
            expression_encoder.to(device)
            data_encoder.to(device)
            
            logger.info(f"成功加载预训练模型")
            print("模型加载完成")
            
            return expression_encoder, data_encoder, True
            
        except Exception as e:
            logger.warning(f"加载预训练模型失败: {e}")
            print(f"加载预训练模型失败: {e}")
            if not auto_create:
                raise e
            # 如果加载失败且不自动创建，则抛出异常
            if not auto_create:
                raise e
    else:
        print(f"未发现已有权重，将在 {model_dir} 创建新模型")
    
    # 创建新模型
    if not has_pretrained or auto_create:
        # 这里需要config参数来创建模型，暂时返回None让调用方处理
        return None, None, False
    
    return None, None, False


def save_models(
    expression_encoder: ExpressionEncoder,
    data_encoder: DataEncoder,
    model_dir: str,
    metadata: dict = None
):
    """
    保存模型
    
    Args:
        expression_encoder: 表达式编码器
        data_encoder: 数据编码器
        model_dir: 保存目录
        metadata: 额外元数据
    """
    os.makedirs(model_dir, exist_ok=True)
    
    expr_path = os.path.join(model_dir, 'expression_encoder')
    data_path = os.path.join(model_dir, 'data_encoder')
    
    expression_encoder.save_pretrained(expr_path)
    data_encoder.save_pretrained(data_path)
    
    # 保存元数据
    if metadata:
        import json
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"模型已保存到: {model_dir}")


def check_model_exists(model_dir: str) -> bool:
    """
    检查模型是否存在
    
    Args:
        model_dir: 模型目录
    
    Returns:
        模型是否存在
    """
    expr_path = os.path.join(model_dir, 'expression_encoder')
    data_path = os.path.join(model_dir, 'data_encoder')
    
    return (
        os.path.exists(expr_path) and os.path.exists(os.path.join(expr_path, 'pytorch_model.bin')) and
        os.path.exists(data_path) and os.path.exists(os.path.join(data_path, 'pytorch_model.bin'))
    )