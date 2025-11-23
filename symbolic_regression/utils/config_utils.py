"""
配置工具模块

提供统一的配置加载功能
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为"config.yaml"
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 当配置文件不存在时
        yaml.YAMLError: 当配置文件格式错误时
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)