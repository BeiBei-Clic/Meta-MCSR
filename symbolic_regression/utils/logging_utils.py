"""
统一的日志设置工具模块
"""

import os
import logging


def setup_logger(log_file: str = 'training.log') -> logging.Logger:
    """
    设置统一的日志配置

    Args:
        log_file: 日志文件名，将保存在 results/logs/ 目录下

    Returns:
        配置好的logger对象
    """
    # 确保日志目录存在
    os.makedirs('results/logs', exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'results/logs/{log_file}')
        ]
    )

    # 返回当前模块的logger
    return logging.getLogger(__name__)
