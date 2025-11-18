"""
基于对比学习预训练与真实解引导微调的MCTS集成框架

这个包实现了：
1. 阶段零：基于对比学习的预训练
2. 在线主循环：MCTS探索与真实解引导微调
"""

__version__ = "0.1.0"
__author__ = "iFlow CLI"

# 导入核心类
from .core.mcts_engine import EnhancedMCTSEngine
from .core.reward_calculator import RewardCalculator
from .models.expression_encoder import ExpressionEncoder
from .models.data_encoder import DataEncoder
from .training.pretrain_pipeline import PretrainPipeline
from .training.finetune_loop import FinetuneLoop

__all__ = [
    "EnhancedMCTSEngine",
    "RewardCalculator", 
    "ExpressionEncoder",
    "DataEncoder",
    "PretrainPipeline",
    "FinetuneLoop"
]