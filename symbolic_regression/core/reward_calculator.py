"""
复合奖励计算器

实现结构引导、数据对齐和真实精度的三维协同评估。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math


class RewardCalculator:
    """复合奖励计算器，支持多种奖励机制"""
    
    def __init__(
        self,
        reward_weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        epsilon: float = 1e-8
    ):
        """
        初始化奖励计算器
        
        Args:
            reward_weights: 奖励权重字典
            temperature: 温度参数，用于调整奖励分布
            epsilon: 数值稳定性参数
        """
        self.reward_weights = reward_weights or {
            'structure_alignment': 0.3,
            'data_alignment': 0.4,
            'accuracy': 0.3
        }
        
        self.temperature = temperature
        self.epsilon = epsilon
        
        # 奖励归一化参数
        self.reward_history = {
            'structure_alignment': [],
            'data_alignment': [],
            'accuracy': []
        }
        self.max_history = 1000
    
    def calculate_composite_reward(
        self,
        expr_embedding: np.ndarray,
        data_embedding: np.ndarray,
        target_expr_embedding: Optional[np.ndarray] = None,
        predicted_values: Optional[np.ndarray] = None,
        true_values: Optional[np.ndarray] = None,
        r2_score: float = -np.inf,
        complexity: float = 0.0,
        expression_str: Optional[str] = None
    ) -> Dict[str, float]:
        """
        计算复合奖励
        
        Args:
            expr_embedding: 表达式嵌入向量
            data_embedding: 数据嵌入向量
            target_expr_embedding: 目标表达式嵌入向量（真实解）
            predicted_values: 预测值
            true_values: 真实值
            r2_score: R2分数
            complexity: 表达式复杂度
            expression_str: 表达式字符串
            
        Returns:
            包含各部分奖励和总奖励的字典
        """
        rewards = {}
        
        # 1. 数据对齐奖励
        rewards['data_alignment'] = self._calculate_data_alignment_reward(
            expr_embedding, data_embedding
        )
        
        # 2. 结构对齐奖励
        rewards['structure_alignment'] = self._calculate_structure_alignment_reward(
            expr_embedding, target_expr_embedding
        )
        
        # 3. 准确度奖励
        rewards['accuracy'] = self._calculate_accuracy_reward(r2_score)
        
        # 4. 复杂度奖励（可选，用于惩罚过复杂的表达式）
        rewards['complexity'] = self._calculate_complexity_penalty(complexity)
        
        # 5. 稳定性奖励（可选，用于奖励表达式的数值稳定性）
        rewards['stability'] = self._calculate_stability_reward(
            predicted_values, true_values
        )
        
        # 计算加权总奖励
        total_reward = 0.0
        for component, weight in self.reward_weights.items():
            if component in rewards:
                total_reward += weight * rewards[component]
        
        rewards['total'] = total_reward
        
        # 更新历史记录
        self._update_reward_history(rewards)
        
        return rewards
    
    def _calculate_data_alignment_reward(
        self,
        expr_embedding: np.ndarray,
        data_embedding: np.ndarray
    ) -> float:
        """计算数据对齐奖励"""
        try:
            # 确保输入是张量格式并移动到CPU
            if isinstance(expr_embedding, torch.Tensor):
                expr_embedding = expr_embedding.cpu().numpy()
            if isinstance(data_embedding, torch.Tensor):
                data_embedding = data_embedding.cpu().numpy()

            # 计算余弦相似度
            expr_tensor = torch.FloatTensor(expr_embedding).unsqueeze(0)
            data_tensor = torch.FloatTensor(data_embedding).unsqueeze(0)

            # 余弦相似度
            cosine_sim = F.cosine_similarity(expr_tensor, data_tensor).item()

            # 转换为奖励（0-1范围）
            alignment_reward = max(0, (cosine_sim + 1) / 2)

            # 应用温度缩放
            alignment_reward = alignment_reward / self.temperature

            return alignment_reward

        except Exception:
            return 0.0
    
    def _calculate_structure_alignment_reward(
        self,
        expr_embedding: np.ndarray,
        target_expr_embedding: Optional[np.ndarray] = None
    ) -> float:
        """计算结构对齐奖励"""
        if target_expr_embedding is None:
            return 0.0

        try:
            # 确保输入是张量格式并移动到CPU
            if isinstance(expr_embedding, torch.Tensor):
                expr_embedding = expr_embedding.cpu().numpy()
            if isinstance(target_expr_embedding, torch.Tensor):
                target_expr_embedding = target_expr_embedding.cpu().numpy()

            # 计算与真实解的余弦相似度
            expr_tensor = torch.FloatTensor(expr_embedding).unsqueeze(0)
            target_tensor = torch.FloatTensor(target_expr_embedding).unsqueeze(0)

            cosine_sim = F.cosine_similarity(expr_tensor, target_tensor).item()

            # 转换为奖励（0-1范围）
            structure_reward = max(0, (cosine_sim + 1) / 2)

            return structure_reward

        except Exception:
            return 0.0
    
    def _calculate_accuracy_reward(self, r2_score: float) -> float:
        """计算准确度奖励"""
        # R2分数范围通常是(-inf, 1]，我们将负值和0值映射为0奖励
        if r2_score <= 0:
            return 0.0
        
        # 使用sigmoid函数平滑地映射到(0, 1)范围
        # r2_score=0 -> 接近0，r2_score=1 -> 接近1
        accuracy_reward = 1 / (1 + math.exp(-5 * (r2_score - 0.5)))
        
        return accuracy_reward
    
    def _calculate_complexity_penalty(self, complexity: float) -> float:
        """计算复杂度惩罚奖励（负值）"""
        # 使用指数衰减惩罚复杂度
        # 复杂度越高，惩罚越大
        max_complexity = 1000  # 假设最大复杂度为1000
        complexity_penalty = -0.1 * math.exp(complexity / max_complexity)
        
        return complexity_penalty
    
    def _calculate_stability_reward(
        self, 
        predicted_values: Optional[np.ndarray] = None, 
        true_values: Optional[np.ndarray] = None
    ) -> float:
        """计算数值稳定性奖励"""
        if predicted_values is None or true_values is None:
            return 0.0
            
        try:
            # 计算预测值的方差
            pred_var = np.var(predicted_values)
            true_var = np.var(true_values)
            
            # 计算稳定性指标（预测值方差与真实值方差的比值）
            stability_ratio = pred_var / (true_var + self.epsilon)
            
            # 计算稳定性奖励（1表示稳定，>1表示不稳定）
            if stability_ratio <= 1.5:  # 允许一些波动
                stability_reward = 0.1
            elif stability_ratio <= 3.0:
                stability_reward = 0.05
            else:
                stability_reward = 0.0
            
            return stability_reward
            
        except Exception:
            return 0.0
    
    def _update_reward_history(self, rewards: Dict[str, float]):
        """更新奖励历史记录"""
        for component, value in rewards.items():
            if component != 'total':
                self.reward_history[component].append(value)
                
                # 保持历史记录在限制范围内
                if len(self.reward_history[component]) > self.max_history:
                    self.reward_history[component].pop(0)
    
    def get_normalized_rewards(
        self, 
        rewards: Dict[str, float], 
        method: str = 'min_max'
    ) -> Dict[str, float]:
        """
        归一化奖励值
        
        Args:
            rewards: 原始奖励字典
            method: 归一化方法 ('min_max', 'z_score', 'percentile')
            
        Returns:
            归一化后的奖励字典
        """
        normalized_rewards = {}
        
        for component, value in rewards.items():
            if component == 'total':
                normalized_rewards[component] = value
                continue
                
            history = self.reward_history.get(component, [])
            if not history:
                normalized_rewards[component] = value
                continue
            
            if method == 'min_max':
                min_val = min(history)
                max_val = max(history)
                if max_val > min_val:
                    normalized_rewards[component] = (value - min_val) / (max_val - min_val)
                else:
                    normalized_rewards[component] = value
                    
            elif method == 'z_score':
                mean_val = np.mean(history)
                std_val = np.std(history)
                if std_val > self.epsilon:
                    normalized_rewards[component] = (value - mean_val) / std_val
                else:
                    normalized_rewards[component] = value
                    
            elif method == 'percentile':
                percentile_5 = np.percentile(history, 5)
                percentile_95 = np.percentile(history, 95)
                if percentile_95 > percentile_5:
                    normalized_rewards[component] = (value - percentile_5) / (percentile_95 - percentile_5)
                else:
                    normalized_rewards[component] = value
            else:
                normalized_rewards[component] = value
        
        # 重新计算总奖励
        total_reward = 0.0
        for component, weight in self.reward_weights.items():
            if component in normalized_rewards:
                total_reward += weight * normalized_rewards[component]
        normalized_rewards['total'] = total_reward
        
        return normalized_rewards
    
    def get_reward_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取奖励统计信息"""
        stats = {}
        
        for component, history in self.reward_history.items():
            if history:
                stats[component] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'median': np.median(history),
                    'count': len(history)
                }
            else:
                stats[component] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 
                    'max': 0.0, 'median': 0.0, 'count': 0
                }
        
        return stats
    
    def reset_history(self):
        """重置历史记录"""
        for component in self.reward_history:
            self.reward_history[component].clear()
    
    def adaptive_weight_adjustment(self, performance_window: int = 100):
        """
        自适应调整奖励权重
        
        Args:
            performance_window: 性能评估窗口大小
        """
        # 计算最近性能窗口内各组件的平均奖励
        recent_performance = {}
        
        for component, history in self.reward_history.items():
            if len(history) >= performance_window:
                recent_performance[component] = np.mean(history[-performance_window:])
            elif history:
                recent_performance[component] = np.mean(history)
            else:
                recent_performance[component] = 0.0
        
        # 基于性能调整权重
        if recent_performance:
            # 计算每个组件的相对性能
            total_performance = sum(recent_performance.values())
            
            if total_performance > 0:
                # 重新归一化权重
                for component in self.reward_weights:
                    if component in recent_performance:
                        old_weight = self.reward_weights[component]
                        performance_ratio = recent_performance[component] / total_performance
                        
                        # 保留部分原始权重，进行平滑调整
                        new_weight = 0.7 * old_weight + 0.3 * performance_ratio
                        self.reward_weights[component] = max(0.1, new_weight)  # 最小权重为0.1
        
        # 归一化权重
        total_weight = sum(self.reward_weights.values())
        if total_weight > 0:
            for component in self.reward_weights:
                self.reward_weights[component] /= total_weight