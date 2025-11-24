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
            'accuracy': 0.3,
            'complexity': 0.05,
            'rollout_reward': 0.2
        }
        
        # 归一化权重确保总和为1（除了complexity和stability是惩罚项）
        main_weights = ['structure_alignment', 'data_alignment', 'accuracy']
        # 只有当rollout_reward存在时才加入归一化
        if 'rollout_reward' in self.reward_weights:
            main_weights.append('rollout_reward')
        
        main_sum = sum(self.reward_weights[w] for w in main_weights)
        if main_sum > 0:
            for w in main_weights:
                self.reward_weights[w] /= main_sum
        
        self.temperature = temperature
        self.epsilon = epsilon
        
        # 奖励归一化参数
        self.reward_history = {
            'structure_alignment': [],
            'data_alignment': [],
            'accuracy': [],
            'rollout_reward': []
        }
        self.max_history = 1000
    
    def _calculate_data_alignment_reward(
        self,
        expr_embedding: np.ndarray,
        data_embedding: np.ndarray
    ) -> float:
        """计算数据对齐奖励"""
        expr_tensor = torch.FloatTensor(expr_embedding).unsqueeze(0)
        data_tensor = torch.FloatTensor(data_embedding).unsqueeze(0)

        cosine_sim = F.cosine_similarity(expr_tensor, data_tensor).item()

        alignment_reward = max(0, (cosine_sim + 1) / 2)
        return alignment_reward / self.temperature
    
    def _calculate_structure_alignment_reward(
        self,
        expr_embedding: np.ndarray,
        target_expr_embedding: Optional[np.ndarray] = None
    ) -> float:
        """计算结构对齐奖励"""
        if target_expr_embedding is None:
            return 0.0

        expr_tensor = torch.FloatTensor(expr_embedding).unsqueeze(0)
        target_tensor = torch.FloatTensor(target_expr_embedding).unsqueeze(0)

        cosine_sim = F.cosine_similarity(expr_tensor, target_tensor).item()

        return max(0, (cosine_sim + 1) / 2)
    
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

    def get_normalized_rewards(
        self,
        rewards: Dict[str, float],
        method: str = 'min_max'
    ) -> Dict[str, float]:
        """归一化奖励值"""
        normalized_rewards = {}
        history = self.reward_history

        for component, value in rewards.items():
            if component == 'total':
                normalized_rewards[component] = value
                continue

            component_history = history.get(component, [])
            if not component_history:
                normalized_rewards[component] = value
                continue

            if method == 'min_max':
                min_val, max_val = min(component_history), max(component_history)
                normalized_rewards[component] = (value - min_val) / (max_val - min_val) if max_val > min_val else value
            elif method == 'z_score':
                mean_val = np.mean(component_history)
                std_val = np.std(component_history)
                normalized_rewards[component] = (value - mean_val) / std_val if std_val > self.epsilon else value
            elif method == 'percentile':
                p5, p95 = np.percentile(component_history, [5, 95])
                normalized_rewards[component] = (value - p5) / (p95 - p5) if p95 > p5 else value
            else:
                normalized_rewards[component] = value

        normalized_rewards['total'] = sum(self.reward_weights[c] * normalized_rewards.get(c, 0.0) for c in self.reward_weights)
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
        recent_performance = {}
        for component, history in self.reward_history.items():
            if history:
                recent_performance[component] = np.mean(history[-performance_window:] if len(history) >= performance_window else history)
            else:
                recent_performance[component] = 0.0

        if recent_performance and sum(recent_performance.values()) > 0:
            for component in self.reward_weights:
                if component in recent_performance:
                    old_weight = self.reward_weights[component]
                    performance_ratio = recent_performance[component] / sum(recent_performance.values())
                    self.reward_weights[component] = max(0.1, 0.7 * old_weight + 0.3 * performance_ratio)

        total_weight = sum(self.reward_weights.values())
        if total_weight > 0:
            for component in self.reward_weights:
                self.reward_weights[component] /= total_weight

    def calculate_reward(
        self,
        expression: str,
        expr_embedding: np.ndarray,
        task_embedding: Optional[np.ndarray] = None,
        target_embedding: Optional[np.ndarray] = None,
        task_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        rollout_path: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, float]]:
        reward_dict = {
            'accuracy': 0.0,
            'data_alignment': 0.0,
            'structure_alignment': 0.0,
            'complexity': 0.0,
            'stability': 0.0,
            'rollout_reward': 0.0
        }

        if task_embedding is not None:
            reward_dict['data_alignment'] = self._calculate_data_alignment_reward(
                expr_embedding, task_embedding
            )

        if target_embedding is not None:
            reward_dict['structure_alignment'] = self._calculate_structure_alignment_reward(
                expr_embedding, target_embedding
            )

        if task_data is not None:
            X, y = task_data
            y_pred = self._safe_eval_expression(expression, X)
            if np.all(np.isfinite(y_pred)):
                y_mean = np.mean(y)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y_mean) ** 2)
                if ss_tot > 1e-10:
                    r2_ratio = np.clip(ss_res / ss_tot, 0, 1e6)
                    r2_score = 1 - r2_ratio
                    reward_dict['accuracy'] = self._calculate_accuracy_reward(r2_score)

        complexity = self._calculate_expression_complexity(expression)
        reward_dict['complexity'] = self._calculate_complexity_penalty(complexity)

        if rollout_path is not None:
            reward_dict['rollout_reward'] = self._calculate_rollout_reward(rollout_path, task_data)

        total_reward = sum(self.reward_weights[c] * reward_dict[c] for c in reward_dict if c in self.reward_weights)

        return total_reward, reward_dict

    def _calculate_rollout_reward(
        self,
        rollout_path: List[str],
        task_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> float:
        if len(rollout_path) < 2:
            return 0.0

        reward = 0.0
        path_length_reward = min(len(rollout_path) / 10.0, 1.0)
        reward += 0.3 * path_length_reward

        unique_expressions = len(set(rollout_path))
        diversity_reward = unique_expressions / len(rollout_path)
        reward += 0.2 * diversity_reward

        final_expression = rollout_path[-1]
        complexity = self._calculate_expression_complexity(final_expression)
        optimal_complexity = 50.0
        complexity_reward = max(0.0, 1.0 - abs(complexity - optimal_complexity) / optimal_complexity)
        reward += 0.3 * complexity_reward

        if task_data is not None:
            X, y = task_data
            try:
                y_pred = self._safe_eval_expression(final_expression, X)
                if np.all(np.isfinite(y_pred)):
                    y_mean = np.mean(y)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y_mean) ** 2)
                    if ss_tot > 1e-10:
                        r2_ratio = np.clip(ss_res / ss_tot, 0, 1e6)
                        final_r2 = 1 - r2_ratio
                        if final_r2 > -np.inf:
                            accuracy_reward = max(0.0, min(1.0, final_r2))
                            reward += 0.2 * accuracy_reward
            except:
                pass

        return min(reward, 2.0)

    def _calculate_expression_complexity(self, expression: str) -> float:
        complexity = len(expression)
        for op in ['+', '-', '*', '/', '^']:
            complexity += expression.count(op) * 1.0
        for func in ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs']:
            complexity += expression.count(func) * 2.0
        complexity += expression.count('(') * 0.5

        max_nesting = 0
        current_nesting = 0
        for char in expression:
            if char == '(':
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == ')':
                current_nesting -= 1

        complexity += max_nesting * 2.0
        return complexity

    def _safe_eval_expression(self, expression: str, X: np.ndarray) -> np.ndarray:
        n_vars = X.shape[1]
        var_dict = {}
        for i in range(n_vars):
            var_dict[f'x{i + 1}'] = X[:, i]

        var_dict.update({
            'sin': np.sin,
            'cos': np.cos,
            'tan': lambda x: np.clip(np.tan(x), -1e6, 1e6),
            'exp': lambda x: np.clip(np.exp(x), -1e6, 1e6),
            'log': lambda x: np.log(np.clip(x, 1e-10, np.inf)),
            'sqrt': lambda x: np.sqrt(np.clip(x, 0, np.inf)),
            'abs': np.abs,
        })

        expr_str = expression.replace('^', '**')
        safe_dict = {"__builtins__": {}}
        return eval(expr_str, safe_dict, var_dict)