"""
MCTS引擎

实现蒙特卡洛树搜索，用于在语义空间中高效探索表达式结构。
支持冻结编码器推理和在线微调两种模式。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
import math
import random
from collections import defaultdict
import copy
import re

from .reward_calculator import RewardCalculator


class MCTSNode:
    """MCTS节点类"""

    def __init__(
        self,
        expression: Optional[str] = None,
        parent: Optional['MCTSNode'] = None,
        depth: int = 0,
        max_depth: int = 10
    ):
        """
        初始化MCTS节点

        Args:
            expression: 当前节点表示的表达式字符串
            parent: 父节点
            depth: 当前深度
            max_depth: 最大搜索深度
        """
        self.expression = expression
        self.parent = parent
        self.depth = depth
        self.max_depth = max_depth

        # MCTS统计信息
        self.visits = 0
        self.value_sum = 0.0
        self.total_reward = 0.0

        # 奖励分解
        self.reward_breakdown = {
            'accuracy': 0.0,
            'data_alignment': 0.0,
            'structure_alignment': 0.0,
            'complexity': 0.0,
            'stability': 0.0
        }

        # 子节点
        self.children = []
        self.children_dict = {}  # 用于快速查找子节点

        # 是否为叶子节点
        self.is_terminal = depth >= max_depth

        # 节点对应的数据嵌入（对于某些任务可缓存）
        self.data_embedding = None

    def add_child(self, child_expression: str) -> 'MCTSNode':
        """添加子节点"""
        # 检查是否已存在
        if child_expression in self.children_dict:
            return self.children_dict[child_expression]

        # 创建新子节点
        child = MCTSNode(
            expression=child_expression,
            parent=self,
            depth=self.depth + 1,
            max_depth=self.max_depth
        )

        self.children.append(child)
        self.children_dict[child_expression] = child

        return child

    def update(self, reward: float, reward_breakdown: Optional[Dict[str, float]] = None):
        """更新节点统计信息"""
        self.visits += 1
        self.value_sum += reward
        self.total_reward = self.value_sum / self.visits

        if reward_breakdown:
            # 更新奖励分解
            for key, value in reward_breakdown.items():
                if key in self.reward_breakdown:
                    self.reward_breakdown[key] = value

    def get_average_reward(self) -> float:
        """获取平均奖励"""
        return self.total_reward

    def get_ucb_score(self, exploration_constant: float = 1.4) -> float:
        """计算UCB (Upper Confidence Bound) 分数"""
        if self.parent is None:
            return float('inf') if self.visits == 0 else self.total_reward

        if self.visits == 0:
            return float('inf')

        # UCB公式: Q + c * sqrt(ln(N) / n)
        # 其中 Q 是平均价值，c 是探索常数，N 是父节点访问次数，n 是当前节点访问次数
        exploitation = self.total_reward
        exploration = exploration_constant * math.sqrt(
            math.log(max(1, self.parent.visits)) / self.visits
        )

        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        """检查节点是否已完全展开"""
        return len(self.children) > 0 and all(
            child.visits > 0 for child in self.children
        )

    def get_most_visited_child(self) -> Optional['MCTSNode']:
        """获取访问次数最多的子节点"""
        if not self.children:
            return None

        return max(self.children, key=lambda c: c.visits)

    def get_best_child(self, by: str = 'reward') -> Optional['MCTSNode']:
        """获取最佳子节点"""
        if not self.children:
            return None

        if by == 'reward':
            return max(self.children, key=lambda c: c.total_reward)
        elif by == 'visits':
            return max(self.children, key=lambda c: c.visits)
        elif by == 'ucb':
            return max(self.children, key=lambda c: c.get_ucb_score())
        else:
            return max(self.children, key=lambda c: c.total_reward)

    def get_all_expressions(self) -> List[str]:
        """获取从根到当前节点的所有表达式"""
        expressions = []
        node = self
        while node is not None:
            if node.expression is not None:
                expressions.append(node.expression)
            node = node.parent
        return list(reversed(expressions))


class MCTSEngine:
    """MCTS引擎主类"""

    def __init__(
        self,
        expression_encoder,
        data_encoder,
        reward_calculator: RewardCalculator,
        config: Dict[str, Any],
        device: str = 'cpu',
        freeze_encoders: bool = False
    ):
        """
        初始化MCTS引擎

        Args:
            expression_encoder: 表达式编码器
            data_encoder: 数据编码器
            reward_calculator: 奖励计算器
            config: 配置字典
            device: 设备 (cuda/cpu)
            freeze_encoders: 是否冻结编码器（用于推理阶段）
        """
        self.expression_encoder = expression_encoder
        self.data_encoder = data_encoder
        self.reward_calculator = reward_calculator
        self.config = config
        self.device = device
        self.freeze_encoders = freeze_encoders

        # MCTS参数
        self.max_depth = config.get('max_depth', 10)
        self.max_iterations = config.get('max_iterations', 1000)
        self.exploration_constant = config.get('exploration_constant', 1.4)
        self.simulation_count = config.get('simulation_count', 10)

        # 表达式构建参数
        self.max_variables = config.get('max_variables', 5)
        self.allowed_operators = config.get('allowed_operators', [
            '+', '-', '*', '/', '^'
        ])
        self.allowed_functions = config.get('allowed_functions', [
            'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs'
        ])

        # 预定义表达式模板（用于快速扩展）
        self.expression_templates = config.get('expression_templates', [
            # 基础操作
            "x{i}",
            "x{i} + x{j}",
            "x{i} - x{j}",
            "x{i} * x{j}",
            "x{i} / (x{j} + 1e-8)",
            "x{i}^2",
            "x{i}^3",

            # 三角函数
            "sin(x{i})",
            "cos(x{i})",
            "sin(x{i} + x{j})",
            "cos(x{i} * x{j})",

            # 指数和对数
            "exp(x{i})",
            "log(abs(x{i}) + 1e-8)",
            "exp(-x{i}^2)",

            # 复合表达式
            "x{i} * sin(x{j})",
            "x{i} + cos(x{j})",
            "sin(x{i}) * cos(x{j})",
            "x{i}^2 + x{j}^2",
            "x{i} * exp(-x{j})",
            "sqrt(x{i}^2 + x{j}^2)",
            "x{i}^3 + x{j}^3 - x{i} * x{j}",
            "sin(x{i} * x{j}) + cos(x{i} / x{j})",
            "exp(-x{i}^2) * sin(x{j})",
            "log(x{i}^2 + x{j}^2) + x{i}",
        ])

        # 缓存
        self.embedding_cache = {}  # 缓存表达式嵌入
        self.expr_cache = {}  # 缓存表达式计算结果

        # 统计信息
        self.statistics = {
            'total_iterations': 0,
            'total_simulations': 0,
            'unique_expressions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    def search(
        self,
        task_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        task_embedding: Optional[torch.Tensor] = None,
        target_expression: Optional[str] = None,
        verbose: bool = False
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        执行MCTS搜索

        Args:
            task_data: 任务数据 (X, y)
            task_embedding: 任务数据的嵌入向量
            target_expression: 目标表达式（真实解，用于微调阶段）
            verbose: 是否打印详细信息

        Returns:
            (最佳表达式, 最佳奖励, 奖励分解)
        """
        # 预处理任务数据
        if task_embedding is None and task_data is not None:
            X, y = task_data
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            if self.freeze_encoders:
                self.expression_encoder.eval()
                self.data_encoder.eval()
                with torch.no_grad():
                    task_embedding = self.data_encoder.encode(X_tensor, y_tensor)
            else:
                task_embedding = self.data_encoder.encode(X_tensor, y_tensor)

        # 初始化根节点
        root = MCTSNode(depth=0, max_depth=self.max_depth)

        # 主搜索循环
        best_expression = None
        best_reward = float('-inf')
        best_reward_breakdown = {}

        for iteration in range(self.max_iterations):
            # MCTS的四个步骤：选择、扩展、模拟、反向传播

            # 1. 选择 (Selection)
            node = self._select_node(root)

            # 2. 扩展 (Expansion)
            if not node.is_terminal:
                node = self._expand_node(node)

            # 3. 模拟 (Simulation/Rollout)
            if node is not None and node.expression:
                simulation_reward, reward_breakdown = self._simulate(
                    node.expression,
                    task_embedding,
                    target_expression,
                    task_data
                )

                # 4. 反向传播 (Backpropagation)
                self._backpropagate(node, simulation_reward, reward_breakdown)

                # 更新最佳解
                if simulation_reward > best_reward:
                    best_reward = simulation_reward
                    best_expression = node.expression
                    best_reward_breakdown = copy.deepcopy(reward_breakdown)
            else:
                # 如果节点没有表达式，使用默认奖励
                simulation_reward = 0.0
                reward_breakdown = {}

            # 更新统计信息
            self.statistics['total_iterations'] += 1
            self.statistics['total_simulations'] += 1

            # 打印进度
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, "
                      f"Best Reward: {best_reward:.4f}, "
                      f"Best Expression: {best_expression}")

        # 获取最终结果
        if best_expression is None:
            best_child = root.get_best_child(by='reward')
            if best_child:
                best_expression = best_child.expression
                best_reward = best_child.get_average_reward()
                best_reward_breakdown = best_child.reward_breakdown

        # 打印最终统计信息
        if verbose:
            self._print_statistics()

        return best_expression or "", best_reward, best_reward_breakdown

    def _select_node(self, root: MCTSNode) -> MCTSNode:
        """选择步骤：使用UCB选择最佳节点"""
        node = root

        # 沿着UCB分数最高的路径向下选择
        while not node.is_terminal and node.is_fully_expanded():
            node = node.get_best_child(by='ucb')

        return node

    def _expand_node(self, node: MCTSNode) -> Optional[MCTSNode]:
        """扩展步骤：添加新的子节点"""
        # 获取候选表达式
        candidates = self._generate_candidate_expressions(node)

        if not candidates:
            return node  # 如果没有候选表达式，返回当前节点

        # 随机选择一个候选表达式
        new_expression = random.choice(candidates)

        # 添加为子节点
        child = node.add_child(new_expression)

        return child

    def _generate_candidate_expressions(self, node: MCTSNode) -> List[str]:
        """生成候选表达式"""
        candidates = []
        depth = node.depth

        if depth == 0:
            # 根节点：生成基础表达式
            for i in range(1, min(3, self.max_variables) + 1):
                candidates.append(f"x{i}")
        else:
            # 非根节点：基于模板生成复合表达式
            for template in self.expression_templates:
                try:
                    # 替换模板中的变量
                    expr = self._fill_template(template, depth)
                    if expr and self._is_valid_expression(expr):
                        candidates.append(expr)
                except Exception:
                    continue

        # 去重
        candidates = list(set(candidates))

        # 限制候选数量
        if len(candidates) > 20:
            candidates = random.sample(candidates, 20)

        return candidates

    def _fill_template(self, template: str, depth: int) -> Optional[str]:
        """填充表达式模板"""
        try:
            # 替换变量索引
            expr = template.replace('{i}', str(random.randint(1, min(3, self.max_variables))))
            expr = expr.replace('{j}', str(random.randint(1, min(3, self.max_variables))))

            # 检查深度，避免过于复杂
            if depth >= self.max_depth - 2:
                # 接近最大深度时，只返回简单表达式
                if 'sin' in expr or 'cos' in expr or 'exp' in expr:
                    return f"x{random.randint(1, min(3, self.max_variables))}"
                if '^' in expr:
                    expr = re.sub(r'\^\d+', '^2', expr)

            return expr
        except Exception:
            return None

    def _is_valid_expression(self, expression: str) -> bool:
        """检查表达式是否有效"""
        if not expression or len(expression) > 200:
            return False

        # 基本语法检查
        try:
            # 检查括号匹配
            if expression.count('(') != expression.count(')'):
                return False

            # 检查基本模式
            invalid_patterns = [
                r'\.\.',  # 双点
                r'\+\+',  # 双加号
                r'--',    # 双减号
                r'\*\*',  # 双星号（除了指数）
            ]

            for pattern in invalid_patterns:
                if re.search(pattern, expression):
                    return False

            return True

        except Exception:
            return False

    def _simulate(
        self,
        expression: str,
        task_embedding: Optional[torch.Tensor],
        target_expression: Optional[str],
        task_data: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[float, Dict[str, float]]:
        """模拟步骤：评估表达式的奖励"""
        try:
            # 计算表达式嵌入
            expr_embedding = self._get_expression_embedding(expression)

            # 计算奖励
            reward_dict = self._calculate_reward(
                expression,
                expr_embedding,
                task_embedding,
                target_expression,
                task_data
            )

            # 提取总奖励
            total_reward = reward_dict.get('total', 0.0)

            return total_reward, reward_dict

        except Exception as e:
            # 如果评估失败，返回低奖励
            return 0.0, {
                'accuracy': 0.0,
                'data_alignment': 0.0,
                'structure_alignment': 0.0,
                'complexity': 0.0,
                'total': 0.0
            }

    def _get_expression_embedding(self, expression: str) -> torch.Tensor:
        """获取表达式嵌入（带缓存）"""
        # 检查缓存
        if expression in self.embedding_cache:
            self.statistics['cache_hits'] += 1
            return self.embedding_cache[expression]

        self.statistics['cache_misses'] += 1

        # 计算嵌入
        if self.freeze_encoders:
            self.expression_encoder.eval()
            with torch.no_grad():
                embedding = self.expression_encoder.encode(expression, training=False)
        else:
            embedding = self.expression_encoder.encode(expression, training=True)

        # 缓存
        self.embedding_cache[expression] = embedding

        return embedding

    def _calculate_reward(
        self,
        expression: str,
        expr_embedding: torch.Tensor,
        task_embedding: Optional[torch.Tensor],
        target_expression: Optional[str],
        task_data: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """计算复合奖励"""
        reward_dict = {
            'accuracy': 0.0,
            'data_alignment': 0.0,
            'structure_alignment': 0.0,
            'complexity': 0.0,
            'stability': 0.0
        }

        # 1. 数据对齐奖励
        if task_embedding is not None:
            reward_dict['data_alignment'] = self.reward_calculator._calculate_data_alignment_reward(
                expr_embedding.cpu().numpy(),
                task_embedding.cpu().numpy()
            )

        # 2. 结构对齐奖励（仅在微调阶段使用）
        if target_expression is not None:
            try:
                target_embedding = self._get_expression_embedding(target_expression)
                reward_dict['structure_alignment'] = self.reward_calculator._calculate_structure_alignment_reward(
                    expr_embedding.cpu().numpy(),
                    target_embedding.cpu().numpy()
                )
            except Exception:
                reward_dict['structure_alignment'] = 0.0

        # 3. 准确度奖励（真实精度）
        if task_data is not None:
            X, y = task_data
            try:
                r2_score = self._evaluate_expression_accuracy(expression, X, y)
                reward_dict['accuracy'] = self.reward_calculator._calculate_accuracy_reward(r2_score)
            except Exception:
                reward_dict['accuracy'] = 0.0

        # 4. 复杂度惩罚
        complexity = self._calculate_expression_complexity(expression)
        reward_dict['complexity'] = self.reward_calculator._calculate_complexity_penalty(complexity)

        # 计算总奖励
        total_reward = 0.0
        for component, weight in self.reward_calculator.reward_weights.items():
            if component in reward_dict:
                total_reward += weight * reward_dict[component]

        reward_dict['total'] = total_reward

        return reward_dict

    def _evaluate_expression_accuracy(
        self,
        expression: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """评估表达式的准确度（R²分数）"""
        try:
            # 检查缓存
            cache_key = f"{expression}_{hash(str(X.shape))}_{hash(str(y.shape))}"
            if cache_key in self.expr_cache:
                return self.expr_cache[cache_key]

            # 安全求值
            y_pred = self._safe_eval_expression(expression, X)

            # 计算R²分数
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            # 避免除零
            if ss_tot == 0:
                r2_score = 0.0 if np.allclose(y_pred, y) else -np.inf
            else:
                r2_score = 1 - (ss_res / ss_tot)

            # 缓存结果
            self.expr_cache[cache_key] = r2_score

            return r2_score

        except Exception:
            return -np.inf

    def _safe_eval_expression(self, expression: str, X: np.ndarray) -> np.ndarray:
        """安全地求值表达式"""
        # 获取变量数量
        n_vars = X.shape[1]

        # 构建变量字典
        var_dict = {}
        for i in range(n_vars):
            var_dict[f'x{i + 1}'] = X[:, i]

        # 添加数学函数
        var_dict.update({
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
        })

        # 处理表达式
        expr_str = expression.replace('^', '**')

        # 限制eval的安全性
        safe_dict = {"__builtins__": {}}

        y_pred = eval(expr_str, safe_dict, var_dict)

        return y_pred

    def _calculate_expression_complexity(self, expression: str) -> float:
        """计算表达式复杂度"""
        complexity = 0.0

        # 基础复杂度：token数量
        complexity += len(expression)

        # 运算符复杂度
        for op in ['+', '-', '*', '/', '^']:
            complexity += expression.count(op) * 1.0

        # 函数复杂度
        for func in self.allowed_functions:
            complexity += expression.count(func) * 2.0

        # 括号复杂度
        complexity += expression.count('(') * 0.5

        # 嵌套复杂度（粗略估计）
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

    def _backpropagate(
        self,
        node: MCTSNode,
        reward: float,
        reward_breakdown: Dict[str, float]
    ):
        """反向传播：更新路径上所有节点的统计信息"""
        current = node
        while current is not None:
            current.update(reward, reward_breakdown)
            current = current.parent

    def _print_statistics(self):
        """打印搜索统计信息"""
        print("\n=== MCTS搜索统计 ===")
        for key, value in self.statistics.items():
            print(f"{key}: {value}")
        print(f"缓存命中率: {self.statistics['cache_hits'] / max(1, self.statistics['cache_hits'] + self.statistics['cache_misses']):.2%}")
        print("=====================\n")

    def reset_cache(self):
        """重置缓存"""
        self.embedding_cache.clear()
        self.expr_cache.clear()
        self.statistics['cache_hits'] = 0
        self.statistics['cache_misses'] = 0

    def get_search_tree_info(self, root: MCTSNode) -> Dict[str, Any]:
        """获取搜索树信息"""
        def collect_info(node: MCTSNode) -> Dict[str, Any]:
            info = {
                'expression': node.expression,
                'depth': node.depth,
                'visits': node.visits,
                'value': node.total_reward,
                'reward_breakdown': node.reward_breakdown,
                'children': []
            }

            for child in node.children:
                info['children'].append(collect_info(child))

            return info

        return collect_info(root)
