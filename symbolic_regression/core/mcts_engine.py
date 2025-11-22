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
        max_depth: int = 10,
        feature_count: int = 3
    ):
        """
        初始化MCTS节点

        Args:
            expression: 当前节点表示的表达式字符串
            parent: 父节点
            depth: 当前深度
            max_depth: 最大搜索深度
            feature_count: 数据特征数量，用于动态生成变量
        """
        self.expression = expression
        self.parent = parent
        self.depth = depth
        self.max_depth = max_depth
        self.feature_count = feature_count

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
            'stability': 0.0,
            'rollout_reward': 0.0
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
            max_depth=self.max_depth,
            feature_count=self.feature_count
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

        # 缓存清理配置
        self.cache_size_limit = config.get('cache_size_limit', 5000)  # 缓存大小限制
        self.cache_cleanup_interval = config.get('cache_cleanup_interval', 100)  # 清理间隔（迭代次数）
        self.last_cache_cleanup = 0  # 上次清理时的迭代计数
        self.auto_cleanup_enabled = config.get('auto_cleanup_enabled', True)  # 是否启用自动清理

        # 统计信息
        self.statistics = {
            'total_iterations': 0,
            'total_simulations': 0,
            'unique_expressions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_cleanups': 0,  # 缓存清理次数
            'gpu_memory_peak': 0,  # GPU内存峰值
            'gpu_cache_cleanups': 0,  # GPU缓存清理次数
        }

        # 初始化GPU内存监控
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()  # 预清理

    def search(
        self,
        task_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        task_embedding: Optional[torch.Tensor] = None,
        target_expression: Optional[str] = None,
        verbose: bool = False
    ) -> Tuple[str, Dict[str, float]]:
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

        # 提取任务数据的特征数
        feature_count = 3  # 默认值
        if task_data is not None:
            X, y = task_data
            feature_count = X.shape[1]
        
        # 初始化根节点
        root = MCTSNode(depth=0, max_depth=self.max_depth, feature_count=feature_count)

        # 主搜索循环
        best_expression = None
        best_reward = float('-inf')

        for iteration in range(self.max_iterations):
            # MCTS的四个步骤：选择、扩展、模拟、反向传播

            # 1. 选择 (Selection)
            node = self._select_node(root)

            # 2. 扩展 (Expansion)
            if not node.is_terminal:
                node = self._expand_node(node)

            # 3. 模拟 (Simulation/Rollout)
            simulation_reward, reward_breakdown = self._simulate(
                node.expression,
                task_embedding,
                target_expression,
                task_data
            )

            # 4. 反向传播 (Backpropagation)
            self._backpropagate(node, simulation_reward, reward_breakdown)

            # 更新最佳解（使用rollout的最终表达式）
            if simulation_reward > best_reward:
                best_reward = simulation_reward
                # 注意：这里我们仍然使用原始节点表达式，但在实际应用中，
                # 可能需要记录rollout的最终表达式
                best_expression = node.expression

            # 更新统计信息
            self.statistics['total_iterations'] += 1
            self.statistics['total_simulations'] += 1

        return best_expression,best_reward

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
        candidates = self._generate_candidate_expressions(node, node.feature_count)

        if not candidates:
            return node  # 如果没有候选表达式，返回当前节点

        # 随机选择一个候选表达式
        new_expression = random.choice(candidates)

        # 添加为子节点
        child = node.add_child(new_expression)

        return child

    def _generate_candidate_expressions(self, node: MCTSNode, feature_count: int) -> List[str]:
        """生成候选表达式"""
        candidates = []
        depth = node.depth

        if depth == 0:
            # 根节点：生成基础表达式，使用实际特征数
            for i in range(1, min(feature_count, self.max_variables) + 1):
                candidates.append(f"x{i}")
        else:
            # 非根节点：基于模板生成复合表达式
            for template in self.expression_templates:
                try:
                    # 替换模板中的变量
                    expr = self._fill_template(template, depth, feature_count)
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

    def _fill_template(self, template: str, depth: int, feature_count: int) -> Optional[str]:
        """填充表达式模板"""
        try:
            # 替换变量索引，使用实际特征数
            max_var = min(feature_count, self.max_variables)
            expr = template.replace('{i}', str(random.randint(1, max_var)))
            expr = expr.replace('{j}', str(random.randint(1, max_var)))

            # 检查深度，避免过于复杂
            if depth >= self.max_depth - 2:
                # 接近最大深度时，只返回简单表达式
                if 'sin' in expr or 'cos' in expr or 'exp' in expr:
                    return f"x{random.randint(1, min(feature_count, self.max_variables))}"
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
        """模拟步骤：通过rollout评估表达式的奖励"""
        # 提取特征数用于rollout
        feature_count = 3  # 默认值
        if task_data is not None:
            X, y = task_data
            feature_count = X.shape[1]
        
        # 执行rollout：从当前表达式开始随机扩展
        rollout_expression, rollout_path = self._rollout(
            expression, 
            remaining_depth=self.max_depth - self._get_expression_depth(expression),
            feature_count=feature_count
        )

        # 计算最终表达式的奖励
        expr_embedding = self._get_expression_embedding(rollout_expression)

        # 计算奖励（包含rollout路径奖励）
        total_reward, reward_dict = self._calculate_reward(
            rollout_expression,
            expr_embedding,
            task_embedding,
            target_expression,
            task_data,
            rollout_path=rollout_path  # 传递rollout路径
        )

        # GPU内存优化：及时清理临时变量
        del expr_embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return total_reward, reward_dict


    def _rollout(self, start_expression: str, remaining_depth: int, feature_count: int = 3) -> Tuple[str, List[str]]:
        """执行rollout：从起始表达式随机扩展到最大深度
        
        Args:
            start_expression: 起始表达式
            remaining_depth: 剩余扩展深度
            
        Returns:
            (最终表达式, 扩展路径)
        """
        if remaining_depth <= 0:
            return start_expression, [start_expression]

        current_expression = start_expression
        rollout_path = [current_expression]
        depth = 0
        
        # 随机扩展直到达到最大深度
        while depth < remaining_depth:
            # 生成候选扩展
            candidates = self._generate_rollout_candidates(current_expression, feature_count)
            
            if not candidates:
                break
            
            # 随机选择一个扩展
            next_expression = random.choice(candidates)
            current_expression = next_expression
            rollout_path.append(current_expression)
            depth += 1
            
            # 简单的停止条件：表达式过于复杂时提前停止
            if self._is_expression_too_complex(current_expression):
                break
        
        return current_expression, rollout_path

    def _generate_rollout_candidates(self, current_expression: str, feature_count: int) -> List[str]:
        """为rollout生成候选扩展
        
        Args:
            current_expression: 当前表达式
            feature_count: 特征数量
            
        Returns:
            候选扩展表达式列表
        """
        candidates = []
        
        # 基于当前表达式生成扩展
        base_candidates = self._generate_base_extensions(current_expression)
        template_candidates = self._generate_template_extensions(feature_count)
        
        candidates.extend(base_candidates)
        candidates.extend(template_candidates)
        
        # 去重并过滤
        candidates = list(set(candidates))
        candidates = [expr for expr in candidates if self._is_valid_expression(expr)]
        
        # 限制候选数量以避免计算爆炸
        # if len(candidates) > 10:
        #     candidates = random.sample(candidates, 10)
            
        return candidates

    def _generate_base_extensions(self, current_expression: str) -> List[str]:
        """基于当前表达式生成基础扩展"""
        extensions = []
        
        # 获取变量
        variables = self._extract_variables(current_expression)
        
        if not variables:
            return []
        
        # 生成基础操作扩展
        if len(variables) >= 2:
            var1, var2 = random.sample(variables, 2)
        else:
            var1 = variables[0]
            var2 = variables[0]  # 使用相同变量
        
        # 加法扩展
        extensions.append(f"({current_expression}) + {var1}")
        if len(variables) >= 2:
            extensions.append(f"({current_expression}) + {var2}")
        
        # 乘法扩展
        extensions.append(f"({current_expression}) * {var1}")
        if len(variables) >= 2:
            extensions.append(f"({current_expression}) * {var2}")
        
        # 函数扩展
        extensions.append(f"sin(({current_expression}) + {var1})")
        if len(variables) >= 2:
            extensions.append(f"cos(({current_expression}) * {var2})")
            extensions.append(f"exp(({current_expression}) / {var2})")
        
        # 替换扩展
        for var in variables:
            extensions.append(f"({current_expression}) - {var}")
            if '^' not in current_expression:
                extensions.append(f"({current_expression})^2")
        
        return extensions

    def _generate_template_extensions(self, feature_count: int) -> List[str]:
        """生成基于模板的扩展
        
        Args:
            feature_count: 特征数量
        """
        if not self.expression_templates:
            return []
        
        # 随机选择一些模板
        selected_templates = random.sample(
            self.expression_templates, 
            min(5, len(self.expression_templates))
        )
        
        extensions = []
        for template in selected_templates:
            try:
                expr = self._fill_template(template, depth=1, feature_count=feature_count)
                if expr and self._is_valid_expression(expr):
                    extensions.append(expr)
            except Exception:
                continue
        
        return extensions

    def _get_expression_depth(self, expression: str) -> int:
        """估算表达式当前的深度（简化计算）"""
        depth = 0
        
        # 计算括号嵌套深度
        max_nesting = 0
        current_nesting = 0
        for char in expression:
            if char == '(':
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == ')':
                current_nesting -= 1
        
        depth = max_nesting
        
        # 根据运算符数量调整
        operator_count = sum(1 for op in ['+', '-', '*', '/', '^'] if op in expression)
        depth += operator_count // 2  # 每2个运算符增加1层深度
        
        return depth

    def _extract_variables(self, expression: str) -> List[str]:
        """从表达式中提取变量"""
        variables = []
        import re
        
        # 匹配 x1, x2, x3 等变量
        var_pattern = r'x\d+'
        matches = re.findall(var_pattern, expression)
        
        # 去重并排序
        variables = sorted(list(set(matches)))
        
        return variables

    def _is_expression_too_complex(self, expression: str) -> bool:
        """检查表达式是否过于复杂"""
        # 长度检查
        if len(expression) > 100:
            return True
        
        # 嵌套深度检查
        if self._get_expression_depth(expression) > 8:
            return True
        
        # 括号平衡检查
        if expression.count('(') != expression.count(')'):
            return True
        
        return False

    def _get_expression_embedding(self, expression: str, force_cpu: bool = False) -> torch.Tensor:
        """获取表达式嵌入（带缓存和GPU内存优化）"""
        # 检查缓存
        if expression in self.embedding_cache:
            self.statistics['cache_hits'] += 1
            cached_embedding = self.embedding_cache[expression]
            
            # 如果请求CPU模式且当前在GPU上，转移
            if force_cpu and cached_embedding.is_cuda:
                return cached_embedding.cpu()
            return cached_embedding

        self.statistics['cache_misses'] += 1

        try:
            # 计算嵌入
            if self.freeze_encoders:
                self.expression_encoder.eval()
                with torch.no_grad():
                    embedding = self.expression_encoder.encode(expression, training=False)
            else:
                embedding = self.expression_encoder.encode(expression, training=True)

            # GPU内存优化：使用detach()和可选的CPU转换
            if force_cpu:
                # 如果需要CPU模式，直接计算在CPU上
                if embedding.is_cuda:
                    embedding = embedding.cpu()
            else:
                # 保持GPU内存但detach避免梯度追踪
                embedding = embedding.detach().clone()

            # 缓存策略：如果缓存太大或请求CPU模式，存储CPU版本
            if force_cpu or self._should_cache_on_cpu():
                self.embedding_cache[expression] = embedding.cpu()
            else:
                self.embedding_cache[expression] = embedding

            return self.embedding_cache[expression]

        except Exception as e:
            print(f"警告: 计算嵌入失败 {expression}: {e}")
            # 返回零张量
            if self.freeze_encoders:
                self.expression_encoder.eval()
                with torch.no_grad():
                    zero_embedding = torch.zeros(512, dtype=torch.float32)
                    if not force_cpu and torch.cuda.is_available():
                        zero_embedding = zero_embedding.cuda()
                    return zero_embedding
            else:
                zero_embedding = torch.zeros(512, dtype=torch.float32)
                if not force_cpu and torch.cuda.is_available():
                    zero_embedding = zero_embedding.cuda()
                return zero_embedding

    def _should_cache_on_cpu(self) -> bool:
        """判断是否应该将缓存存储在CPU上以节省GPU内存"""
        if not torch.cuda.is_available():
            return True  # 没有GPU，全部放CPU
        
        # 检查GPU内存使用率
        try:
            import psutil
            gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            
            # 如果GPU内存使用超过1GB，建议CPU缓存
            if gpu_memory > 1e9 or gpu_memory_reserved > 2e9:
                return True
        except:
            pass
        
        # 如果缓存项数超过限制，使用CPU缓存
        if len(self.embedding_cache) > self.cache_size_limit // 2:
            return True
        
        return False

    def _calculate_reward(
        self,
        expression: str,
        expr_embedding: torch.Tensor,
        task_embedding: Optional[torch.Tensor],
        target_expression: Optional[str],
        task_data: Optional[Tuple[np.ndarray, np.ndarray]],
        rollout_path: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """计算复合奖励
        
        Returns:
            (总奖励, 奖励分解字典)
        """
        reward_dict = {
            'accuracy': 0.0,
            'data_alignment': 0.0,
            'structure_alignment': 0.0,
            'complexity': 0.0,
            'stability': 0.0,
            'rollout_reward': 0.0
        }

        # 1. 数据对齐奖励
        if task_embedding is not None:
            reward_dict['data_alignment'] = self.reward_calculator._calculate_data_alignment_reward(
                expr_embedding.detach().cpu().numpy(),
                task_embedding.detach().cpu().numpy()
            )

        # 2. 结构对齐奖励（仅在微调阶段使用）
        if target_expression is not None:
            try:
                # 使用GPU内存优化模式获取目标嵌入
                target_embedding = self._get_expression_embedding(target_expression, force_cpu=True)
                
                reward_dict['structure_alignment'] = self.reward_calculator._calculate_structure_alignment_reward(
                    expr_embedding.detach().cpu().numpy(),
                    target_embedding.detach().numpy()
                )
                
                # 及时清理临时变量（如果存在的话）
                del target_embedding
                
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

        # 5. Rollout奖励（新功能）
        if rollout_path is not None:
            reward_dict['rollout_reward'] = self._calculate_rollout_reward(rollout_path, task_data)

        # 计算总奖励
        total_reward = 0.0
        for component, weight in self.reward_calculator.reward_weights.items():
            if component in reward_dict:
                total_reward += weight * reward_dict[component]

        # 如果有rollout奖励，给予额外权重
        if rollout_path is not None:
            rollout_weight = self.config.get('rollout_weight', 0.2)
            total_reward += rollout_weight * reward_dict['rollout_reward']

        return total_reward, reward_dict

    def _calculate_rollout_reward(self, rollout_path: List[str], task_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> float:
        """计算rollout奖励
        
        基于rollout路径的完整性和最终表达能力
        """
        if len(rollout_path) < 2:
            return 0.0
        
        reward = 0.0
        
        # 1. 路径长度奖励（鼓励探索）
        path_length_reward = min(len(rollout_path) / 10.0, 1.0)  # 归一化到[0,1]
        reward += 0.3 * path_length_reward
        
        # 2. 路径多样性奖励（鼓励表达式的多样性）
        unique_expressions = len(set(rollout_path))
        diversity_reward = unique_expressions / len(rollout_path)  # [0,1]
        reward += 0.2 * diversity_reward
        
        # 3. 最终表达式复杂度奖励（适中复杂度的表达式得分更高）
        final_expression = rollout_path[-1]
        complexity = self._calculate_expression_complexity(final_expression)
        
        # 理想的复杂度范围（根据任务调整）
        optimal_complexity = 50.0
        complexity_reward = max(0.0, 1.0 - abs(complexity - optimal_complexity) / optimal_complexity)
        reward += 0.3 * complexity_reward
        
        # 4. 如果有任务数据，评估最终表达式的准确度
        if task_data is not None and len(rollout_path) > 0:
            X, y = task_data
            try:
                final_r2 = self._evaluate_expression_accuracy(final_expression, X, y)
                if final_r2 > -np.inf:  # 有效值
                    # R²分数转换为[0,1]范围
                    accuracy_reward = max(0.0, min(1.0, final_r2))
                    reward += 0.2 * accuracy_reward
            except Exception:
                pass
        
        # 归一化到合理范围
        return min(reward, 2.0)  # 最多2.0分

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

            # 检查预测值是否包含无效值
            if not np.all(np.isfinite(y_pred)):
                r2_score = -np.inf
            else:
                # 使用安全的数值计算
                with np.errstate(invalid='ignore', over='ignore', under='ignore'):
                    # 计算R²分数
                    y_mean = np.mean(y)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y_mean) ** 2)

                    # 避免除零并处理数值问题
                    if ss_tot <= 1e-10:
                        r2_score = 0.0 if np.allclose(y_pred, y_mean, atol=1e-6) else -np.inf
                    else:
                        r2_ratio = ss_res / ss_tot
                        # 确保r2_ratio在合理范围内
                        r2_ratio = np.clip(r2_ratio, 0, 1e6)
                        r2_score = 1 - r2_ratio

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

        # 添加安全的数学函数
        var_dict.update({
            'sin': self._safe_sin,
            'cos': self._safe_cos,
            'tan': self._safe_tan,
            'exp': self._safe_exp,
            'log': self._safe_log,
            'sqrt': self._safe_sqrt,
            'abs': np.abs,
        })

        # 处理表达式
        expr_str = expression.replace('^', '**')

        # 限制eval的安全性
        safe_dict = {"__builtins__": {}}

        y_pred = eval(expr_str, safe_dict, var_dict)

        return y_pred

    def _safe_sin(self, x):
        """安全的sin函数"""
        with np.errstate(invalid='ignore'):
            return np.sin(np.clip(x, -np.pi * 4, np.pi * 4))

    def _safe_cos(self, x):
        """安全的cos函数"""
        with np.errstate(invalid='ignore'):
            return np.cos(np.clip(x, -np.pi * 4, np.pi * 4))

    def _safe_tan(self, x):
        """安全的tan函数"""
        with np.errstate(invalid='ignore'):
            return np.tan(np.clip(x, -np.pi/2 + 1e-6, np.pi/2 - 1e-6))

    def _safe_exp(self, x):
        """安全的exp函数"""
        with np.errstate(over='ignore', under='ignore'):
            return np.exp(np.clip(x, -50, 50))

    def _safe_log(self, x):
        """安全的log函数"""
        with np.errstate(invalid='ignore'):
            return np.log(np.clip(x, 1e-10, np.inf))

    def _safe_sqrt(self, x):
        """安全的sqrt函数"""
        with np.errstate(invalid='ignore'):
            return np.sqrt(np.clip(x, 0, np.inf))

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
        
        # 基础统计信息
        basic_stats = ['total_iterations', 'total_simulations', 'unique_expressions', 
                      'cache_hits', 'cache_misses', 'cache_cleanups']
        for key in basic_stats:
            if key in self.statistics:
                print(f"{key}: {self.statistics[key]}")
        
        # 缓存统计
        cache_hit_rate = self.statistics['cache_hits'] / max(1, self.statistics['cache_hits'] + self.statistics['cache_misses'])
        print(f"缓存命中率: {cache_hit_rate:.2%}")
        print(f"当前缓存大小: {len(self.embedding_cache)}")
        
        # GPU内存统计
        if torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated()
            current_reserved = torch.cuda.memory_reserved()
            peak_allocated = torch.cuda.max_memory_allocated()
            
            print(f"当前GPU已分配内存: {current_allocated / 1e6:.1f} MB")
            print(f"当前GPU保留内存: {current_reserved / 1e6:.1f} MB") 
            print(f"峰值GPU内存: {peak_allocated / 1e6:.1f} MB")
            print(f"GPU缓存清理次数: {self.statistics['gpu_cache_cleanups']}")
            
            # 更新峰值统计
            if peak_allocated > self.statistics.get('gpu_memory_peak', 0):
                self.statistics['gpu_memory_peak'] = peak_allocated
        else:
            print("未检测到GPU设备")
        
        print("=====================\n")

    def _auto_cleanup_cache(self):
        """智能自动清理缓存以释放GPU内存"""
        if not self.auto_cleanup_enabled:
            return

        initial_size = len(self.embedding_cache)
        
        # GPU内存使用情况检查
        gpu_memory_info = {}
        if torch.cuda.is_available():
            gpu_memory_info = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }

        # 智能清理策略
        needs_cleanup = False
        cleanup_reasons = []

        # 策略1：缓存大小超限
        if initial_size > self.cache_size_limit:
            needs_cleanup = True
            cleanup_reasons.append(f"缓存过大({initial_size}>{self.cache_size_limit})")

        # 策略2：GPU内存使用超限（超过2GB）
        if gpu_memory_info.get('allocated', 0) > 2e9:
            needs_cleanup = True
            cleanup_reasons.append(f"GPU内存过多({gpu_memory_info['allocated']/1e9:.1f}GB)")

        # 策略3：GPU内存保留过多（超过3GB）
        if gpu_memory_info.get('reserved', 0) > 3e9:
            needs_cleanup = True
            cleanup_reasons.append(f"GPU保留过多({gpu_memory_info['reserved']/1e9:.1f}GB)")

        if needs_cleanup:
            # 计算清理目标
            if initial_size > self.cache_size_limit:
                target_size = min(self.cache_size_limit // 2, initial_size - 100)
            else:
                target_size = max(100, initial_size // 2)  # 至少保留100项

            items_to_remove = initial_size - target_size
            
            # 智能选择删除的缓存项（优先删除GPU上的项）
            cache_keys = list(self.embedding_cache.keys())
            gpu_keys = [k for k in cache_keys if self.embedding_cache[k].is_cuda]
            cpu_keys = [k for k in cache_keys if not self.embedding_cache[k].is_cuda]
            
            # 优先删除GPU缓存项，然后是CPU缓存项
            keys_to_remove = []
            if gpu_keys:
                keys_to_remove.extend(gpu_keys[:min(len(gpu_keys), items_to_remove)])
                items_to_remove -= len(keys_to_remove)
            
            if items_to_remove > 0 and cpu_keys:
                keys_to_remove.extend(cpu_keys[:min(len(cpu_keys), items_to_remove)])
            
            # 执行清理
            for key in keys_to_remove:
                cached_item = self.embedding_cache.pop(key, None)
                # 立即释放GPU内存
                if cached_item is not None and cached_item.is_cuda:
                    del cached_item
                
                # 清理表达式计算缓存
                keys_to_remove_expr = [k for k in self.expr_cache.keys() if k.startswith(key)]
                for expr_key in keys_to_remove_expr:
                    self.expr_cache.pop(expr_key, None)

            # 清理后的GPU内存释放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.statistics['gpu_cache_cleanups'] += 1

            # 更新统计信息
            self.statistics['cache_cleanups'] += 1
            self.last_cache_cleanup = self.statistics['total_iterations']
            
            final_size = len(self.embedding_cache)
            reason_str = ", ".join(cleanup_reasons)
            print(f"[Auto Cleanup] 原因: {reason_str} | 清理缓存: {initial_size} -> {final_size} 项", flush=True)
            
            # 显示GPU内存使用情况
            if torch.cuda.is_available():
                gpu_memory_after = {
                    'allocated': torch.cuda.memory_allocated(),
                    'reserved': torch.cuda.memory_reserved(),
                }
                freed_allocated = gpu_memory_info['allocated'] - gpu_memory_after['allocated']
                freed_reserved = gpu_memory_info['reserved'] - gpu_memory_after['reserved']
                print(f"[GPU Memory] 释放已分配: {freed_allocated/1e6:.1f}MB, 释放保留: {freed_reserved/1e6:.1f}MB", flush=True)

    def reset_cache(self):
        """重置缓存"""
        # 清空GPU缓存中的张量
        for key, embedding in list(self.embedding_cache.items()):
            if hasattr(embedding, 'is_cuda') and embedding.is_cuda:
                del embedding
        
        # 清空CPU缓存中的张量
        for key, embedding in list(self.embedding_cache.items()):
            if hasattr(embedding, 'cpu'):
                del embedding
                
        self.embedding_cache.clear()
        self.expr_cache.clear()
        
        # 重置统计信息
        self.statistics['cache_hits'] = 0
        self.statistics['cache_misses'] = 0
        self.statistics['cache_cleanups'] = 0
        self.statistics['gpu_cache_cleanups'] = 0
        self.last_cache_cleanup = 0
        
        # 释放GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print("[Cache Reset] 缓存和GPU内存已重置", flush=True)

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
