"""
增强的MCTS引擎

集成表达式编码器和数据编码器的功能，使用三维复合奖励进行探索。
基于原有的MCTS实现，添加了结构引导和数据对齐功能。
"""

import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import sys
import os
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..'))
import nd2py as nd
from nd2py.utils import R2_score, RMSE_score


class EnhancedNode:
    """增强的MCTS节点，支持嵌入向量和复合奖励"""
    
    def __init__(self, eqtrees=None, embedding=None):
        # 支持多个表达式树
        self.eqtrees = eqtrees or [nd.Variable('x1')]
        self.embedding = embedding  # 表达式嵌入向量
        self.parent = None
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.reward = 0.0
        self.r2 = -np.inf
        self.complexity = 0
        self.phi = None  # 组合后的最终表达式
        self.data_embedding = None  # 对应的数据嵌入
        self.structure_alignment = 0.0  # 结构对齐分数
        self.data_alignment = 0.0      # 数据对齐分数
        self.accuracy_reward = 0.0     # 真实精度奖励
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        
    def uct_value(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * np.sqrt(np.log(self.parent.visits + 1) / (self.visits + 1))


class EnhancedMCTSEngine:
    """增强的MCTS引擎，集成了编码器功能"""
    
    def __init__(
        self,
        expression_encoder,
        data_encoder,
        max_depth=12,
        max_iterations=1000,
        max_vars=5,
        exploration_constant=1.4,
        simulation_count=10,
        reward_weights=None,
        device='cpu'
    ):
        self.expression_encoder = expression_encoder
        self.data_encoder = data_encoder
        self.device = device
        
        # MCTS参数
        self.max_vars = max_vars
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.simulation_count = simulation_count
        
        # 奖励权重
        self.reward_weights = reward_weights or {
            'structure_alignment': 0.3,
            'data_alignment': 0.4,
            'accuracy': 0.3
        }
        
        # nd2py操作符
        self.binary_ops = [nd.Add, nd.Sub, nd.Mul, nd.Div]
        self.unary_ops = [nd.Sin, nd.Cos, nd.Sqrt, nd.Log, nd.Exp]
        self.constants = [nd.Number(1), nd.Number(2), nd.Number(0.5)]
        
        # 变量列表
        self.variables = []
        
        # 最佳结果
        self.best_expr = None
        self.best_r2 = -np.inf
        
        # 经验回放池
        self.experience_buffer = deque(maxlen=10000)
        
    def fit(
        self, 
        X, 
        y, 
        variables=None, 
        true_expression=None,
        target_data_embedding=None
    ):
        """
        使用增强的MCTS进行符号回归
        
        Args:
            X: 输入数据
            y: 目标值
            variables: 变量列表
            true_expression: 真实表达式（用于微调）
            target_data_embedding: 目标数据嵌入
        """
        # 预处理数据
        X, y = self._preprocess_data(X, y)
        
        # 设置变量
        if variables is not None:
            self.variables = variables
        else:
            self.variables = [nd.Variable(f'x{i+1}') for i in range(X.shape[1])]
        
        # 计算目标嵌入
        target_expr_embedding = None
        if true_expression is not None:
            target_expr_embedding = self.expression_encoder.encode(true_expression)
            
        if target_data_embedding is None:
            target_data_embedding = self.data_encoder.encode(X, y)
            
        # 初始化根节点
        self.root = EnhancedNode(
            self.variables[:min(len(self.variables), self.max_vars)]
        )
        self.root.data_embedding = target_data_embedding
        
        # 评估根节点
        self._evaluate_node(self.root, X, y, target_data_embedding, target_expr_embedding)
        
        for i in range(self.max_iterations):
            # 1. 选择
            node = self._select(self.root)
            
            # 2. 扩展
            if not self._is_terminal(node) and node.visits > 0:
                node = self._expand(node, X, y, target_data_embedding, target_expr_embedding)
            
            # 3. 模拟
            reward, best_node = self._simulate(
                node, X, y, target_data_embedding, target_expr_embedding
            )
            
            # 4. 回溯
            self._backpropagate(node, reward)
            
            # 更新最佳表达式
            if best_node.r2 > self.best_r2:
                self.best_r2 = best_node.r2
                self.best_expr = best_node.phi
                
            if i % 100 == 0:
                print(f"Iteration {i}: Best R2 = {self.best_r2:.4f}, Best Expr = {self.best_expr}")
                
        return self.best_expr
    
    def predict(self, X):
        """使用最佳表达式进行预测"""
        if self.best_expr is None:
            raise ValueError("Model not fitted yet")
        X, _ = self._preprocess_data(X, None)
        return self._evaluate_expression(self.best_expr, X)
    
    def get_score(self, X, y):
        """获取模型分数"""
        y_pred = self.predict(X)
        r2 = R2_score(y, y_pred)
        rmse = RMSE_score(y, y_pred)
        return r2, rmse
    
    def _preprocess_data(self, X, y=None):
        """数据预处理"""
        if isinstance(X, np.ndarray):
            X = {f'x{i+1}': X[:, i] for i in range(X.shape[1])}
        if y is not None and not isinstance(y, np.ndarray):
            y = np.array(y)
        return X, y
    
    def _select(self, node):
        """使用UCT策略选择节点"""
        while not node.is_leaf():
            node = max(node.children, key=lambda n: n.uct_value(self.exploration_constant))
        return node
    
    def _is_terminal(self, node):
        """检查是否达到终端条件"""
        return self._get_depth(node) >= self.max_depth
    
    def _get_depth(self, node):
        """获取节点深度"""
        depth = 0
        while node.parent:
            depth += 1
            node = node.parent
        return depth
    
    def _expand(self, node, X, y, target_data_embedding, target_expr_embedding):
        """扩展节点，生成子节点"""
        for _ in range(10):  # 每次扩展10个子节点
            new_eqtrees = self._mutate_eqtrees(node.eqtrees)
            child = EnhancedNode(new_eqtrees)
            
            # 计算子节点的嵌入
            try:
                expr_str = self._eqtrees_to_string(new_eqtrees)
                child.embedding = self.expression_encoder.encode(expr_str)
            except:
                child.embedding = np.zeros(self.expression_encoder.projection_dim)
            
            child.data_embedding = target_data_embedding
            
            # 评估子节点
            self._evaluate_node(child, X, y, target_data_embedding, target_expr_embedding)
            node.add_child(child)
            
        return random.choice(node.children)
    
    def _eqtrees_to_string(self, eqtrees):
        """将表达式树转换为字符串"""
        if not eqtrees:
            return "0"
        expr_parts = []
        for eqtree in eqtrees:
            try:
                expr_parts.append(str(eqtree))
            except:
                expr_parts.append("x1")
        return " + ".join(expr_parts)
    
    def _mutate_eqtrees(self, eqtrees):
        """变异表达式树"""
        mutation_type = random.choice(['add', 'replace', 'modify'])
        
        if mutation_type == 'add' and len(eqtrees) < self.max_vars:
            # 添加新的表达式树
            op = random.choice(self.binary_ops)
            operand1 = random.choice(eqtrees + self.variables + self.constants)
            operand2 = random.choice(eqtrees + self.variables + self.constants)
            new_eqtree = copy.deepcopy(op(operand1, operand2))
            return eqtrees + [new_eqtree]
            
        elif mutation_type == 'replace':
            # 替换一个表达式树
            idx = random.randint(0, len(eqtrees) - 1)
            new_eqtree = self._generate_random_expr()
            return eqtrees[:idx] + [new_eqtree] + eqtrees[idx+1:]
            
        else:  # modify
            # 修改一个表达式树
            idx = random.randint(0, len(eqtrees) - 1)
            eqtree_copy = copy.deepcopy(eqtrees[idx])
            
            if random.random() < 0.5:
                # 添加二元运算
                op = random.choice(self.binary_ops)
                operand = random.choice(self.variables + self.constants)
                if random.random() < 0.5:
                    new_eqtree = copy.deepcopy(op(eqtree_copy, operand))
                else:
                    new_eqtree = copy.deepcopy(op(operand, eqtree_copy))
            else:
                # 添加一元运算
                op = random.choice(self.unary_ops)
                new_eqtree = copy.deepcopy(op(eqtree_copy))
                
            return eqtrees[:idx] + [new_eqtree] + eqtrees[idx+1:]
    
    def _generate_random_expr(self):
        """生成随机表达式"""
        expr_type = random.choice(['variable', 'constant', 'unary', 'binary'])
        
        if expr_type == 'variable':
            return copy.deepcopy(random.choice(self.variables))
        elif expr_type == 'constant':
            return copy.deepcopy(random.choice(self.constants))
        elif expr_type == 'unary':
            op = random.choice(self.unary_ops)
            operand = self._generate_random_expr()
            return copy.deepcopy(op(operand))
        else:  # binary
            op = random.choice(self.binary_ops)
            operand1 = self._generate_random_expr()
            operand2 = self._generate_random_expr()
            return copy.deepcopy(op(operand1, operand2))
    
    def _simulate(self, node, X, y, target_data_embedding, target_expr_embedding):
        """模拟阶段"""
        current_eqtrees = node.eqtrees.copy()
        best_node = node
        
        for _ in range(self.simulation_count):
            depth = self._get_depth(node)
            temp_eqtrees = current_eqtrees.copy()
            
            while depth < self.max_depth:
                temp_eqtrees = self._mutate_eqtrees(temp_eqtrees)
                depth += 1
            
            # 评估表达式
            temp_node = EnhancedNode(temp_eqtrees)
            
            # 计算嵌入
            try:
                expr_str = self._eqtrees_to_string(temp_eqtrees)
                temp_node.embedding = self.expression_encoder.encode(expr_str)
            except:
                temp_node.embedding = np.zeros(self.expression_encoder.projection_dim)
            
            temp_node.data_embedding = target_data_embedding
            
            self._evaluate_node(temp_node, X, y, target_data_embedding, target_expr_embedding)
            
            if temp_node.r2 > best_node.r2:
                best_node = temp_node
                
        return best_node.reward, best_node
    
    def _evaluate_node(self, node, X, y, target_data_embedding, target_expr_embedding):
        """评估节点，计算复合奖励"""
        try:
            # 首先计算传统的R2分数
            self._evaluate_node_traditional(node, X, y)
            
            # 计算嵌入对齐奖励
            self._evaluate_embedding_alignment(
                node, target_data_embedding, target_expr_embedding
            )
            
            # 计算最终复合奖励
            node.reward = (
                self.reward_weights['structure_alignment'] * node.structure_alignment +
                self.reward_weights['data_alignment'] * node.data_alignment +
                self.reward_weights['accuracy'] * max(0, node.r2)
            )
            
            # 存储到经验回放池
            self.experience_buffer.append({
                'expression': node.phi,
                'r2': node.r2,
                'reward': node.reward,
                'embedding': node.embedding,
                'data_embedding': node.data_embedding
            })
            
        except Exception as e:
            print(f"Error evaluating node: {e}")
            node.r2 = -np.inf
            node.complexity = float('inf')
            node.reward = 0
            node.phi = nd.Number(0.0)
            node.structure_alignment = 0.0
            node.data_alignment = 0.0
    
    def _evaluate_node_traditional(self, node, X, y):
        """传统的节点评估（R2和复杂度）"""
        try:
            # 计算每个表达式树的输出
            Z = np.zeros((len(y), 1 + len(node.eqtrees)))
            Z[:, 0] = 1.0  # 常数项
            
            for i, eqtree in enumerate(node.eqtrees):
                try:
                    expr_copy = copy.deepcopy(eqtree)
                    Z[:, i+1] = self._evaluate_expression(expr_copy, X)
                except:
                    Z[:, i+1] = np.zeros(len(y))
            
            # 检查有效性
            if not np.isfinite(Z).all():
                node.r2 = -np.inf
                node.complexity = float('inf')
                node.phi = nd.Number(0.0)
                return
                
            # 避免溢出
            Z_max = np.max(np.abs(Z))
            if Z_max > 1e6:
                Z = Z / Z_max
                
            # 线性组合
            try:
                reg = 1e-6 * np.eye(Z.shape[1])
                A = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
                A = np.round(A, 6)
                y_pred = Z @ A
                node.r2 = R2_score(y, y_pred)
            except:
                A = np.linalg.pinv(Z) @ y
                A = np.round(A, 6)
                y_pred = Z @ A
                node.r2 = R2_score(y, y_pred)
            
            # 构建最终表达式
            node.phi = nd.Number(A[0]) if abs(A[0]) > 1e-6 else None
            for a, eqtree in zip(A[1:], node.eqtrees):
                if abs(a) < 1e-6:
                    continue
                elif abs(a - 1.0) < 1e-6:
                    if node.phi is None:
                        node.phi = eqtree
                    else:
                        node.phi = node.phi + eqtree
                elif abs(a + 1.0) < 1e-6:
                    if node.phi is None:
                        node.phi = -eqtree
                    else:
                        node.phi = node.phi - eqtree
                else:
                    if node.phi is None:
                        node.phi = nd.Number(a) * eqtree
                    else:
                        node.phi = node.phi + nd.Number(a) * eqtree
                        
            if node.phi is None:
                node.phi = nd.Number(0.0)
                
            node.complexity = len(str(node.phi))
            
        except:
            node.r2 = -np.inf
            node.complexity = float('inf')
            node.phi = nd.Number(0.0)
    
    def _evaluate_embedding_alignment(self, node, target_data_embedding, target_expr_embedding):
        """评估嵌入对齐"""
        try:
            # 数据对齐奖励
            if node.embedding is not None and target_data_embedding is not None:
                expr_embedding = torch.FloatTensor(node.embedding).unsqueeze(0).to(self.device)
                data_embedding = torch.FloatTensor(target_data_embedding).unsqueeze(0).to(self.device)
                
                # 计算余弦相似度
                data_alignment = F.cosine_similarity(expr_embedding, data_embedding).item()
                node.data_alignment = max(0, data_alignment)
            
            # 结构对齐奖励（如果提供了真实表达式）
            if target_expr_embedding is not None and node.embedding is not None:
                target_embedding = torch.FloatTensor(target_expr_embedding).unsqueeze(0).to(self.device)
                expr_embedding = torch.FloatTensor(node.embedding).unsqueeze(0).to(self.device)
                
                structure_alignment = F.cosine_similarity(expr_embedding, target_embedding).item()
                node.structure_alignment = max(0, structure_alignment)
            else:
                node.structure_alignment = 0.0
            
        except Exception as e:
            print(f"Error computing alignment: {e}")
            node.data_alignment = 0.0
            node.structure_alignment = 0.0
    
    def _evaluate_expression(self, expr, X):
        """评估单个表达式"""
        try:
            return expr.eval(X)
        except:
            return np.zeros(len(next(iter(X.values()))))
    
    def _backpropagate(self, node, reward):
        """回溯更新节点统计信息"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def get_experience_buffer(self):
        """获取经验回放池"""
        return list(self.experience_buffer)
    
    def clear_experience_buffer(self):
        """清空经验回放池"""
        self.experience_buffer.clear()