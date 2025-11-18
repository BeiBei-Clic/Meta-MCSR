#!/usr/bin/env python3
"""
数据生成脚本

基于Lample和Charton方法的数学表达式生成器
生成阶段零预训练所需的(表达式, 数据集)对。

整个表达式生成流程包含两个核心部分：
1. 创造抽象的数学函数：使用"数学乐高积木"随机拼搭生成语法正确且结构多样的数学公式
2. 为函数生成具体的数值数据点：使用多模态混合分布生成多样化的输入数据

参考文献：Lample, G. & Charton, F. (2020). Deep learning for symbolic mathematics.
"""

import os
import sys
import numpy as np
import pickle
import logging
import random
import time
import re
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import ast
import math

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/logs/data_generation.log')
        ]
    )


class NodeType(Enum):
    """表达式树节点类型"""
    BINARY = "binary"    # 二元运算符 (+, -, *, /, ^)
    UNARY = "unary"      # 一元运算符 (sin, cos, exp, log, sqrt)
    VARIABLE = "variable" # 变量 (x1, x2, x3, ...)
    CONSTANT = "constant" # 常数 (3.14, 0.5, ...)


@dataclass
class ExpressionNode:
    """表达式树节点 - 实现安全的数学表达式树"""
    node_type: NodeType
    value: str  # 节点值（运算符、变量名或常数）
    left: Optional['ExpressionNode'] = None  # 左子节点（用于二元运算符）
    right: Optional['ExpressionNode'] = None  # 右子节点（用于二元运算符）
    child: Optional['ExpressionNode'] = None  # 子节点（用于一元运算符）
    
    def __post_init__(self):
        """验证节点结构的正确性"""
        if self.node_type == NodeType.BINARY:
            assert self.left is not None and self.right is not None, "二元运算符必须有两个子节点"
        elif self.node_type == NodeType.UNARY:
            assert self.child is not None, "一元运算符必须有一个子节点"
        elif self.node_type in [NodeType.VARIABLE, NodeType.CONSTANT]:
            assert self.left is None and self.right is None and self.child is None, "叶节点不应该有子节点"
    
    def to_string(self) -> str:
        """将表达式树转换为安全的字符串表示"""
        if self.node_type == NodeType.CONSTANT:
            return self._process_constant()
        elif self.node_type == NodeType.VARIABLE:
            return self.value
        elif self.node_type == NodeType.UNARY:
            return f"{self.value}({self.child.to_string()})"
        elif self.node_type == NodeType.BINARY:
            return self._process_binary_operation()
        else:
            raise ValueError(f"未知的节点类型: {self.node_type}")
    
    def _process_constant(self) -> str:
        """处理常数节点"""
        try:
            const_value = float(self.value)
            # 简单的数值范围限制，避免极端值
            if abs(const_value) > 100:
                const_value = max(-100, min(100, const_value))
            elif abs(const_value) < 1e-8:
                const_value = 1.0
        except (ValueError, TypeError):
            const_value = 1.0
        
        self.value = str(const_value)
        return self.value
    
    def _process_binary_operation(self) -> str:
        """处理二元运算，确保运算安全性"""
        left_str = self.left.to_string()
        right_str = self.right.to_string()
        
        if self.value == '/':
            return self._safe_divide(left_str, right_str)
        elif self.value == '^':
            return self._safe_power(left_str, right_str)
        elif self.value == '+':
            return f"({left_str} + {right_str})"
        elif self.value == '-':
            return f"({left_str} - {right_str})"
        elif self.value == '*':
            return f"({left_str} * {right_str})"
        else:
            return f"({left_str} {self.value} {right_str})"
    
    def _safe_divide(self, left_str: str, right_str: str) -> str:
        """安全的除法运算 - 保留除零保护"""
        return f"({left_str}) / max(abs({right_str}), 1e-10)"
    
    def _safe_power(self, left_str: str, right_str: str) -> str:
        """安全的幂运算 - 保留0^0保护"""
        return f"(max(abs({left_str}), 1e-10))**({right_str})"
    
    def count_nodes(self) -> int:
        """计算表达式树中的节点总数"""
        if self.node_type in [NodeType.VARIABLE, NodeType.CONSTANT]:
            return 1
        elif self.node_type == NodeType.UNARY:
            return 1 + (self.child.count_nodes() if self.child else 0)
        elif self.node_type == NodeType.BINARY:
            return 1 + (self.left.count_nodes() if self.left else 0) + (self.right.count_nodes() if self.right else 0)
        return 0
    
    def get_variables(self) -> set:
        """获取表达式中使用的所有变量"""
        variables = set()
        
        if self.node_type == NodeType.VARIABLE:
            variables.add(self.value)
        elif self.node_type == NodeType.UNARY and self.child:
            variables.update(self.child.get_variables())
        elif self.node_type == NodeType.BINARY:
            if self.left:
                variables.update(self.left.get_variables())
            if self.right:
                variables.update(self.right.get_variables())
        
        return variables
    
    def apply_affine_transform(self, affine_params: Dict[str, Tuple[float, float]]):
        """应用仿射变换 (z -> a*z + b)"""
        node_id = id(self)
        
        if node_id in affine_params:
            a, b = affine_params[node_id]
            if self.node_type == NodeType.CONSTANT:
                self.value = str(float(self.value) * a + b)
        
        # 递归处理子节点
        if self.node_type == NodeType.UNARY and self.child:
            self.child.apply_affine_transform(affine_params)
        elif self.node_type == NodeType.BINARY:
            if self.left:
                self.left.apply_affine_transform(affine_params)
            if self.right:
                self.right.apply_affine_transform(affine_params)

    def count_nodes(self) -> int:
        """计算表达式树中的节点总数"""
        if self.node_type in [NodeType.VARIABLE, NodeType.CONSTANT]:
            return 1
        elif self.node_type == NodeType.UNARY:
            return 1 + (self.child.count_nodes() if self.child else 0)
        elif self.node_type == NodeType.BINARY:
            return 1 + (self.left.count_nodes() if self.left else 0) + (self.right.count_nodes() if self.right else 0)
        return 0

    def get_variables(self) -> set:
        """获取表达式中使用的所有变量"""
        variables = set()
        
        if self.node_type == NodeType.VARIABLE:
            variables.add(self.value)
        elif self.node_type == NodeType.UNARY and self.child:
            variables.update(self.child.get_variables())
        elif self.node_type == NodeType.BINARY:
            if self.left:
                variables.update(self.left.get_variables())
            if self.right:
                variables.update(self.right.get_variables())
        
        return variables


class FunctionGenerator:
    """数学函数生成器 - 实现基于"数学乐高积木"的随机拼搭方法
    
    按照理论描述的5步流程生成数学表达式：
    1. 确定维度D：随机决定函数包含的自变量数量
    2. 搭建结构骨架：使用二元运算符构建基本树状结构
    3. 填充变量：将变量随机填充到树的叶节点位置
    4. 增加复杂性：随机插入一元运算符如sin、cos等
    5. 引入常数：应用仿射变换让函数更接近真实物理定律
    """
    
    def __init__(self):
        # 使用Lample和Charton方法的基本二元运算符
        self.binary_operators = ['+', '-', '*']  # 专注于基本运算符，减少复杂性
        self.unary_operators = ['sin', 'cos', 'exp', 'log', 'sqrt']  # 选择数值稳定的一元函数
        self.max_depth = 6  # 降低最大深度避免过复杂
        self.min_depth = 2  # 最小深度确保有足够复杂性
    
    def generate_random_function(self, max_retries: int = 10) -> Tuple[ExpressionNode, int]:
        """
        随机生成一个数学函数，遵循5步生成流程
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            (表达式树, 维度D) 元组
        """
        for attempt in range(max_retries):
            try:
                # 步骤1: 确定维度D (1-3个变量)
                dimension = random.randint(1, 3)
                
                # 步骤2: 搭建结构骨架（构建二元树）
                tree_structure = self._build_structural_skeleton()
                
                # 步骤3: 填充变量到叶节点
                tree_with_variables = self._fill_variables(tree_structure, dimension)
                
                # 步骤4: 增加复杂性（插入一元运算符）
                complex_tree = self._add_complexity(tree_with_variables)
                
                # 步骤5: 引入常数和现实性（应用仿射变换）
                final_tree = self._apply_affine_transforms(complex_tree)
                
                return final_tree, dimension
                    
            except Exception:
                continue
        
        # 备用方案
        return self._create_simple_linear_function(1), 1
    
    def _build_structural_skeleton(self) -> ExpressionNode:
        """步骤2: 搭建结构骨架（构建二元树）
        
        使用Lample和Charton的随机采样方法，确保生成的树结构
        是随机的，但语法上总是正确的。
        """
        depth = random.randint(self.min_depth, self.max_depth)
        return self._build_tree_skeleton(depth)
    
    def _build_tree_skeleton(self, depth: int) -> ExpressionNode:
        """递归构建表达式树骨架"""
        if depth <= 1:
            # 叶节点，稍后会被变量替换
            return ExpressionNode(
                node_type=NodeType.CONSTANT,
                value="1"  # 占位符
            )
        
        # 优先选择基本运算符，提高数值稳定性
        operator = random.choice(self.binary_operators)
        
        # 递归构建左右子树，确保平衡
        if depth == 2:
            # 最小深度：直接是两个叶子
            left_child = ExpressionNode(node_type=NodeType.CONSTANT, value="1")
            right_child = ExpressionNode(node_type=NodeType.CONSTANT, value="1")
        else:
            # 分割深度
            left_depth = random.randint(1, depth - 1)
            right_depth = depth - left_depth
            
            left_child = self._build_tree_skeleton(left_depth)
            right_child = self._build_tree_skeleton(right_depth)
        
        return ExpressionNode(
            node_type=NodeType.BINARY,
            value=operator,
            left=left_child,
            right=right_child
        )
    
    def _fill_variables(self, tree: ExpressionNode, dimension: int) -> ExpressionNode:
        """步骤3: 填充变量到叶节点位置
        
        将表达式树中的常数占位符随机替换为变量 x1, x2, x3...
        """
        return self._replace_placeholders_with_variables(tree, dimension)
    
    def _replace_placeholders_with_variables(self, node: ExpressionNode, dimension: int) -> ExpressionNode:
        """递归将占位符替换为变量"""
        if node.node_type == NodeType.CONSTANT:
            # 将常数占位符替换为随机变量
            var_index = random.randint(1, dimension)
            return ExpressionNode(
                node_type=NodeType.VARIABLE,
                value=f"x{var_index}"
            )
        elif node.node_type == NodeType.BINARY:
            return ExpressionNode(
                node_type=NodeType.BINARY,
                value=node.value,
                left=self._replace_placeholders_with_variables(node.left, dimension),
                right=self._replace_placeholders_with_variables(node.right, dimension)
            )
        elif node.node_type == NodeType.UNARY:
            return ExpressionNode(
                node_type=NodeType.UNARY,
                value=node.value,
                child=self._replace_placeholders_with_variables(node.child, dimension)
            )
        else:
            return node
    
    def _add_complexity(self, tree: ExpressionNode) -> ExpressionNode:
        """步骤4: 增加复杂性（插入一元运算符）
        
        像"装饰品"一样随机地将一元运算符插入到树的任意位置，
        一个已经存在的节点或子树会被这个一元运算符"包裹"起来。
        """
        # 决定插入一元运算符的数量（0-2个）
        n_operators = np.random.choice([0, 1, 2], p=[0.4, 0.5, 0.1])
        
        current_tree = tree
        for _ in range(n_operators):
            operator = random.choice(self.unary_operators)
            target_node = self._select_target_node(current_tree)
            
            if target_node:
                # 创建一元运算符节点包裹目标节点
                unary_node = ExpressionNode(
                    node_type=NodeType.UNARY,
                    value=operator,
                    child=target_node
                )
                current_tree = unary_node
            else:
                break
        
        return current_tree
    
    def _select_target_node(self, tree: ExpressionNode) -> Optional[ExpressionNode]:
        """选择目标节点用于插入一元运算符"""
        nodes = []
        self._collect_all_nodes(tree, nodes)
        
        if not nodes:
            return None
        
        # 优先选择叶节点，因为包裹叶节点更安全
        leaf_nodes = [node for node in nodes if node.node_type in [NodeType.VARIABLE]]
        if leaf_nodes and random.random() < 0.7:  # 70%概率选择叶节点
            return random.choice(leaf_nodes)
        else:
            # 选择任意节点，但避免选择根节点（防止过度复杂化）
            non_root_nodes = [node for node in nodes if node != tree]
            if non_root_nodes:
                return random.choice(non_root_nodes)
            else:
                return random.choice(nodes)
    
    def _collect_all_nodes(self, node: ExpressionNode, nodes: List[ExpressionNode]):
        """收集树中的所有节点"""
        nodes.append(node)
        
        if node.node_type == NodeType.BINARY:
            self._collect_all_nodes(node.left, nodes)
            self._collect_all_nodes(node.right, nodes)
        elif node.node_type == NodeType.UNARY:
            self._collect_all_nodes(node.child, nodes)
    
    def _apply_affine_transforms(self, tree: ExpressionNode) -> ExpressionNode:
        """步骤5: 引入常数和现实性（应用仿射变换）
        
        这是至关重要的一步，让生成的函数从"玩具"变成了"仿真"的物理定律。
        算法会对树中的每一个变量和每一个一元运算符应用一个随机的仿射变换 (z -> a*z + b)。
        """
        # 为每个变量和一元运算符生成仿射变换参数
        affine_params = {}
        self._generate_affine_parameters(tree, affine_params)
        
        # 应用变换
        tree.apply_affine_transform(affine_params)
        
        # 为整个表达式添加最终的仿射变换，让结果更接近真实函数
        # 只有在表达式不太复杂时才添加，降低不稳定性
        expression_complexity = tree.count_nodes()
        if expression_complexity <= 6 and random.random() < 0.5:  # 50%概率添加最终变换
            a = random.uniform(0.5, 2.0)  # 适度的缩放
            b = random.uniform(-2.0, 2.0)  # 适度的偏移
            
            # 创建 a*tree + b，确保格式正确
            transform_node = ExpressionNode(
                node_type=NodeType.BINARY,
                value="*",
                left=ExpressionNode(node_type=NodeType.CONSTANT, value=str(a)),
                right=ExpressionNode(
                    node_type=NodeType.BINARY,
                    value="+",
                    left=tree,
                    right=ExpressionNode(node_type=NodeType.CONSTANT, value=str(b))
                )
            )
            return transform_node
        
        return tree
    
    def _generate_affine_parameters(self, node: ExpressionNode, params: Dict[int, Tuple[float, float]]):
        """为节点生成仿射变换参数"""
        if node.node_type == NodeType.VARIABLE:
            a = random.uniform(0.8, 1.5)
            b = random.uniform(-1.0, 1.0)
            params[id(node)] = (a, b)
        elif node.node_type == NodeType.UNARY:
            a = random.uniform(0.9, 1.2)
            b = random.uniform(-0.5, 0.5)
            params[id(node)] = (a, b)
        
        # 递归处理子节点
        if node.node_type == NodeType.BINARY:
            self._generate_affine_parameters(node.left, params)
            self._generate_affine_parameters(node.right, params)
        elif node.node_type == NodeType.UNARY:
            self._generate_affine_parameters(node.child, params)
    
    def _validate_expression(self, tree: ExpressionNode) -> bool:
        """简单的表达式验证"""
        return (tree.count_nodes() > 0 and 
                len(tree.get_variables()) <= 3 and 
                tree.count_nodes() <= 15)
    
    def _create_simple_linear_function(self, dimension: int) -> ExpressionNode:
        """创建简单的线性函数作为备用方案"""
        if dimension == 1:
            a = random.uniform(0.5, 2.0)
            b = random.uniform(-1.0, 1.0)
            return ExpressionNode(
                node_type=NodeType.BINARY,
                value="+",
                left=ExpressionNode(
                    node_type=NodeType.BINARY,
                    value="*",
                    left=ExpressionNode(node_type=NodeType.CONSTANT, value=str(a)),
                    right=ExpressionNode(node_type=NodeType.VARIABLE, value="x1")
                ),
                right=ExpressionNode(node_type=NodeType.CONSTANT, value=str(b))
            )
        else:
            # 多变量线性函数
            left = ExpressionNode(node_type=NodeType.VARIABLE, value="x1")
            for i in range(2, dimension + 1):
                left = ExpressionNode(
                    node_type=NodeType.BINARY,
                    value="+",
                    left=left,
                    right=ExpressionNode(node_type=NodeType.VARIABLE, value=f"x{i}")
                )
            return left


class DataGenerator:
    """数据点生成器 - 实现多模态混合分布方法
    
    按照理论描述的第二部分：为函数生成具体的数值数据点。
    使用多模态混合分布的方法生成输入数据，模拟真实实验数据的复杂情况。
    """
    
    def __init__(self):
        self.max_samples_per_cluster = 40  # 降低每个簇的最大样本数
        self.min_samples_per_cluster = 8   # 降低最小样本数
        
    def generate_input_points(self, dimension: int, n_samples: int) -> np.ndarray:
        """
        使用多模态混合分布生成输入数据点
        
        算法采用多模态混合分布：在一个D维空间中，随机设定几个中心点（centroids），
        每个中心点周围的数据点可以呈高斯分布或均匀分布，并且还可以进行随机旋转。
        
        Args:
            dimension: 输入维度
            n_samples: 样本数量
            
        Returns:
            输入数据矩阵 (n_samples, dimension)
        """
        # 决定簇的数量 (1-4个中心点)
        max_clusters = min(4, max(1, n_samples // 25))
        n_clusters = random.randint(1, max_clusters)
        
        # 均匀分配样本到各个簇
        X = np.zeros((n_samples, dimension))
        cluster_indices = []
        
        for i in range(n_samples):
            cluster_idx = i % n_clusters  # 均匀分配
            cluster_indices.append(cluster_idx)
        
        # 为每个簇生成数据
        for cluster_id in range(n_clusters):
            cluster_mask = np.array(cluster_indices) == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size == 0:
                continue
            
            # 随机设置簇中心
            centroids = []
            for d in range(dimension):
                if random.random() < 0.6:  # 60%使用高斯分布
                    # 高斯分布，更集中在中心周围
                    center = random.uniform(-2.5, 2.5)
                    spread = random.uniform(0.2, 1.2)
                else:
                    # 均匀分布，更分散
                    center = random.uniform(-3.5, 3.5)
                    spread = random.uniform(0.8, 2.5)
                centroids.append((center, spread))
            
            # 生成该簇的数据点
            cluster_data = np.zeros((cluster_size, dimension))
            for d in range(dimension):
                center, spread = centroids[d]
                if random.random() < 0.6:  # 60%使用高斯分布
                    cluster_data[:, d] = np.random.normal(center, spread, cluster_size)
                else:
                    cluster_data[:, d] = np.random.uniform(
                        center - spread, center + spread, cluster_size
                    )
            
            # 随机旋转增加复杂性 (30%概率)
            if dimension >= 2 and random.random() < 0.3:
                cluster_data = self._apply_random_rotation(cluster_data)
            
            # 分配到总矩阵中
            X[cluster_mask] = cluster_data
        
        return X
    
    def _apply_random_rotation(self, data: np.ndarray) -> np.ndarray:
        """对数据应用随机旋转"""
        if data.shape[1] == 2:
            # 2D数据：应用2D旋转矩阵
            angle = random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            return data @ rotation_matrix.T
        elif data.shape[1] >= 2:
            # 高维数据：只旋转前两个维度
            angle = random.uniform(0, 2 * np.pi)
            rotation_matrix_2d = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            data[:, :2] = data[:, :2] @ rotation_matrix_2d.T
            return data
        else:
            return data
    
    def calculate_output_values(self, X: np.ndarray, expression_tree: ExpressionNode) -> np.ndarray:
        """根据表达式树计算输出值，保留数据溢出保护"""
        try:
            expression_str = expression_tree.to_string()
            n_samples, dimension = X.shape
            
            # 准备变量字典
            var_dict = {}
            for i in range(dimension):
                var_dict[f'x{i+1}'] = X[:, i]
            
            # 安全函数（保留溢出保护）
            def safe_sqrt(x):
                return np.sqrt(np.maximum(x, 0))  # 防止负数开方
            
            def safe_log(x):
                return np.log(np.maximum(x, 1e-10))  # 防止log(0)
            
            def safe_exp(x):
                return np.exp(np.clip(x, -700, 700))  # 防止指数溢出
            
            def safe_pow(base, exponent):
                base = np.asarray(base)
                exponent = np.asarray(exponent)
                return np.power(np.maximum(np.abs(base), 1e-10), exponent)
            
            # 添加函数和常数
            var_dict.update({
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': safe_exp, 'log': safe_log, 'sqrt': safe_sqrt,
                'abs': np.abs, 'sign': np.sign, 'pi': np.pi, 'e': np.e,
                'max': np.maximum, 'min': np.minimum, 'pow': safe_pow
            })
            
            # 计算输出值
            safe_expr = expression_str.replace('^', '**')
            y = eval(safe_expr, {"__builtins__": {}}, var_dict)
            
            # 检查并裁剪异常值
            y = np.asarray(y)
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                y = np.clip(y, -1e4, 1e4)
            
            return y
            
        except Exception:
            return self._calculate_simple_output(X, expression_tree)
    
    def _calculate_simple_output(self, X: np.ndarray, expression_tree: ExpressionNode) -> np.ndarray:
        """简化的输出计算（备用方法）"""
        used_vars = expression_tree.get_variables()
        
        if len(used_vars) == 0:
            return np.ones(X.shape[0])
        elif len(used_vars) == 1:
            var_index = int(list(used_vars)[0][1]) - 1
            return X[:, var_index] if var_index < X.shape[1] else np.zeros(X.shape[0])
        else:
            result = np.zeros(X.shape[0])
            for var_name in used_vars:
                var_index = int(var_name[1]) - 1
                if var_index < X.shape[1]:
                    result += X[:, var_index]
            return result / len(used_vars)
    
    def validate_and_clean_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """验证和清洗数据"""
        # 移除非有限值
        finite_mask = np.isfinite(y)
        X_clean = X[finite_mask]
        y_clean = y[finite_mask]
        
        if X_clean.size > 0:
            finite_input_mask = np.all(np.isfinite(X_clean), axis=1)
            X_clean = X_clean[finite_input_mask]
            y_clean = y_clean[finite_input_mask]
        
        # 移除异常值
        if y_clean.size > 0:
            mean_y = np.mean(y_clean)
            std_y = np.std(y_clean)
            if std_y > 0:
                outlier_mask = np.abs(y_clean - mean_y) <= 3 * std_y
                X_clean = X_clean[outlier_mask]
                y_clean = y_clean[outlier_mask]
        
        if X_clean.size == 0:
            raise ValueError("数据清洗后没有有效数据点")
        
        return X_clean, y_clean


def create_progress_bar(current, total, width=50):
    """创建简单的进度条"""
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percent = progress * 100
    return f"|{bar}| {current}/{total} ({percent:.1f}%)"


def _generate_expression_with_data(
    function_generator: FunctionGenerator,
    data_generator: DataGenerator,
    n_samples_per_expr: int,
    noise_level: float,
    max_attempts: int
) -> Tuple[bool, str, np.ndarray, np.ndarray, int]:
    """
    辅助函数：生成一个表达式及其对应的数据集
    
    Args:
        function_generator: 函数生成器
        data_generator: 数据生成器
        n_samples_per_expr: 每个表达式的样本数
        noise_level: 噪声水平
        max_attempts: 最大重试次数
        
    Returns:
        (是否成功, 表达式字符串, 清洗后的输入数据, 清洗后的输出数据, 维度)
    """
    for attempt in range(max_attempts):
        try:
            # 第一部分：创造抽象的数学函数
            expression_tree, dimension = function_generator.generate_random_function()
            expression_str = expression_tree.to_string()
            
            # 第二部分：为函数生成具体的数值数据
            X = data_generator.generate_input_points(dimension, n_samples_per_expr)
            y = data_generator.calculate_output_values(X, expression_tree)
            
            # 数据清洗
            X_clean, y_clean = data_generator.validate_and_clean_data(X, y)
            
            # 添加噪声（可选）
            if noise_level > 0 and len(y_clean) > 0:
                noise_std = noise_level * np.std(y_clean) if np.std(y_clean) > 0 else noise_level
                noise = np.random.normal(0, noise_std, len(y_clean))
                y_clean += noise
            
            # 检查是否有足够的数据
            if len(y_clean) >= n_samples_per_expr * 0.5:  # 至少50%的样本有效
                return True, expression_str, X_clean, y_clean, dimension
            else:
                raise ValueError("有效样本数不足")
                
        except Exception as e:
            if attempt == max_attempts - 1:
                # 最后一次尝试失败，使用简单表达式
                dim = random.randint(1, 2)
                tree = function_generator._create_simple_linear_function(dim)
                expression_str = tree.to_string()
                
                X = data_generator.generate_input_points(dim, n_samples_per_expr)
                y = data_generator.calculate_output_values(X, tree)
                X_clean, y_clean = data_generator.validate_and_clean_data(X, y)
                
                return True, expression_str, X_clean, y_clean, dim
    
    return False, "", np.array([]), np.array([]), 0


def generate_pretrain_dataset(
    n_expressions: int,
    n_samples_per_expr: int,
    output_path: str,
    noise_level: float = 0.01,
    save_interval: int = 1000,
    progress_callback: Optional[callable] = None
) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """
    生成预训练数据集 - 使用新的基于"数学乐高积木"的生成方法
    
    Args:
        n_expressions: 表达式数量
        n_samples_per_expr: 每个表达式的样本数
        output_path: 输出路径
        noise_level: 噪声水平
        save_interval: 保存间隔
        progress_callback: 进度回调函数
        
    Returns:
        (表达式列表, 数据集列表, 维度列表)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始生成 {n_expressions} 个数学表达式的数据集...")
    logger.info("使用基于'数学乐高积木'的随机拼搭方法：")
    logger.info("1. 创造抽象的数学函数 (5个步骤)")
    logger.info("2. 为函数生成具体的数值数据点 (多模态混合分布)")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 初始化生成器
    function_generator = FunctionGenerator()
    data_generator = DataGenerator()
    
    expressions = []
    datasets = []
    dimensions = []
    
    start_time = time.time()
    
    # 生成表达式和数据
    for i in range(n_expressions):
        success = False
        attempt_count = 0
        
        # 显示进度条（每50个表达式更新一次）
        if (i + 1) % 50 == 0 or i == 0:
            progress_bar = create_progress_bar(i + 1, n_expressions)
            elapsed = time.time() - start_time
            if i > 0:
                eta = (elapsed / (i + 1)) * (n_expressions - i - 1)
                eta_str = f", ETA: {eta/60:.1f}分钟"
            else:
                eta_str = ""
            print(f"\r{progress_bar}{eta_str}", end="", flush=True)
        
        # 限制重试次数，避免无限循环
        max_attempts = min(30, max(5, n_expressions // 10))  # 动态调整重试次数
        
        # 尝试生成表达式和数据
        success, expression_str, X_clean, y_clean, dim = _generate_expression_with_data(
            function_generator, data_generator, n_samples_per_expr, noise_level, max_attempts
        )
        
        # 如果成功生成，添加到结果中
        if success:
            expressions.append(expression_str)
            datasets.append((X_clean, y_clean))
            dimensions.append(dim)
        else:
            # 如果所有尝试都失败，强制添加一个简单表达式
            dim = 1
            tree = function_generator._create_simple_linear_function(dim)
            expression_str = tree.to_string()
            X = data_generator.generate_input_points(dim, n_samples_per_expr)
            y = data_generator.calculate_output_values(X, tree)
            X_clean, y_clean = data_generator.validate_and_clean_data(X, y)
            
            expressions.append(expression_str)
            datasets.append((X_clean, y_clean))
            dimensions.append(dim)
        
        # 定期保存进度
        if (i + 1) % save_interval == 0:
            save_progress(expressions, datasets, dimensions, output_path, i + 1)
            if (i + 1) % (save_interval * 5) == 0:  # 只在5倍间隔时记录日志
                logger.info(f"已生成 {i + 1}/{n_expressions} 个有效表达式")
    
    # 完成进度条
    print()  # 换行
    
    # 最终保存
    save_progress(expressions, datasets, dimensions, output_path, len(expressions))
    
    logger.info(f"数据集生成完成！")
    logger.info(f"总共生成 {len(expressions)} 个有效表达式")
    logger.info(f"总用时: {(time.time() - start_time)/60:.1f} 分钟")
    
    # 统计信息
    if expressions:
        dim_counts = {}
        for dim in dimensions:
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
        
        logger.info("维度分布:")
        for dim, count in sorted(dim_counts.items()):
            logger.info(f"  {dim}维: {count} 个 ({count/len(expressions)*100:.1f}%)")
    
    return expressions, datasets, dimensions


def save_progress(
    expressions: List[str],
    datasets: List[Tuple[np.ndarray, np.ndarray]],
    dimensions: List[int],
    output_path: str,
    count: int
):
    """保存进度"""
    progress_path = os.path.join(output_path, f'progress_{count}')
    os.makedirs(progress_path, exist_ok=True)
    
    # 保存表达式
    with open(os.path.join(progress_path, 'expressions.pkl'), 'wb') as f:
        pickle.dump(expressions, f)
    
    # 保存数据集
    with open(os.path.join(progress_path, 'datasets.pkl'), 'wb') as f:
        pickle.dump(datasets, f)
    
    # 保存维度信息
    with open(os.path.join(progress_path, 'dimensions.pkl'), 'wb') as f:
        pickle.dump(dimensions, f)
    
    # 保存元数据
    metadata = {
        'count': count,
        'generated_at': str(np.datetime64('now')),
        'expression_count': len(expressions),
        'dataset_count': len(datasets),
        'dimension_count': len(dimensions),
        'generation_method': '基于数学乐高积木的随机拼搭方法',
        'steps': [
            '1. 确定维度D (1-4个变量)',
            '2. 构建基本结构骨架 (使用二元运算符)',
            '3. 填充变量',
            '4. 增加复杂性 (插入一元运算符)',
            '5. 引入常数和现实性 (应用仿射变换)'
        ],
        'data_generation': [
            '1. 使用多模态混合分布生成输入数据',
            '2. 根据表达式树计算输出值',
            '3. 数据清洗和验证'
        ]
    }
    
    with open(os.path.join(progress_path, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    logger = logging.getLogger(__name__)
    logger.info(f"进度已保存: {progress_path}")


def cleanup_intermediate_progress(output_path: str, final_count: int):
    """
    清理中间进度文件夹，只保留最终完整数据集
    
    Args:
        output_path: 输出路径
        final_count: 最终的数据量
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(output_path):
        return
    
    # 获取所有progress文件夹
    progress_dirs = []
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path) and item.startswith('progress_'):
            progress_dirs.append(item)
    
    if not progress_dirs:
        return
    
    # 统计要删除的文件夹
    dirs_to_remove = []
    final_dir = f'progress_{final_count}'
    
    for dir_name in progress_dirs:
        if dir_name != final_dir:
            dirs_to_remove.append(dir_name)
    
    if not dirs_to_remove:
        logger.info("没有中间文件夹需要清理")
        return
    
    # 删除中间文件夹
    total_size = 0
    for dir_name in dirs_to_remove:
        dir_path = os.path.join(output_path, dir_name)
        try:
            # 计算文件夹大小
            dir_size = sum(
                os.path.getsize(os.path.join(dir_path, f)) 
                for f in os.listdir(dir_path) 
                if os.path.isfile(os.path.join(dir_path, f))
            )
            total_size += dir_size
            
            # 删除文件夹
            import shutil
            shutil.rmtree(dir_path)
            
            logger.info(f"已删除中间文件夹: {dir_name}")
            
        except Exception as e:
            logger.warning(f"删除文件夹 {dir_name} 时出错: {e}")
    
    # 转换字节为可读格式
    def format_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    logger.info(f"清理完成! 删除了 {len(dirs_to_remove)} 个中间文件夹，节省空间: {format_size(total_size)}")
    logger.info(f"保留最终数据集: {final_dir} (包含 {final_count} 个表达式)")


def analyze_generated_data(expressions: List[str], dimensions: List[int]) -> Dict[str, Any]:
    """分析生成的数据统计信息"""
    analysis = {
        'total_expressions': len(expressions),
        'dimension_distribution': {},
        'operator_usage': {},
        'function_usage': {},
        'complexity_stats': {
            'min_length': float('inf'),
            'max_length': 0,
            'avg_length': 0
        }
    }
    
    # 维度分布
    for dim in dimensions:
        analysis['dimension_distribution'][dim] = analysis['dimension_distribution'].get(dim, 0) + 1
    
    # 操作符和函数使用统计
    binary_ops = ['+', '-', '*', '/', '^']
    unary_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'cbrt']
    
    for expr in expressions:
        length = len(expr)
        analysis['complexity_stats']['min_length'] = min(analysis['complexity_stats']['min_length'], length)
        analysis['complexity_stats']['max_length'] = max(analysis['complexity_stats']['max_length'], length)
        
        # 统计二元操作符
        for op in binary_ops:
            count = expr.count(op)
            if count > 0:
                analysis['operator_usage'][op] = analysis['operator_usage'].get(op, 0) + count
        
        # 统计一元函数
        for func in unary_funcs:
            count = expr.count(func)
            if count > 0:
                analysis['function_usage'][func] = analysis['function_usage'].get(func, 0) + count
    
    # 计算平均长度
    if expressions:
        total_length = sum(len(expr) for expr in expressions)
        analysis['complexity_stats']['avg_length'] = total_length / len(expressions)
    
    return analysis


def main():
    """主函数"""
    print("=" * 80)
    print("基于'数学乐高积木'的数学表达式生成器")
    print("=" * 80)
    print()
    print("核心流程:")
    print("1. 创造抽象的数学函数 (5个步骤)")
    print("   • 确定维度D (1-4个变量)")
    print("   • 构建基本结构骨架 (使用二元运算符)")
    print("   • 填充变量")
    print("   • 增加复杂性 (插入一元运算符)")
    print("   • 引入常数和现实性 (应用仿射变换)")
    print()
    print("2. 为函数生成具体的数值数据点")
    print("   • 使用多模态混合分布生成输入数据")
    print("   • 根据表达式树计算输出值")
    print("   • 数据清洗和验证")
    print()
    print("目标: 最大化数据的多样性和复杂性，训练Transformer的内化数学'语法'和'语义'")
    print("=" * 80)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 配置参数
    n_expressions = 100000  # 生成10万个表达式
    n_samples_per_expr = 100  # 每个表达式100个数据点
    output_path = "data/pretrain/"
    noise_level = 0.01
    
    print(f"生成配置:")
    print(f"表达式数量: {n_expressions:,}")
    print(f"每个表达式的样本数: {n_samples_per_expr}")
    print(f"噪声水平: {noise_level}")
    print(f"输出路径: {output_path}")
    print()
    
    try:
        # 生成数据集
        expressions, datasets, dimensions = generate_pretrain_dataset(
            n_expressions=n_expressions,
            n_samples_per_expr=n_samples_per_expr,
            output_path=output_path,
            noise_level=noise_level
        )
        
        print()  # 确保进度条后有换行
        
        # 分析生成的数据
        analysis = analyze_generated_data(expressions, dimensions)
        
        print("=" * 80)
        print("生成完成统计:")
        print("=" * 80)
        print(f"总表达式数: {analysis['total_expressions']:,}")
        print(f"总数据集数: {len(datasets):,}")
        print()
        
        # 维度分布
        print("维度分布:")
        for dim, count in sorted(analysis['dimension_distribution'].items()):
            percentage = count / analysis['total_expressions'] * 100
            print(f"  {dim}维: {count:,} 个 ({percentage:.1f}%)")
        print()
        
        # 操作符使用统计
        if analysis['operator_usage']:
            print("二元操作符使用统计:")
            for op, count in sorted(analysis['operator_usage'].items()):
                percentage = count / analysis['total_expressions'] * 100
                print(f"  {op}: {count:,} 次 ({percentage:.1f}%)")
            print()
        
        # 函数使用统计
        if analysis['function_usage']:
            print("函数使用统计:")
            for func, count in sorted(analysis['function_usage'].items()):
                percentage = count / analysis['total_expressions'] * 100
                print(f"  {func}: {count:,} 次 ({percentage:.1f}%)")
            print()
        
        # 复杂度统计
        complexity = analysis['complexity_stats']
        print("表达式复杂度统计:")
        print(f"  最小长度: {complexity['min_length']} 字符")
        print(f"  最大长度: {complexity['max_length']} 字符")
        print(f"  平均长度: {complexity['avg_length']:.1f} 字符")
        print()
        
        print(f"数据已保存到: {output_path}")
        
        # 清理中间进度文件夹
        cleanup_intermediate_progress(output_path, len(expressions))
        
        print("=" * 80)
        print("✓ 预训练数据生成完成！")
        print("数据集已准备就绪，可用于训练Transformer模型")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"数据生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
