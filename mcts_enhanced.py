import numpy as np
import random
import copy
import warnings
import sys
import os
import torch

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from nd2py.utils import R2_score, RMSE_score
from reward_network import RewardNetwork, ExperienceReplayBuffer


class EnhancedNode:
    """增强的MCTS节点，支持奖励网络"""
    
    def __init__(self, eqtrees=None):
        # 支持多个表达式树
        self.eqtrees = eqtrees or [nd.Variable('x1')]
        self.parent = None
        self.children = []
        self.visits = 0
        self.value = 0
        self.reward = 0
        self.r2 = -np.inf
        self.complexity = 0
        self.phi = None  # 组合后的最终表达式
        self.reward_network_reward = 0  # 奖励网络的奖励
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        
    def uct_value(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * np.sqrt(np.log(self.parent.visits + 1) / (self.visits + 1))


class MCTSWithRewardNetwork:
    """使用奖励网络的增强MCTS符号回归"""
    
    def __init__(self, max_depth=10, max_iterations=1000, max_vars=5, 
                 eta=0.999, reward_network=None, experience_buffer=None,
                 alpha_hybrid=0.7):
        self.max_vars = max_vars
        self.binary_ops = [nd.Add, nd.Sub, nd.Mul, nd.Div]
        self.unary_ops = [nd.Sin, nd.Cos, nd.Sqrt, nd.Log, nd.Exp]
        self.constants = [nd.Number(1), nd.Number(2), nd.Number(0.5)]
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.best_expr = None
        self.best_r2 = -np.inf
        self.variables = []
        self.eta = eta  # 复杂度惩罚系数
        self.alpha_hybrid = alpha_hybrid  # 混合奖励中的性能权重
        
        # 奖励网络相关
        self.reward_network = reward_network
        self.experience_buffer = experience_buffer or ExperienceReplayBuffer()
        self.oracle_target = None  # 神谕目标向量
        self.training_data = None  # 训练数据 (X, y)
        
    def set_training_data(self, X, y):
        """设置训练数据"""
        if isinstance(X, np.ndarray):
            self.training_data = {
                'X': torch.FloatTensor(X),
                'y': torch.FloatTensor(y)
            }
        else:
            self.training_data = {
                'X': X,
                'y': y
            }
        
        # 初始化奖励网络的数据编码器维度
        if self.reward_network:
            input_dim = self.training_data['X'].shape[1] if len(self.training_data['X'].shape) > 1 else 1
            self.reward_network.set_data_encoder_dim(input_dim)
    
    def set_oracle_target(self, true_expression):
        """设置神谕目标向量"""
        if self.reward_network is None:
            raise ValueError("奖励网络未设置")
        
        self.reward_network.eval()
        with torch.no_grad():
            # 使用真实表达式作为目标
            oracle_embedding = self.reward_network.expression_embedding.encode_expressions([str(true_expression)])
            self.oracle_target = torch.FloatTensor(oracle_embedding)
        
        print(f"设置神谕目标: {true_expression}")
    
    def compute_hybrid_reward(self, expression_str, r2_score, complexity):
        """计算混合奖励"""
        # 性能分量
        r2_clipped = max(0, min(1, r2_score))  # 限制在[0,1]范围
        z_perf = 1.0 / (1.0 + np.exp(-10 * (r2_clipped - 0.5)))  # Sigmoid函数
        
        # 结构分量
        if self.oracle_target is not None and self.reward_network is not None:
            self.reward_network.eval()
            with torch.no_grad():
                expr_embedding = self.reward_network.expression_embedding.encode_expressions([expression_str])
                expr_tensor = torch.FloatTensor(expr_embedding)
                z_struct = torch.cosine_similarity(expr_tensor, self.oracle_target).item()
        else:
            # 如果没有神谕目标，使用复杂度惩罚作为结构奖励
            complexity_penalty = np.exp(-complexity / 100)  # 复杂度越高奖励越低
            z_struct = complexity_penalty
        
        # 混合奖励
        z_hybrid = self.alpha_hybrid * z_perf + (1 - self.alpha_hybrid) * z_struct
        
        return z_hybrid, z_perf, z_struct
    
    def evaluate_with_reward_network(self, expression_str):
        """使用奖励网络评估表达式"""
        if self.reward_network is None:
            raise ValueError("奖励网络未设置")
        
        if self.training_data is None:
            raise ValueError("训练数据未设置")
        
        # 准备数据
        X = self.training_data['X']
        
        # 使用奖励网络预测奖励
        reward_network_reward = self.reward_network.predict_reward(expression_str, X)
        
        return reward_network_reward
    
    def add_experience(self, expression_str, r2_score, complexity, hybrid_reward):
        """添加经验到回放池"""
        if self.training_data is None:
            return
        
        # 准备数据
        X = self.training_data['X']
        target_reward = torch.FloatTensor([[hybrid_reward]])
        
        # 存储经验
        self.experience_buffer.push(expression_str, X, target_reward)
    
    def fit(self, X, y, variables=None):
        """
        使用增强的蒙特卡洛树搜索进行符号回归
        
        参数:
        X: 输入数据
        y: 目标值
        variables: 可选，预定义的变量列表
        
        返回:
        最佳表达式
        """
        # 预处理数据
        X = self._preprocess(X)
        
        # 设置训练数据
        self.set_training_data(X, y)
        
        # 如果提供了变量列表，使用它们；否则从数据中提取
        if variables is not None:
            self.variables = variables
        else:
            self.variables = [nd.Variable(var) for var in X.keys()]
        
        # 初始化根节点，包含所有变量
        self.root = EnhancedNode(self.variables[:min(len(self.variables), self.max_vars)])
        self._evaluate_node(self.root, X, y)
        
        for i in range(self.max_iterations):
            # 1. 选择
            node = self._select(self.root)
            
            # 2. 扩展
            if not self._is_terminal(node) and node.visits > 0:
                node = self._expand(node, X, y)
            
            # 3. 模拟
            reward, best_node = self._simulate(node, X, y)
            
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
        """
        使用最佳表达式进行预测
        """
        if self.best_expr is None:
            raise ValueError("Model not fitted yet")
        X = self._preprocess(X)
        return self._evaluate_expression(self.best_expr, X)
    
    def get_score(self, X, y):
        """
        获取模型在给定数据上的R2和RMSE分数
        """
        y_pred = self.predict(X)
        r2 = R2_score(y, y_pred)
        rmse = RMSE_score(y, y_pred)
        return r2, rmse
    
    def _preprocess(self, X):
        """数据预处理函数"""
        if isinstance(X, np.ndarray):
            X = {f'x{i+1}': X[:, i] for i in range(X.shape[1])}
        return X
    
    def _select(self, node):
        """选择阶段：使用UCT策略选择最有希望的节点"""
        while not node.is_leaf():
            node = max(node.children, key=lambda n: n.uct_value())
        return node
    
    def _is_terminal(self, node):
        """检查节点是否达到最大深度"""
        return self._get_depth(node) >= self.max_depth
    
    def _get_depth(self, node):
        """获取节点深度"""
        depth = 0
        while node.parent:
            depth += 1
            node = node.parent
        return depth
    
    def _expand(self, node, X, y):
        """扩展阶段：生成子节点"""
        # 生成子节点
        for _ in range(10):  # 每次扩展10个子节点
            new_eqtrees = self._mutate_eqtrees(node.eqtrees)
            child = EnhancedNode(new_eqtrees)
            self._evaluate_node(child, X, y)
            node.add_child(child)
            
        return random.choice(node.children)
    
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
    
    def _simulate(self, node, X, y):
        """模拟阶段：从当前节点开始随机扩展到终端节点"""
        current_eqtrees = node.eqtrees.copy()
        best_node = node
        
        for _ in range(10):  # 10次模拟
            depth = self._get_depth(node)
            temp_eqtrees = current_eqtrees.copy()
            
            while depth < self.max_depth:
                temp_eqtrees = self._mutate_eqtrees(temp_eqtrees)
                depth += 1
            
            # 评估表达式
            temp_node = EnhancedNode(temp_eqtrees)
            self._evaluate_node(temp_node, X, y)
            
            if temp_node.r2 > best_node.r2:
                best_node = temp_node
                
        return best_node.reward, best_node
    
    def _evaluate_node(self, node, X, y):
        """评估节点：计算各种奖励分数"""
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
            
            # 检查Z的有效性
            if not np.isfinite(Z).all():
                node.r2 = -np.inf
                node.complexity = float('inf')
                node.reward = 0
                node.reward_network_reward = 0
                node.phi = nd.Number(0.0)
                return
                
            # 检查数值范围，避免溢出
            Z_max = np.max(np.abs(Z))
            if Z_max > 1e6:
                Z = Z / Z_max  # 归一化以避免溢出
                
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
            
            # 计算混合奖励
            if node.phi is not None:
                expression_str = str(node.phi)
                z_hybrid, z_perf, z_struct = self.compute_hybrid_reward(
                    expression_str, node.r2, node.complexity
                )
                node.reward = z_hybrid
                
                # 使用奖励网络评估
                if self.reward_network is not None:
                    try:
                        node.reward_network_reward = self.evaluate_with_reward_network(expression_str)
                        # 添加经验到回放池
                        self.add_experience(expression_str, node.r2, node.complexity, z_hybrid)
                    except:
                        node.reward_network_reward = 0
                else:
                    node.reward_network_reward = 0
            else:
                node.reward = 0
                node.reward_network_reward = 0
                
        except:
            node.r2 = -np.inf
            node.complexity = float('inf')
            node.reward = 0
            node.reward_network_reward = 0
            node.phi = nd.Number(0.0)
    
    def _evaluate_expression(self, expr, X):
        """评估单个表达式"""
        try:
            return expr.eval(X)
        except:
            return np.zeros(len(next(iter(X.values()))))
    
    def _backpropagate(self, node, reward):
        """回溯阶段：更新节点统计信息"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent


if __name__ == "__main__":
    # 测试代码
    print("测试增强MCTS...")
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    X = np.random.uniform(-5, 5, (n_samples, 2))
    y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
    
    # 创建增强MCTS
    mcts = MCTSWithRewardNetwork(max_depth=5, max_iterations=100)
    
    print("开始训练...")
    best_expr = mcts.fit(X, y)
    print(f"最佳表达式: {best_expr}")
    
    # 评估
    r2, rmse = mcts.get_score(X, y)
    print(f"R2: {r2:.4f}, RMSE: {rmse:.4f}")
    
    print("测试完成!")