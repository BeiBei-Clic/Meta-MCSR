import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import warnings
import sys
import os
from collections import defaultdict

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from nd2py.utils import R2_score, RMSE_score
from reward_network import RewardNetwork, ExperienceReplayBuffer, RewardNetworkTrainer


class SelfLearningNode:
    """自学习MCTS节点"""
    
    def __init__(self, eqtrees=None):
        self.eqtrees = eqtrees or [nd.Variable('x1')]
        self.parent = None
        self.children = []
        self.visits = 0
        self.value = 0
        self.reward = 0
        self.r2 = -np.inf
        self.complexity = 0
        self.phi = None  # 组合后的最终表达式
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        
    def uct_value(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * np.sqrt(np.log(self.parent.visits + 1) / (self.visits + 1))


class SelfLearningMCTS:
    """自学习MCTS系统 - 实现完整的Epoch循环和神谕目标机制"""
    
    def __init__(self, max_depth=10, max_iterations=1000, max_vars=5, 
                 eta=0.999, alpha_hybrid=0.7, experience_buffer_size=10000):
        self.max_vars = max_vars
        self.binary_ops = [nd.Add, nd.Sub, nd.Mul, nd.Div]
        self.unary_ops = [nd.Sin, nd.Cos, nd.Sqrt, nd.Log, nd.Exp]
        self.constants = [nd.Number(1), nd.Number(2), nd.Number(0.5)]
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.best_expr = None
        self.best_r2 = -np.inf
        self.variables = []
        self.eta = eta
        self.alpha_hybrid = alpha_hybrid
        
        # 核心组件
        self.reward_network = None
        self.reward_trainer = None
        self.experience_buffer = ExperienceReplayBuffer(max_size=experience_buffer_size)
        
        # 神谕目标
        self.oracle_target = None
        self.true_expression = None
        
        # 训练数据
        self.training_data = None
        self.data_stats = None
        
        # 统计信息
        self.epoch_stats = defaultdict(list)
        
    def initialize_reward_network(self, expr_encoder_path, data_dim=None):
        """初始化奖励网络"""
        print("初始化奖励网络...")
        
        self.reward_network = RewardNetwork(
            expr_encoder_path=expr_encoder_path,
            d_model=256,
            data_hidden_dims=[128, 64],
            fusion_type='attention',
            dropout=0.1
        )
        
        # 设置数据编码器维度
        if data_dim is None and self.data_stats is not None:
            data_dim = 4  # 使用统计特征维度
        
        self.reward_network.set_data_encoder_dim(data_dim)
        
        # 创建训练器
        self.reward_trainer = RewardNetworkTrainer(
            self.reward_network,
            lr=1e-3,
            weight_decay=1e-5
        )
        
        print(f"奖励网络已初始化，数据编码器维度: {data_dim}")
    
    def set_training_data(self, X, y):
        """设置训练数据并计算统计特征"""
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
        
        # 计算数据统计特征（用于数据编码器）
        X_tensor = torch.FloatTensor(X)
        if X_tensor.dim() > 2:
            X_tensor = X_tensor.view(X_tensor.size(0), -1)
            
        self.data_stats = {
            'mean': X_tensor.mean().item(),
            'std': X_tensor.std().item() if X_tensor.numel() > 1 else 0.0,
            'min': X_tensor.min().item(),
            'max': X_tensor.max().item()
        }
        
        # 初始化奖励网络的数据编码器维度
        if self.reward_network is not None:
            self.reward_network.set_data_encoder_dim(4)
    
    def compute_oracle_target(self, true_expression):
        """阶段一：计算神谕目标向量"""
        print(f"计算神谕目标: {true_expression}")
        
        if self.reward_network is None:
            raise ValueError("奖励网络未初始化")
        
        self.true_expression = true_expression
        
        self.reward_network.eval()
        with torch.no_grad():
            try:
                # 使用当前版本的奖励网络表达式编码器
                oracle_embedding = self.reward_network.expression_embedding.encode_expressions([str(true_expression)])
                self.oracle_target = torch.FloatTensor(oracle_embedding)
                
                # 确保是2D张量
                if self.oracle_target.dim() == 1:
                    self.oracle_target = self.oracle_target.unsqueeze(0)
                elif self.oracle_target.dim() > 2:
                    self.oracle_target = self.oracle_target.view(self.oracle_target.size(0), -1)
                    
                print(f"神谕目标向量形状: {self.oracle_target.shape}")
                return True
                
            except Exception as e:
                print(f"计算神谕目标失败: {e}")
                self.oracle_target = None
                return False
    
    def compute_hybrid_reward(self, expression_str, r2_score, complexity):
        """计算混合奖励"""
        # 性能分量
        r2_clipped = max(0, min(1, r2_score))
        z_perf = 1.0 / (1.0 + np.exp(-10 * (r2_clipped - 0.5)))
        
        # 结构分量
        if self.oracle_target is not None and self.reward_network is not None:
            self.reward_network.eval()
            with torch.no_grad():
                try:
                    expr_embedding = self.reward_network.expression_embedding.encode_expressions([expression_str])
                    expr_tensor = torch.FloatTensor(expr_embedding)
                    z_struct = torch.cosine_similarity(expr_tensor, self.oracle_target).item()
                except Exception as e:
                    complexity_penalty = np.exp(-complexity / 100)
                    z_struct = complexity_penalty
        else:
            complexity_penalty = np.exp(-complexity / 100)
            z_struct = complexity_penalty
        
        # 混合奖励
        z_hybrid = self.alpha_hybrid * z_perf + (1 - self.alpha_hybrid) * z_struct
        
        return z_hybrid, z_perf, z_struct
    
    def add_experience(self, expression_str, r2_score, complexity, hybrid_reward):
        """添加经验到回放池"""
        if self.data_stats is None:
            return
        
        # 使用数据统计特征作为代表性样本
        X_repr = torch.tensor([
            self.data_stats['mean'],
            self.data_stats['std'],
            self.data_stats['min'],
            self.data_stats['max']
        ], dtype=torch.float32).unsqueeze(0)  # (1, 4)
        
        target_reward = torch.FloatTensor([[hybrid_reward]])
        
        # 存储经验
        self.experience_buffer.push(expression_str, X_repr, target_reward)
    
    def train_reward_network(self, num_epochs=1, batch_size=32):
        """阶段四：训练奖励网络"""
        if len(self.experience_buffer) == 0:
            print("经验池为空，跳过训练")
            return 0
        
        print(f"训练奖励网络，经验池大小: {len(self.experience_buffer)}")
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # 随机打乱经验池
            indices = list(range(len(self.experience_buffer)))
            random.shuffle(indices)
            
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                try:
                    # 采样批次
                    batch_experiences = [self.experience_buffer.buffer[idx] for idx in batch_indices]
                    expressions = [exp[0] for exp in batch_experiences]
                    X_batch = torch.stack([exp[1] for exp in batch_experiences])
                    target_rewards = torch.stack([exp[2] for exp in batch_experiences])
                    
                    # 训练一步
                    loss = self.reward_trainer.train_step(expressions, X_batch, target_rewards)
                    total_loss += loss
                    num_batches += 1
                    
                except Exception as e:
                    print(f"训练批次出错: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"奖励网络训练完成，平均损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def run_mcts_exploration(self, X, y, variables=None):
        """阶段二：MCTS探索与数据生成"""
        print(f"开始MCTS探索，迭代次数: {self.max_iterations}")
        
        # 预处理数据
        X = self._preprocess(X)
        y = np.array(y)
        
        # 设置变量
        if variables is not None:
            self.variables = variables
        else:
            if isinstance(X, np.ndarray):
                num_vars = X.shape[1] if X.ndim > 1 else 1
                self.variables = [nd.Variable(f'x{i+1}') for i in range(num_vars)]
            else:
                self.variables = [nd.Variable('x1')]
        
        # 初始化最佳表达式
        self.best_expr = None
        self.best_r2 = -np.inf
        
        # 初始化根节点
        self.root = SelfLearningNode(self.variables[:min(len(self.variables), self.max_vars)])
        self._evaluate_node(self.root, X, y)
        
        if self.root.r2 > self.best_r2:
            self.best_r2 = self.root.r2
            self.best_expr = self.root.phi
        
        # MCTS主循环
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
                print(f"  迭代 {i}: 最佳R2 = {self.best_r2:.4f}, 最佳表达式 = {self.best_expr}")
        
        return self.best_expr
    
    def run_epoch(self, X, y, true_expression, variables=None, 
                  mcts_iterations=1000, training_epochs=1):
        """运行一个完整的Epoch"""
        print(f"\n{'='*60}")
        print(f"开始新Epoch")
        print(f"{'='*60}")
        
        # 阶段一：生成神谕目标向量
        oracle_success = self.compute_oracle_target(true_expression)
        if not oracle_success:
            print("警告：神谕目标生成失败，使用简化策略")
        
        # 阶段二：MCTS探索与数据生成
        print("\n阶段二：MCTS探索与数据生成")
        print("-" * 40)
        self.max_iterations = mcts_iterations
        best_expr = self.run_mcts_exploration(X, y, variables)
        
        # 阶段三：目标生成与经验存储（在MCTS过程中实时进行）
        print(f"\n阶段三：经验收集完成")
        print(f"经验池大小: {len(self.experience_buffer)}")
        
        # 阶段四：网络训练
        print(f"\n阶段四：网络训练")
        print("-" * 40)
        training_loss = self.train_reward_network(num_epochs=training_epochs)
        
        # 记录Epoch统计信息
        epoch_info = {
            'best_expression': str(best_expr),
            'best_r2': self.best_r2,
            'experience_count': len(self.experience_buffer),
            'training_loss': training_loss,
            'oracle_available': oracle_success
        }
        
        self.epoch_stats['epochs'].append(epoch_info)
        
        # 显示Epoch结果
        print(f"\nEpoch结果:")
        print(f"最佳表达式: {best_expr}")
        print(f"最佳R2: {self.best_r2:.4f}")
        print(f"经验数量: {len(self.experience_buffer)}")
        print(f"训练损失: {training_loss:.4f}")
        
        return epoch_info
    
    def fit(self, X, y, true_expression, variables=None, 
            num_epochs=5, mcts_iterations_per_epoch=1000, 
            training_epochs_per_epoch=1):
        """运行完整的自学习训练"""
        print("开始自学习MCTS符号回归")
        print("=" * 60)
        
        # 设置训练数据
        self.set_training_data(X, y)
        
        # 初始化奖励网络（如果尚未初始化）
        if self.reward_network is None:
            expr_encoder_path = 'weights/expression_encoder'
            if not os.path.exists(expr_encoder_path + '_tokenizer.pkl'):
                raise FileNotFoundError(f"未找到表达式编码器模型: {expr_encoder_path}")
            
            self.initialize_reward_network(expr_encoder_path)
        
        # 运行多个Epoch
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            epoch_info = self.run_epoch(
                X, y, true_expression, variables,
                mcts_iterations=mcts_iterations_per_epoch,
                training_epochs=training_epochs_per_epoch
            )
        
        # 显示最终结果
        print(f"\n{'='*60}")
        print("训练完成！")
        print(f"{'='*60}")
        print(f"最终最佳表达式: {self.best_expr}")
        print(f"最终最佳R2: {self.best_r2:.4f}")
        print(f"总经验数量: {len(self.experience_buffer)}")
        
        # 保存模型
        self.save_model('weights/self_learning_mcts')
        
        return self.best_expr
    
    def predict(self, X):
        """使用最佳表达式进行预测"""
        if self.best_expr is None:
            X = self._preprocess(X)
            if isinstance(X, np.ndarray):
                return np.zeros(X.shape[0])
            else:
                return np.zeros(len(next(iter(X.values()))))
        
        X = self._preprocess(X)
        return self._evaluate_expression(self.best_expr, X)
    
    def get_score(self, X, y):
        """获取模型在给定数据上的R2和RMSE分数"""
        y_pred = self.predict(X)
        r2 = R2_score(y, y_pred)
        rmse = RMSE_score(y, y_pred)
        return r2, rmse
    
    def save_model(self, model_path):
        """保存模型"""
        if self.reward_network:
            self.reward_network.save_model(model_path)
            print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        if self.reward_network:
            self.reward_network.load_model(model_path)
            print(f"模型已从 {model_path} 加载")
    
    # 以下方法与原始MCTS实现相同
    def _preprocess(self, X):
        """数据预处理函数"""
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X
        return X
    
    def _select(self, node):
        """选择阶段"""
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
        """扩展阶段"""
        for _ in range(10):
            new_eqtrees = self._mutate_eqtrees(node.eqtrees)
            child = SelfLearningNode(new_eqtrees)
            self._evaluate_node(child, X, y)
            node.add_child(child)
            
        return random.choice(node.children)
    
    def _mutate_eqtrees(self, eqtrees):
        """变异表达式树"""
        mutation_type = random.choice(['add', 'replace', 'modify'])
        
        if mutation_type == 'add' and len(eqtrees) < self.max_vars:
            op = random.choice(self.binary_ops)
            operand1 = random.choice(eqtrees + self.variables + self.constants)
            operand2 = random.choice(eqtrees + self.variables + self.constants)
            new_eqtree = copy.deepcopy(op(operand1, operand2))
            return eqtrees + [new_eqtree]
            
        elif mutation_type == 'replace':
            idx = random.randint(0, len(eqtrees) - 1)
            new_eqtree = self._generate_random_expr()
            return eqtrees[:idx] + [new_eqtree] + eqtrees[idx+1:]
            
        else:  # modify
            idx = random.randint(0, len(eqtrees) - 1)
            eqtree_copy = copy.deepcopy(eqtrees[idx])
            
            if random.random() < 0.5:
                op = random.choice(self.binary_ops)
                operand = random.choice(self.variables + self.constants)
                if random.random() < 0.5:
                    new_eqtree = copy.deepcopy(op(eqtree_copy, operand))
                else:
                    new_eqtree = copy.deepcopy(op(operand, eqtree_copy))
            else:
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
        """模拟阶段"""
        current_eqtrees = node.eqtrees.copy()
        best_node = node
        best_r2 = -np.inf
        
        for _ in range(10):
            depth = self._get_depth(node)
            temp_eqtrees = current_eqtrees.copy()
            
            while depth < self.max_depth:
                temp_eqtrees = self._mutate_eqtrees(temp_eqtrees)
                depth += 1
            
            temp_node = SelfLearningNode(temp_eqtrees)
            self._evaluate_node(temp_node, X, y)
            
            if temp_node.r2 != -np.inf and temp_node.r2 > best_r2:
                best_node = temp_node
                best_r2 = temp_node.r2
                
        return best_node.reward, best_node
    
    def _evaluate_node(self, node, X, y):
        """评估节点"""
        try:
            # 计算每个表达式树的输出
            Z = np.zeros((len(y), 1 + len(node.eqtrees)))
            Z[:, 0] = 1.0  # 常数项
            
            for i, eqtree in enumerate(node.eqtrees):
                try:
                    expr_copy = copy.deepcopy(eqtree)
                    Z[:, i+1] = self._evaluate_expression(expr_copy, X)
                except Exception as e:
                    Z[:, i+1] = np.zeros(len(y))
            
            # 检查有效性
            if not np.isfinite(Z).all():
                node.r2 = -np.inf
                node.complexity = float('inf')
                node.reward = 0
                node.phi = nd.Number(0.0)
                return
                
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
            
            # 计算混合奖励并添加经验
            if node.phi is not None:
                expression_str = str(node.phi)
                z_hybrid, z_perf, z_struct = self.compute_hybrid_reward(
                    expression_str, node.r2, node.complexity
                )
                node.reward = z_hybrid
                
                # 添加经验到回放池
                if node.r2 > -np.inf and not np.isnan(z_hybrid) and np.isfinite(z_hybrid):
                    self.add_experience(expression_str, node.r2, node.complexity, z_hybrid)
            else:
                node.reward = 0
                
        except Exception as e:
            node.r2 = -np.inf
            node.complexity = float('inf')
            node.reward = 0
            node.phi = nd.Number(0.0)
    
    def _evaluate_expression(self, expr, X):
        """评估单个表达式"""
        try:
            if isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                X_dict = {f'x{i+1}': X[:, i] for i in range(X.shape[1])}
                return expr.eval(X_dict)
            else:
                return expr.eval(X)
        except Exception as e:
            if isinstance(X, np.ndarray):
                return np.zeros(X.shape[0])
            else:
                return np.zeros(len(next(iter(X.values()))))
    
    def _backpropagate(self, node, reward):
        """回溯阶段"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent


if __name__ == "__main__":
    # 测试代码
    print("测试自学习MCTS...")
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    X = np.random.uniform(-5, 5, (n_samples, 2))
    y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
    true_expression = "x1 + 2*x2"
    
    # 创建自学习MCTS
    mcts = SelfLearningMCTS(
        max_depth=5,
        max_iterations=100,
        max_vars=3,
        alpha_hybrid=0.7
    )
    
    print("开始训练...")
    best_expr = mcts.fit(
        X, y, true_expression,
        num_epochs=3,
        mcts_iterations_per_epoch=100,
        training_epochs_per_epoch=1
    )
    
    print(f"最佳表达式: {best_expr}")
    
    # 评估
    r2, rmse = mcts.get_score(X, y)
    print(f"R2: {r2:.4f}, RMSE: {rmse:.4f}")
    
    print("测试完成!")