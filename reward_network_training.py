import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
from collections import defaultdict
import pickle

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from nd2py.utils import R2_score
from expression_encoder import ExpressionEmbedding
from reward_network import RewardNetwork, ExperienceReplayBuffer, RewardNetworkTrainer
from mcts_enhanced import MCTSWithRewardNetwork


class DataGenerator:
    """数据生成器，生成用于奖励网络训练的数据集"""
    
    def __init__(self, n_samples=100, n_features=3, noise_level=0.1):  # 进一步减小样本数量从200到100
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_level = noise_level
        
    def generate_expression_data(self, expression_func, X_range=(-3, 3)):
        """为给定表达式生成数据"""
        X = np.random.uniform(X_range[0], X_range[1], (self.n_samples, self.n_features))
        y = expression_func(X) + np.random.normal(0, self.noise_level, self.n_samples)
        return X, y
    
    def create_dataset(self):
        """创建完整的数据集"""
        # 定义不同的表达式函数
        expression_functions = [
            # 线性函数
            lambda X: X[:, 0] + 2 * X[:, 1],
            lambda X: 0.5 * X[:, 0] - X[:, 1] + 3 * X[:, 2],
            lambda X: X[:, 0] + X[:, 1] + X[:, 2],
            
            # 多项式
            lambda X: X[:, 0]**2 + X[:, 1]**2,
            lambda X: X[:, 0]**3 + 2 * X[:, 1]**2,
            lambda X: X[:, 0] * X[:, 1] + X[:, 2]**2,
            
            # 三角函数
            lambda X: np.sin(X[:, 0]) + np.cos(X[:, 1]),
            lambda X: np.sin(X[:, 0] * X[:, 1]) + np.cos(X[:, 2]),
            lambda X: np.sin(X[:, 0]) * np.cos(X[:, 1]),
            
            # 指数和对数
            lambda X: np.exp(X[:, 0]) + np.log(np.abs(X[:, 1]) + 1),
            lambda X: np.exp(-X[:, 0]**2) + np.sin(X[:, 1]),
            
            # 复杂组合
            lambda X: np.sin(X[:, 0]) * np.exp(-X[:, 1]**2) + X[:, 2]**2,
            lambda X: np.log(np.abs(X[:, 0]) + 1) * np.cos(X[:, 1]) + X[:, 2]**3,
            lambda X: np.sqrt(np.abs(X[:, 0])) + 1.0 / (1.0 + np.exp(-X[:, 1])),
            
            # 真实解
            lambda X: X[:, 0] + 2 * X[:, 1] * np.sin(X[:, 0]) + 0.5 * X[:, 2]**2,
            lambda X: np.sin(X[:, 0] + X[:, 1]) + np.cos(X[:, 0] - X[:, 1]),
        ]
        
        # 对应的真实表达式字符串
        true_expressions = [
            "x1 + 2*x2",
            "0.5*x1 - x2 + 3*x3",
            "x1 + x2 + x3",
            "x1^2 + x2^2",
            "x1^3 + 2*x2^2",
            "x1*x2 + x2^2",
            "sin(x1) + cos(x2)",
            "sin(x1*x2) + cos(x3)",
            "sin(x1)*cos(x2)",
            "exp(x1) + log(|x2| + 1)",
            "exp(-x1^2) + sin(x2)",
            "sin(x1)*exp(-x2^2) + x3^2",
            "log(|x1| + 1)*cos(x2) + x3^3",
            "sqrt(|x1|) + 1/(1 + exp(-x2))",
            "x1 + 2*x2*sin(x1) + 0.5*x3^2",
            "sin(x1 + x2) + cos(x1 - x2)",
        ]
        
        datasets = []
        
        for i, (func, true_expr) in enumerate(zip(expression_functions, true_expressions)):
            print(f"生成数据集 {i+1}/{len(expression_functions)}: {true_expr}")
            X, y = self.generate_expression_data(func)
            
            datasets.append({
                'X': X,
                'y': y,
                'true_expression': true_expr,
                'expression_id': i
            })
        
        return datasets


class RewardNetworkTrainingManager:
    """增强的奖励网络训练管理器，支持MCTS数据生成和嵌入器微调"""
    
    def __init__(self, expression_encoder_path, max_epochs=5, batch_size=16,  # 进一步减小训练轮次和批次大小
                 learning_rate=1e-4, mcts_iterations=20, experience_buffer_size=1000):  # 进一步减小MCTS迭代次数和经验池大小
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mcts_iterations = mcts_iterations
        self.experience_buffer_size = experience_buffer_size
        
        # 初始化组件
        self.reward_network = None
        self.trainer = None
        self.expression_encoder_path = expression_encoder_path
        
        # 经验回放池
        self.experience_buffer = ExperienceReplayBuffer(max_size=experience_buffer_size)
        
        # 统计信息
        self.training_history = defaultdict(list)
        
    def initialize_reward_network(self, data_encoder_dims):
        """初始化奖励网络"""
        print("初始化奖励网络...")
        
        self.reward_network = RewardNetwork(
            expr_encoder_path=self.expression_encoder_path,
            d_model=256,
            data_hidden_dims=[128, 64],
            fusion_type='attention',
            dropout=0.1
        )
        
        # 设置数据编码器维度
        self.reward_network.set_data_encoder_dim(data_encoder_dims)
        
        # 创建训练器
        self.trainer = RewardNetworkTrainer(
            self.reward_network,
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        print(f"奖励网络已初始化，设备: {self.device}")
    
    def generate_training_data_with_mcts(self, datasets, num_epochs_per_dataset=2):
        """使用MCTS生成训练数据"""
        print("使用MCTS生成训练数据...")
        
        for dataset_idx, dataset in enumerate(datasets):
            print(f"\n处理数据集 {dataset_idx + 1}/{len(datasets)}")
            
            X = dataset['X']
            y = dataset['y']
            
            # 创建增强MCTS实例，使用更小的超参数
            mcts = MCTSWithRewardNetwork(
                max_depth=3,  # 减小最大深度
                max_iterations=5,  # 大幅减少迭代次数
                max_vars=min(3, X.shape[1]),  # 限制变量数量
                reward_network=self.reward_network,
                experience_buffer=self.experience_buffer,
                alpha_hybrid=0.7
            )
            
            # 设置神谕目标
            try:
                if self.reward_network is not None:
                    mcts.set_oracle_target(dataset['true_expression'])
            except Exception as e:
                print(f"警告：无法设置神谕目标: {e}")
                # 即使没有神谕目标，也继续运行
            
            # 运行MCTS多次以收集经验
            for epoch in range(num_epochs_per_dataset):
                print(f"  Epoch {epoch + 1}/{num_epochs_per_dataset}")
                
                # 运行MCTS
                try:
                    best_expr = mcts.fit(X, y)
                    print(f"    最佳表达式: {best_expr}")
                    
                    # 获取性能统计
                    r2, rmse = mcts.get_score(X, y)
                    print(f"    R2: {r2:.4f}, RMSE: {rmse:.4f}")
                    print(f"    经验池大小: {len(mcts.experience_buffer)}")
                    
                except Exception as e:
                    print(f"    MCTS运行出错: {e}")
                    continue
            
            print(f"数据集 {dataset_idx + 1} 完成，总经验: {len(self.experience_buffer)}")
    
    def train_reward_network(self):
        """训练奖励网络"""
        print("\n开始训练奖励网络...")
        
        
        
        if len(self.experience_buffer) == 0:
            print("错误：经验池为空，无法训练")
            return
        
        best_val_loss = float('inf')
        patience = 3  # 减小早停耐心值
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print("-" * 50)
            
            # 训练
            train_losses = []
            num_batches = 0
            
            for batch_idx in range(0, len(self.experience_buffer), self.batch_size):
                batch_size = min(self.batch_size, len(self.experience_buffer) - batch_idx)
                
                try:
                    expressions, X_batch, target_rewards = self.experience_buffer.sample(batch_size)
                    
                    # 确保张量维度正确
                    if X_batch.dim() == 1:
                        X_batch = X_batch.unsqueeze(-1)
                    
                    # 训练一步
                    loss = self.trainer.train_step(expressions, X_batch, target_rewards)
                    train_losses.append(loss)
                    num_batches += 1
                    
                    if batch_idx % (self.batch_size * 5) == 0:
                        print(f"  Batch {batch_idx//self.batch_size}, Loss: {loss:.4f}")
                        
                except Exception as e:
                    print(f"  批次 {batch_idx//self.batch_size} 出错: {e}")
                    continue
            
            avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
            print(f"平均训练损失: {avg_train_loss:.4f}")
            
            # 记录历史
            self.training_history['train_loss'].append(avg_train_loss)
            
            # 早停检查
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model(f'weights/reward_network_best')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"早停：验证损失在 {patience} 个epoch内没有改善")
                break
        
        print(f"训练完成！最佳损失: {best_val_loss:.4f}")
    
    def evaluate_reward_network(self, test_datasets):
        """评估奖励网络"""
        print("\n评估奖励网络...")
        
        results = []
        
        for dataset in test_datasets:
            X, y = dataset['X'], dataset['y']
            true_expr = dataset['true_expression']
            
            # 评估一些候选表达式
            candidate_expressions = [
                str(true_expr),  # 真实解
                "x1 + x2",       # 简单表达式
                "x1",            # 更简单的表达式
                "sin(x1)",       # 包含函数的表达式
                "x1^2 + x2^2",   # 多项式
            ]
            
            scores = []
            for expr in candidate_expressions:
                try:
                    # 确保X是2D张量
                    X_tensor = torch.FloatTensor(X)
                    if X_tensor.dim() == 1:
                        X_tensor = X_tensor.unsqueeze(-1)
                    elif X_tensor.dim() > 2:
                        X_tensor = X_tensor.view(X_tensor.size(0), -1)
                    
                    # 创建一个代表性的数据样本（与训练时保持一致）
                    if X_tensor.numel() > 100:  # 如果数据太大，使用统计特征
                        X_repr = torch.tensor([
                            X_tensor.mean().item(),
                            X_tensor.std().item() if X_tensor.numel() > 1 else 0.0,
                            X_tensor.min().item(),
                            X_tensor.max().item()
                        ], dtype=torch.float32).unsqueeze(0)  # (1, 4)
                    else:
                        # 如果数据小，填充到4维
                        if X_tensor.shape[1] < 4:
                            padding = torch.zeros(X_tensor.shape[0], 4 - X_tensor.shape[1])
                            X_repr = torch.cat([X_tensor, padding], dim=1)
                        else:
                            X_repr = X_tensor[:, :4]  # 取前4个特征
                        X_repr = X_repr[:1]  # 只取第一个样本
                        
                    reward = self.reward_network.predict_reward(expr, X_repr)
                    
                    # 计算R2分数作为参考
                    # 这里简化处理，实际应该评估真正的性能
                    scores.append(reward)
                    
                except Exception as e:
                    scores.append(0.0)
                    print(f"评估表达式 '{expr}' 时出错: {e}")
            
            results.append({
                'true_expression': true_expr,
                'candidate_scores': list(zip(candidate_expressions, scores)),
                'best_candidate': candidate_expressions[np.argmax(scores)]
            })
            
            print(f"表达式: {true_expr}")
            for expr, score in zip(candidate_expressions, scores):
                print(f"  {expr:<20}: {score:.4f}")
            print(f"  最佳候选: {results[-1]['best_candidate']}")
            print()
        
        return results
    
    def save_model(self, model_path):
        """保存模型"""
        if self.reward_network:
            self.reward_network.save_model(model_path)
            
            # 保存训练历史
            with open(model_path + '_training_history.pkl', 'wb') as f:
                pickle.dump(dict(self.training_history), f)
            
            print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        if self.reward_network:
            self.reward_network.load_model(model_path)
            
            # 加载训练历史
            try:
                with open(model_path + '_training_history.pkl', 'rb') as f:
                    self.training_history = pickle.load(f)
                print(f"训练历史已加载")
            except:
                print("无法加载训练历史")
    
    def run_complete_training(self, train_datasets, test_datasets=None):
        """运行完整的训练流程"""
        print("开始完整的奖励网络训练流程")
        print("=" * 60)
        
        # 确定数据维度
        sample_X = train_datasets[0]['X']
        data_dim = sample_X.shape[1] if len(sample_X.shape) > 1 else 1
        
        # 初始化奖励网络
        self.initialize_reward_network(data_dim)
        
        # 阶段1：生成训练数据
        print("\n阶段1：生成训练数据")
        print("-" * 30)
        self.generate_training_data_with_mcts(train_datasets, num_epochs_per_dataset=2)
        
        # 阶段2：训练奖励网络
        print("\n阶段2：训练奖励网络")
        print("-" * 30)
        self.train_reward_network()
        
        # 阶段3：评估（如果有测试数据）
        if test_datasets:
            print("\n阶段3：评估奖励网络")
            print("-" * 30)
            eval_results = self.evaluate_reward_network(test_datasets)
        
        # 保存最终模型
        self.save_model('weights/reward_network_final')
        
        print("\n" + "=" * 60)
        print("奖励网络训练完成！")


def main():
    """主函数"""
    print("奖励网络训练程序")
    print("=" * 50)
    
    # 检查表达式嵌入器模型是否存在
    expr_encoder_path = 'weights/expression_encoder'
    if not os.path.exists(expr_encoder_path + '_tokenizer.pkl'):
        print("错误：未找到预训练的表达式嵌入器模型")
        print("请先运行 expression_encoder_training.py 进行预训练")
        return
    
    # 创建数据生成器
    data_generator = DataGenerator(n_samples=100, n_features=3, noise_level=0.1)
    
    print("生成数据集...")
    datasets = data_generator.create_dataset()
    
    # 减少数据集数量，只使用前4个数据集进行训练
    train_datasets = datasets[:4]
    test_datasets = datasets[4:6]  # 使用2个数据集作为测试集
    
    print(f"训练数据集: {len(train_datasets)}")
    print(f"测试数据集: {len(test_datasets)}")
    
    # 创建训练管理器，使用更小的超参数
    trainer = RewardNetworkTrainingManager(
        expression_encoder_path=expr_encoder_path,
        max_epochs=5,  # 减少训练轮数
        batch_size=16,  # 减小批次大小
        learning_rate=1e-4,
        mcts_iterations=10,  # 大幅减少MCTS迭代次数
        experience_buffer_size=500  # 减小经验缓冲区大小
    )
    
    # 运行完整训练流程
    trainer.run_complete_training(train_datasets, test_datasets)
    
    print("=" * 50)
    print("训练完成！")


if __name__ == "__main__":
    main()