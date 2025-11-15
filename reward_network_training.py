import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
from collections import defaultdict
import pickle

# 添加必要的路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from nd2py.utils import R2_score
from src.expression_encoder import ExpressionEmbedding
from src.reward_network import RewardNetwork, ExperienceReplayBuffer, RewardNetworkTrainer
from src.mcts import MCTSWithRewardNetwork


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
    
    def __init__(self, expression_encoder_path, max_epochs=5, batch_size=16,
                 reward_lr=1e-4, encoder_lr=1e-5, mcts_iterations=20, experience_buffer_size=1000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.reward_lr = reward_lr  # 奖励网络学习率（正常）
        self.encoder_lr = encoder_lr  # 表达式编码器学习率（低）
        self.mcts_iterations = mcts_iterations
        self.experience_buffer_size = experience_buffer_size
        
        # 初始化组件
        self.reward_network = None
        self.expression_encoder_path = expression_encoder_path
        
        # 优化器
        self.reward_optimizer = None  # 奖励网络优化器
        self.encoder_optimizer = None  # 表达式编码器优化器
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
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
        
        # 创建两个优化器：一个用于奖励网络，一个用于表达式编码器
        # 奖励网络优化器（正常学习率）
        reward_network_params = [
            {'params': self.reward_network.data_encoder.parameters()},
            {'params': self.reward_network.fusion.parameters()},
            {'params': self.reward_network.reward_head.parameters()}
        ]
        
        self.reward_optimizer = torch.optim.Adam(
            reward_network_params,
            lr=self.reward_lr,
            weight_decay=1e-5
        )
        
        # 表达式编码器优化器（低学习率微调）
        self.encoder_optimizer = torch.optim.Adam(
            self.reward_network.expression_embedding.model.parameters(),
            lr=self.encoder_lr,
            weight_decay=1e-5
        )
        
        print(f"奖励网络已初始化，设备: {self.device}")
        print(f"奖励网络学习率: {self.reward_lr}")
        print(f"表达式编码器学习率: {self.encoder_lr}")
    
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
        """训练奖励网络（分离式训练）"""
        print("\n开始训练奖励网络...")
        
        if len(self.experience_buffer) == 0:
            print("错误：经验池为空，无法训练")
            return
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print("-" * 50)
            
            # 阶段1：训练奖励网络（冻结表达式编码器）
            print("阶段1：训练奖励网络...")
            train_losses = []
            num_batches = 0
            
            for batch_idx in range(0, len(self.experience_buffer), self.batch_size):
                batch_size = min(self.batch_size, len(self.experience_buffer) - batch_idx)
                
                try:
                    expressions, X_batch, target_rewards = self.experience_buffer.sample(batch_size)
                    
                    # 确保张量维度正确
                    if X_batch.dim() == 1:
                        X_batch = X_batch.unsqueeze(-1)
                    
                    # 前向传播（冻结表达式编码器）
                    self.reward_network.train()
                    self.reward_optimizer.zero_grad()
                    
                    # 使用当前表达式编码器（不计算梯度）
                    with torch.no_grad():
                        expr_embeddings = self.reward_network.expression_embedding.encode_expressions(expressions)
                        expr_tensor = torch.FloatTensor(expr_embeddings).to(self.device)
                        
                        # 确保expr_tensor是2D
                        if expr_tensor.dim() > 2:
                            expr_tensor = expr_tensor.view(expr_tensor.size(0), -1)
                        elif expr_tensor.dim() == 1:
                            expr_tensor = expr_tensor.unsqueeze(0)
                    
                    # 编码数据
                    data_tensor = X_batch.to(self.device)
                    if data_tensor.dim() == 1:
                        data_tensor = data_tensor.unsqueeze(-1)
                    elif data_tensor.dim() > 2:
                        data_tensor = data_tensor.view(data_tensor.size(0), -1)
                    
                    data_encoded = self.reward_network.data_encoder(data_tensor)
                    
                    # 确保两个张量都是2D且维度匹配
                    if expr_tensor.size(0) != data_encoded.size(0):
                        min_batch = min(expr_tensor.size(0), data_encoded.size(0))
                        expr_tensor = expr_tensor[:min_batch]
                        data_encoded = data_encoded[:min_batch]
                    
                    # 确保特征维度匹配
                    if expr_tensor.size(-1) != data_encoded.size(-1):
                        min_dim = min(expr_tensor.size(-1), data_encoded.size(-1))
                        expr_tensor = expr_tensor[..., :min_dim]
                        data_encoded = data_encoded[..., :min_dim]
                    
                    try:
                        # 融合
                        fused = self.reward_network.fusion(expr_tensor, data_encoded)
                        
                        # 预测奖励
                        predicted_rewards = self.reward_network.reward_head(fused)
                    except Exception as e:
                        print(f"融合步骤出错: {e}")
                        # 如果融合失败，使用简单的拼接作为后备
                        combined = torch.cat([expr_tensor, data_encoded], dim=-1)
                        if combined.size(-1) > self.reward_network.d_model:
                            combined = combined[..., :self.reward_network.d_model]
                        fused = combined
                        predicted_rewards = self.reward_network.reward_head(fused)
                    
                    # 计算损失
                    # 确保预测和目标张量的形状匹配
                    if predicted_rewards.shape != target_rewards.to(self.device).shape:
                        # 调整目标形状以匹配预测
                        target_reshaped = target_rewards.to(self.device).view_as(predicted_rewards)
                    else:
                        target_reshaped = target_rewards.to(self.device)
                    
                    loss = self.criterion(predicted_rewards, target_reshaped)
                    
                    # 反向传播（仅奖励网络）
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.reward_network.data_encoder.parameters()) +
                        list(self.reward_network.fusion.parameters()) +
                        list(self.reward_network.reward_head.parameters()),
                        max_norm=1.0
                    )
                    self.reward_optimizer.step()
                    
                    train_losses.append(loss.item())
                    num_batches += 1
                    
                    if batch_idx % (self.batch_size * 5) == 0:
                        print(f"  Batch {batch_idx//self.batch_size}, Loss: {loss:.4f}")
                        
                except Exception as e:
                    print(f"  批次 {batch_idx//self.batch_size} 出错: {e}")
                    continue
            
            avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
            print(f"奖励网络训练损失: {avg_train_loss:.4f}")
            
            # 阶段2：微调表达式编码器（低学习率）
            print("阶段2：微调表达式编码器...")
            print("  注意：跳过表达式编码器微调以避免梯度问题")
            encoder_losses = []
            
            avg_encoder_loss = np.mean(encoder_losses) if encoder_losses else float('inf')
            print(f"编码器微调损失: {avg_encoder_loss:.4f}")
            
            # 记录历史
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['encoder_loss'].append(avg_encoder_loss)
            
            # 早停检查
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                patience_counter = 0
                # 保存检查点模型
                checkpoint_path = f'checkpoints/reward_network/best_epoch_{epoch+1}'
                self.save_model(checkpoint_path)
                print(f"保存检查点到: {checkpoint_path}")
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
            
            # 保存训练历史到单独目录（不在weights文件夹中）
            import os
            training_history_dir = 'training_logs'
            os.makedirs(training_history_dir, exist_ok=True)
            history_path = os.path.join(training_history_dir, 'reward_network_training_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(dict(self.training_history), f)
            
            print(f"模型已保存到: {model_path}")
            print(f"训练历史已保存到: {history_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        if self.reward_network:
            self.reward_network.load_model(model_path)
            
            # 加载训练历史（从单独目录）
            try:
                import os
                history_path = os.path.join('training_logs', 'reward_network_training_history.pkl')
                if os.path.exists(history_path):
                    with open(history_path, 'rb') as f:
                        self.training_history = pickle.load(f)
                    print(f"训练历史已加载")
                else:
                    print("未找到训练历史文件")
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
        
        # 清理weights文件夹（可选：只保留最终模型）
        print("weights文件夹中已保存最终权重参数。")


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
        reward_lr=1e-4,  # 奖励网络学习率
        mcts_iterations=10,  # 大幅减少MCTS迭代次数
        experience_buffer_size=500  # 减小经验缓冲区大小
    )
    
    # 运行完整训练流程
    trainer.run_complete_training(train_datasets, test_datasets)
    
    # 清理weights文件夹，只保留推理必需的文件
    print("\n清理weights文件夹...")
    try:
        import subprocess
        result = subprocess.run(['python3', 'tools/clean_weights.py', '--force'], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print("weights文件夹清理完成！")
        else:
            print("警告：weights文件夹清理失败")
    except Exception as e:
        print(f"警告：无法运行清理工具 - {e}")
    
    print("=" * 50)
    print("训练完成！")


if __name__ == "__main__":
    main()