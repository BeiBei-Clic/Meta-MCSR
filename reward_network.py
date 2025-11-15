import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import pickle
from collections import deque
import random

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from nd2py.utils import R2_score
from expression_encoder import ExpressionEmbedding


class DataEncoder(nn.Module):
    """数据编码器，将数据集特征编码为向量表示"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=256, dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, X):
        """
        编码数据特征
        
        Args:
            X: (batch_size, input_dim) 输入特征矩阵
        
        Returns:
            data_embedding: (batch_size, output_dim) 数据嵌入向量
        """
        return self.encoder(X)


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""
    
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, expr_emb, data_emb):
        """
        融合表达式嵌入和数据嵌入
        
        Args:
            expr_emb: (batch_size, d_model) 表达式嵌入
            data_emb: (batch_size, d_model) 数据嵌入
        
        Returns:
            fused: (batch_size, d_model) 融合后的向量
        """
        # 确保输入是2D张量
        if expr_emb.dim() == 1:
            expr_emb = expr_emb.unsqueeze(0)
        if data_emb.dim() == 1:
            data_emb = data_emb.unsqueeze(0)
            
        # 确保batch sizes匹配
        if expr_emb.size(0) != data_emb.size(0):
            min_batch = min(expr_emb.size(0), data_emb.size(0))
            expr_emb = expr_emb[:min_batch]
            data_emb = data_emb[:min_batch]
            
        # 确保维度匹配
        if expr_emb.size(-1) != data_emb.size(-1):
            min_dim = min(expr_emb.size(-1), data_emb.size(-1))
            expr_emb = expr_emb[..., :min_dim]
            data_emb = data_emb[..., :min_dim]
            
        # 添加序列维度
        expr_seq = expr_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        data_seq = data_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # 交叉注意力：表达式查询数据键值对
        try:
            attn_out, _ = self.attention(expr_seq, data_seq, data_seq)
            attn_out = attn_out.squeeze(1)
        except Exception as e:
            print(f"Attention error: {e}, expr_seq shape: {expr_seq.shape}, data_seq shape: {data_seq.shape}")
            # 如果注意力失败，使用简单的元素级操作作为后备
            attn_out = expr_emb * data_emb
        
        # 残差连接和层归一化
        fused = self.norm1(expr_emb + attn_out)
        
        # FFN
        ffn_out = self.ffn(fused)
        
        # 残差连接和层归一化
        return self.norm2(fused + ffn_out)


class RewardNetwork(nn.Module):
    """奖励网络：评估表达式-数据对的潜力"""
    
    def __init__(self, expr_encoder_path=None, d_model=256, data_hidden_dims=[128, 64], 
                 fusion_type='attention', dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # 初始化表达式嵌入器
        self.expression_embedding = ExpressionEmbedding(expr_encoder_path)
        
        # 初始化数据编码器
        self.data_encoder = DataEncoder(
            input_dim=1,  # 动态设置
            hidden_dims=data_hidden_dims,
            output_dim=d_model,
            dropout=dropout
        )
        
        # 融合模块
        if fusion_type == 'attention':
            self.fusion = CrossAttentionFusion(d_model, dropout=dropout)
        elif fusion_type == 'concat':
            self.fusion = nn.Linear(d_model * 2, d_model)
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
        
        # 输出层：预测奖励值
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()  # 奖励值在[0,1]范围内
        )
        
    def set_data_encoder_dim(self, input_dim):
        """设置数据编码器的输入维度"""
        # 获取原始的hidden_dims配置
        original_hidden_dims = [128, 64]  # 与初始化时保持一致
        
        self.data_encoder = DataEncoder(
            input_dim=input_dim,
            hidden_dims=original_hidden_dims,
            output_dim=self.d_model,
            dropout=0.1
        ).to(self.expression_embedding.device)
        
        # 确保融合模块和奖励头也在正确的设备上
        self.fusion = self.fusion.to(self.expression_embedding.device)
        self.reward_head = self.reward_head.to(self.expression_embedding.device)
    
    def forward(self, expressions, X):
        """
        前向传播
        
        Args:
            expressions: List[str] 或 str 表达式字符串列表
            X: torch.Tensor (batch_size, input_dim) 输入数据特征
        
        Returns:
            rewards: torch.Tensor (batch_size, 1) 预测的奖励值
        """
        if isinstance(expressions, str):
            expressions = [expressions]
        
        batch_size = len(expressions)
        
        # 编码表达式
        expr_embeddings = self.expression_embedding.encode_expressions(expressions)
        expr_tensor = torch.FloatTensor(expr_embeddings).to(self.expression_embedding.device)
        
        # 确保expr_tensor是2D张量
        if expr_tensor.dim() > 2:
            expr_tensor = expr_tensor.view(batch_size, -1)
        
        # 编码数据
        if X.dim() == 1:
            X = X.unsqueeze(-1)  # 添加特征维度
        
        # 确保X是2D张量
        if X.dim() > 2:
            # 如果是4D张量，可能是(batch, seq_len, feature_dim, hidden_dim)
            # 我们需要将其展平为2D，保留batch维度
            batch_size = X.size(0)
            X = X.view(batch_size, -1)  # 展平为2D
            
        # 如果X的样本数与表达式数量不匹配，进行调整
        if X.size(0) != batch_size:
            # 如果X只有一个样本，复制batch_size次
            if X.size(0) == 1:
                X = X.expand(batch_size, -1)
            # 如果表达式只有一个，使用第一个X样本
            elif batch_size == 1:
                X = X[:1]
            # 否则，取最小值
            else:
                min_size = min(X.size(0), batch_size)
                X = X[:min_size]
                expr_tensor = expr_tensor[:min_size]
                batch_size = min_size
                
        data_tensor = X.to(self.expression_embedding.device)
        
        # 确保张量维度匹配
        if expr_tensor.dim() == 1:
            expr_tensor = expr_tensor.unsqueeze(0)
        if data_tensor.dim() == 1:
            data_tensor = data_tensor.unsqueeze(0)
        
        # 编码数据
        data_encoded = self.data_encoder(data_tensor)
            
        # 融合表达和数据
        if self.fusion_type == 'attention':
            fused = self.fusion(expr_tensor, data_encoded)
        else:  # concat
            combined = torch.cat([expr_tensor, data_encoded], dim=-1)
            fused = self.fusion(combined)
        
        # 预测奖励
        rewards = self.reward_head(fused)
        
        return rewards
    
    def predict_reward(self, expressions, X):
        """
        预测表达式-数据对的奖励
        
        Args:
            expressions: List[str] 或 str 表达式字符串列表
            X: torch.Tensor (batch_size, input_dim) 输入数据特征
        
        Returns:
            rewards: torch.Tensor (batch_size, 1) 预测的奖励值
        """
        self.eval()
        with torch.no_grad():
            # 确保X是2D张量
            if X.dim() > 2:
                batch_size = X.size(0)
                X = X.view(batch_size, -1)
            elif X.dim() == 1:
                X = X.unsqueeze(0)
            
            try:
                rewards = self.forward(expressions, X)
            except Exception as e:
                print(f"预测奖励时出错: {e}")
                # 如果预测失败，返回默认值
                return 0.5
            
            # 确保返回的是标量值
            if rewards.numel() == 1:
                return rewards.item()
            else:
                return rewards.mean().item()
    
    def save_model(self, model_path):
        """保存模型"""
        # 保存表达式嵌入器
        self.expression_embedding.save_model(model_path + '_expr_encoder')
        
        # 保存奖励网络的其他部分
        model_state = {
            'data_encoder_state_dict': self.data_encoder.state_dict(),
            'fusion_state_dict': self.fusion.state_dict(),
            'reward_head_state_dict': self.reward_head.state_dict(),
            'd_model': self.d_model,
            'fusion_type': self.fusion_type
        }
        
        torch.save(model_state, model_path + '_reward_network.pth')
        print(f"奖励网络已保存到: {model_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        # 加载表达式嵌入器
        self.expression_embedding.load_model(model_path + '_expr_encoder')
        
        # 加载奖励网络的其他部分
        model_state = torch.load(model_path + '_reward_network.pth', map_location=self.expression_embedding.device)
        
        self.data_encoder.load_state_dict(model_state['data_encoder_state_dict'])
        self.fusion.load_state_dict(model_state['fusion_state_dict'])
        self.reward_head.load_state_dict(model_state['reward_head_state_dict'])
        
        print(f"奖励网络已从 {model_path} 加载")


class ExperienceReplayBuffer:
    """经验回放池"""
    
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        
    def push(self, expression, X, target_reward):
        """添加经验"""
        self.buffer.append((expression, X, target_reward))
    
    def sample(self, batch_size):
        """随机采样批次"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        expressions = [item[0] for item in batch]
        
        # 确保X张量的维度正确
        X_list = [item[1] for item in batch]
        if len(X_list) > 0:
            # 检查第一个张量的维度
            first_X = X_list[0]
            if first_X.dim() > 2:
                # 如果是高维张量，先展平为2D
                X_list = [x.view(x.size(0), -1) if x.dim() > 2 else x for x in X_list]
            
            # 确保所有张量都有相同的形状
            # 取第一个张量的形状作为参考
            target_shape = X_list[0].shape
            X_list = [x.view(target_shape) if x.shape != target_shape else x for x in X_list]
            
            X_batch = torch.stack(X_list)
        else:
            X_batch = torch.empty(0)
        
        targets = torch.stack([item[2] for item in batch])
        
        # 确保targets的形状正确 (batch_size, 1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        elif targets.dim() > 2:
            targets = targets.view(targets.size(0), -1)
        
        return expressions, X_batch, targets
    
    def __len__(self):
        return len(self.buffer)


class RewardNetworkTrainer:
    """奖励网络训练器"""
    
    def __init__(self, reward_network, lr=1e-3, weight_decay=1e-5):
        self.model = reward_network
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.device = self.model.expression_embedding.device
        
    def train_step(self, expressions, X, target_rewards):
        """训练一步"""
        self.model.train()
        
        # 前向传播
        predicted_rewards = self.model(expressions, X)
        
        # 计算损失
        loss = self.criterion(predicted_rewards, target_rewards.to(self.device))
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, expressions, X, target_rewards):
        """验证"""
        self.model.eval()
        
        with torch.no_grad():
            predicted_rewards = self.model(expressions, X)
            loss = self.criterion(predicted_rewards, target_rewards.to(self.device))
        
        return loss.item()


if __name__ == "__main__":
    # 测试代码
    print("测试奖励网络...")
    
    # 创建测试数据
    test_expressions = ["x1 + x2", "sin(x1)", "x1 * x2"]
    X_test = torch.randn(3, 5)  # 3个样本，5个特征
    target_rewards = torch.tensor([[0.8], [0.6], [0.9]], dtype=torch.float32)
    
    # 创建奖励网络
    reward_net = RewardNetwork()
    
    # 训练器
    trainer = RewardNetworkTrainer(reward_net)
    
    print("测试前向传播...")
    rewards = reward_net(test_expressions, X_test)
    print(f"预测奖励: {rewards}")
    print(f"目标奖励: {target_rewards}")
    
    print("测试训练...")
    loss = trainer.train_step(test_expressions, X_test, target_rewards)
    print(f"训练损失: {loss:.4f}")
    
    print("测试完成!")