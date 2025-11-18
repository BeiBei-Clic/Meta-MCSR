"""
数据编码器模块

基于Transformer或深度集合网络架构，将数据集的内在模式与特征
映射到与表达式编码器相同的嵌入空间。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import os


class DataEncoder(nn.Module):
    """数据编码器，将数据集映射到与表达式编码器相同的嵌入空间"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_features: int = 100,
        projection_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim or embedding_dim
        self.max_features = max_features
        
        # 数据特征编码层
        self.feature_embedding = nn.Linear(1, embedding_dim)
        
        # 位置编码
        self.position_embedding = nn.Embedding(max_features, embedding_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 投影层
        self.projection = nn.Linear(embedding_dim, self.projection_dim)
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(self.projection_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 数据张量，形状为 (batch_size, n_features, n_samples)
            
        Returns:
            数据特征嵌入向量
        """
        batch_size, n_features, n_samples = data.shape
        
        # 如果特征数量超过最大值，截断
        if n_features > self.max_features:
            data = data[:, :self.max_features, :]
            n_features = self.max_features
        
        # 对每个特征单独应用线性变换
        # data: (batch_size, n_features, n_samples)
        feature_embeddings_list = []
        for i in range(n_features):
            # 提取第i个特征 (batch_size, n_samples)
            feature_i = data[:, i, :]  # (batch_size, n_samples)
            # 转换为 (batch_size * n_samples, 1)
            feature_i_flat = feature_i.reshape(-1, 1)
            # 应用线性变换 (batch_size * n_samples, embedding_dim)
            feature_i_embedded = self.feature_embedding(feature_i_flat)
            feature_embeddings_list.append(feature_i_embedded)
        
        # 堆叠所有特征 (batch_size * n_samples, n_features, embedding_dim)
        feature_embeddings = torch.stack(feature_embeddings_list, dim=1)
        
        # 重新排列为 (batch_size, n_samples, n_features, embedding_dim)
        feature_embeddings = feature_embeddings.view(batch_size, n_samples, n_features, self.embedding_dim)
        
        # 在样本维度上求平均得到 (batch_size, n_features, embedding_dim)
        feature_embeddings = feature_embeddings.mean(dim=1)
        
        # 位置编码
        positions = torch.arange(n_features, device=data.device).unsqueeze(0)  # (1, n_features)
        position_embeddings = self.position_embedding(positions)  # (1, n_features, embedding_dim)
        
        # 合并特征和位置嵌入
        embeddings = feature_embeddings + position_embeddings  # (batch_size, n_features, embedding_dim)
        
        # Transformer编码
        encoded = self.transformer_encoder(embeddings)  # (batch_size, n_features, embedding_dim)
        
        # 全局池化
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)  # (batch_size, embedding_dim)
        
        # 最终嵌入
        final_embedding = pooled  # (batch_size, embedding_dim)
        
        # 投影和归一化
        projected = self.projection(final_embedding)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)
        
        return projected
    
    def encode(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        编码数据集
        
        Args:
            X: 输入特征，形状为 (n_samples, n_features)
            y: 目标值，形状为 (n_samples,)
            
        Returns:
            归一化的数据嵌入向量
        """
        self.train()

        # 确保数据在正确的设备上
        device = next(self.parameters()).device
        X = X.to(device)
        y = y.to(device)

        # 组合数据和目标
        # 我们的目标是创建 (n_features + 1, n_samples) 的张量，其中最后一维是目标值
        # X: (n_samples, n_features) -> (n_features, n_samples)
        # y: (n_samples,) -> (1, n_samples)
        X_transposed = X.T  # (n_features, n_samples)
        y_transposed = y.unsqueeze(0)  # (1, n_samples)

        # 确保样本数量匹配
        assert X_transposed.shape[1] == y_transposed.shape[1], f"样本数量不匹配: X有{X_transposed.shape[1]}个样本, y有{y_transposed.shape[1]}个样本"

        # 将X和y在特征维度上连接
        # 创建一个 (n_features + 1, n_samples) 的张量
        data = torch.cat([X_transposed, y_transposed], dim=0).unsqueeze(0)  # (1, n_features + 1, n_samples)

        # 编码
        embedding = self.forward(data)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0)  # 返回张量
    
    def save_pretrained(self, save_directory: str):
        """保存预训练权重"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        
        # 保存配置
        config = {
            'embedding_dim': self.embedding_dim,
            'projection_dim': self.projection_dim,
            'dropout': self.dropout.p,
            'max_features': self.max_features,
        }
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            import json
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'DataEncoder':
        """从预训练权重加载模型"""
        # 加载配置
        with open(os.path.join(load_directory, 'config.json'), 'r') as f:
            import json
            config = json.load(f)
        
        # 创建模型实例
        model = cls(**config)
        
        # 加载权重
        model.load_state_dict(torch.load(os.path.join(load_directory, 'pytorch_model.bin')))
        
        return model


class DeepSetDataEncoder(nn.Module):
    """基于深度集合网络的数据编码器"""
    
    def __init__(
        self,
        input_dim: int = 1,  # 特征维度
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.1,
        n_layers: int = 3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 逐点变换网络 (PointNet风格)
        self.point_net = nn.ModuleList()
        self.point_net.append(nn.Linear(input_dim, hidden_dim))
        self.point_net.append(nn.ReLU())
        self.point_net.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 2):
            self.point_net.append(nn.Linear(hidden_dim, hidden_dim))
            self.point_net.append(nn.ReLU())
            self.point_net.append(nn.Dropout(dropout))
            
        self.point_net.append(nn.Linear(hidden_dim, output_dim))
        
        # 池化网络 (注意力池化)
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 最终投影
        self.projection = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 数据张量，形状为 (batch_size, n_features, n_samples)
            
        Returns:
            数据特征嵌入向量
        """
        batch_size, n_features, n_samples = data.shape
        
        # 逐点特征变换
        # data: (batch_size, n_features, n_samples) -> (batch_size * n_features, n_samples, input_dim)
        data_flat = data.transpose(1, 2).reshape(-1, n_samples, self.input_dim)
        
        # 通过点网络
        for layer in self.point_net:
            data_flat = layer(data_flat)
        
        # 重塑回原始形状
        # (batch_size * n_features, n_samples, output_dim) -> (batch_size, n_features, n_samples, output_dim)
        data_reshaped = data_flat.view(batch_size, n_features, n_samples, self.output_dim)
        
        # 池化样本维度
        # (batch_size, n_features, n_samples, output_dim) -> (batch_size, n_features, n_samples, output_dim)
        data_transposed = data_reshaped.transpose(1, 2)  # (batch_size, n_samples, n_features, output_dim)
        
        # 注意力池化
        data_for_pooling = data_transposed.reshape(batch_size * n_samples, n_features, self.output_dim)
        
        # 自注意力池化
        attended, _ = self.attention_pooling(
            data_for_pooling, 
            data_for_pooling, 
            data_for_pooling
        )
        
        # 再次重塑
        attended = attended.view(batch_size, n_samples, n_features, self.output_dim)
        
        # 最终池化
        pooled = attended.mean(dim=1)  # (batch_size, n_features, output_dim)
        
        # 全局池化
        final_embedding = pooled.mean(dim=1)  # (batch_size, output_dim)
        
        # 投影和归一化
        projected = self.projection(final_embedding)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)
        
        return projected
    
    def encode(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        编码数据集
        
        Args:
            X: 输入特征，形状为 (n_samples, n_features)
            y: 目标值，形状为 (n_samples,)
            
        Returns:
            归一化的数据嵌入向量
        """
        self.train()

        # 确保数据在正确的设备上
        device = next(self.parameters()).device
        X = X.to(device)
        y = y.to(device)

        # 组合数据和目标
        # 我们的目标是创建 (n_features + 1, n_samples) 的张量，其中最后一维是目标值
        # X: (n_samples, n_features) -> (n_features, n_samples)
        # y: (n_samples,) -> (1, n_samples)
        X_transposed = X.T  # (n_features, n_samples)
        y_transposed = y.unsqueeze(0)  # (1, n_samples)

        # 确保样本数量匹配
        assert X_transposed.shape[1] == y_transposed.shape[1], f"样本数量不匹配: X有{X_transposed.shape[1]}个样本, y有{y_transposed.shape[1]}个样本"

        # 将X和y在特征维度上连接
        # 创建一个 (n_features + 1, n_samples) 的张量
        data = torch.cat([X_transposed, y_transposed], dim=0).unsqueeze(0)  # (1, n_features + 1, n_samples)

        # 编码
        embedding = self.forward(data)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0)  # 返回张量