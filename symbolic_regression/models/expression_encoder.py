"""
表达式编码器模块

基于Transformer架构，将数学表达式的符号结构映射到语义丰富的嵌入空间。
支持对比学习预训练和真实解引导微调。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Optional, Union
import pickle
import os


class ExpressionTokenizer:
    """表达式分词器，将数学表达式转换为token序列"""
    
    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        # 自定义token映射
        self.special_tokens = {
            '<VAR>': 50000,
            '<CONST>': 50001, 
            '<OP>': 50002,
            '<FUNC>': 50003,
            '<LPAREN>': 50004,
            '<RPAREN>': 50005
        }
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(self.special_tokens.keys())})
        
    def tokenize_expression(self, expression: str) -> Dict[str, torch.Tensor]:
        """
        将表达式字符串token化
        
        Args:
            expression: 数学表达式字符串，如 "x1 + sin(x2)"
            
        Returns:
            包含input_ids, attention_mask的字典
        """
        # 预处理表达式
        processed_expr = self._preprocess_expression(expression)
        
        # 分词
        encoded = self.tokenizer(
            processed_expr,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded
    
    def _preprocess_expression(self, expression: str) -> str:
        """预处理表达式，添加token标记"""
        # 替换变量和常数为特殊token
        import re
        
        # 匹配数字
        expression = re.sub(r'\b\d+\.?\d*\b', '<CONST>', expression)
        
        # 匹配函数
        functions = ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt']
        for func in functions:
            expression = re.sub(rf'\b{func}\s*\(', f'<FUNC>(', expression)
        
        # 匹配运算符
        operators = ['+', '-', '*', '/', '^', '(', ')']
        for op in operators:
            expression = expression.replace(op, f' {op} ')
        
        # 匹配变量 (x1, x2, etc.)
        expression = re.sub(r'\bx\d+\b', '<VAR>', expression)
        
        return expression.strip()


class ExpressionEncoder(nn.Module):
    """表达式编码器，基于Transformer架构"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_seq_length: int = 128,
        projection_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim or embedding_dim
        
        # 加载预训练的Transformer模型
        self.transformer = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
        
        # 调整transformer配置
        config = self.transformer.config
        config.hidden_size = embedding_dim
        config.num_attention_heads = n_heads
        config.num_hidden_layers = n_layers
        config.max_position_embeddings = max_seq_length
        config.hidden_dropout_prob = dropout
        
        # 重新配置模型
        self.transformer.resize_token_embeddings(50006)  # 添加特殊token
        
        # 投影层
        self.projection = nn.Linear(embedding_dim, self.projection_dim)
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(self.projection_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: tokenized expression ids
            attention_mask: attention mask
            
        Returns:
            表达式嵌入向量
        """
        # 获取transformer输出
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用CLS token作为句子表示
        hidden_states = outputs.hidden_states[-1]  # 最后一层
        pooled_output = hidden_states[:, 0]  # [CLS] token
        
        # 投影和归一化
        projected = self.projection(pooled_output)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)
        
        return projected
    
    def encode(self, expression: str) -> np.ndarray:
        """
        编码单个表达式
        
        Args:
            expression: 数学表达式字符串
            
        Returns:
            归一化的嵌入向量
        """
        self.eval()
        with torch.no_grad():
            # 分词
            tokenizer = ExpressionTokenizer()
            encoded = tokenizer.tokenize_expression(expression)
            
            input_ids = encoded['input_ids'].unsqueeze(0)
            attention_mask = encoded['attention_mask'].unsqueeze(0)
            
            # 编码
            embedding = self.forward(input_ids, attention_mask)
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.squeeze(0).cpu().numpy()
    
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
        }
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            import json
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'ExpressionEncoder':
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