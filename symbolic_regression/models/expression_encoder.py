"""
表达式编码器模块

基于Transformer架构，将数学表达式的符号结构映射到语义丰富的嵌入空间。
支持对比学习预训练和真实解引导微调。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union
import pickle
import os
import re
from collections import Counter


class ExpressionTokenizer:
    """表达式分词器，将数学表达式转换为token序列（本地实现）"""
    
    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        
        # 构建词汇表
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # 特殊token的索引
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        
        # 重新构建vocab以包含特殊token
        special_tokens = {
            '<PAD>': self.pad_token_id,
            '<UNK>': self.unk_token_id,
            '<CLS>': self.cls_token_id,
            '<SEP>': self.sep_token_id,
            '<VAR>': self.vocab.get('<VAR>', 10000),
            '<CONST>': self.vocab.get('<CONST>', 10001), 
            '<OP>': self.vocab.get('<OP>', 10002),
            '<FUNC>': self.vocab.get('<FUNC>', 10003),
            '<LPAREN>': self.vocab.get('<LPAREN>', 10004),
            '<RPAREN>': self.vocab.get('<RPAREN>', 10005)
        }
        
        # 重新分配特殊token的索引
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<CLS>': 2,
            '<SEP>': 3
        }
        
        # 合并特殊token到主vocab
        self.vocab.update(self.special_tokens)
        self.vocab_size = len(self.vocab)
        
        # 创建反向词汇表
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def _build_vocab(self) -> Dict[str, int]:
        """构建基本词汇表"""
        # 运算符和标点符号（包括小数点）
        operators = ['+', '-', '*', '/', '^', '(', ')', ',', '=', '<', '>', '!', '.']
        
        # 数学函数
        functions = ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs', 'min', 'max', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'log10', 'sec', 'csc', 'cot', 'floor', 'ceil', 'round']
        
        # 变量模式 - 扩展到更多变量
        variables = [f'x{i}' for i in range(1, 1001)]  # x1, x2, ..., x1000
        
        # 添加更多变量名格式
        additional_variables = [f'v{i}' for i in range(1, 101)] + [f'y{i}' for i in range(1, 51)] + [f'z{i}' for i in range(1, 51)]
        
        # 特殊token
        special = ['<PAD>', '<UNK>', '<CLS>', '<SEP>']
        
        # 构建词汇表
        vocab = {}
        token_list = operators + functions + variables + additional_variables + special
        
        for i, token in enumerate(token_list):
            vocab[token] = i
            
        return vocab
        
    def tokenize_expression(self, expression: str) -> Dict[str, torch.Tensor]:
        """
        将表达式字符串token化
        
        Args:
            expression: 数学表达式字符串，如 "x1 + sin(x2)"
            
        Returns:
            包含input_ids, attention_mask的字典
        """
        # 预处理表达式
        tokens = self._preprocess_and_tokenize(expression)
        
        # 添加CLS和SEP token
        tokens = ['<CLS>'] + tokens + ['<SEP>']
        
        # 截断到max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + ['<SEP>']
        
        # 转换为ID
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # 对于不在词汇表中的token，尝试分解或使用<UNK>
                # 首先尝试将token分解为更小的部分
                if re.match(r'\d+\.\d+', token):  # 浮点数
                    # 将浮点数分解为整数部分和小数部分
                    parts = token.split('.')
                    token_ids.append(self.vocab.get(parts[0], self.unk_token_id))
                    token_ids.append(self.vocab.get('.', self.unk_token_id))
                    token_ids.append(self.vocab.get(parts[1], self.unk_token_id))
                else:
                    # 直接使用<UNK>
                    token_ids.append(self.unk_token_id)
        
        # 创建attention mask
        attention_mask = [1] * len(token_ids)
        
        # Padding
        padding_length = self.max_length - len(token_ids)
        token_ids.extend([self.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def _preprocess_and_tokenize(self, expression: str) -> List[str]:
        """预处理并分词表达式"""
        # 保留所有原始信息，不进行统一替换
        # 定义支持的函数列表
        functions = ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs', 'min', 'max', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'log10', 'sec', 'csc', 'cot', 'floor', 'ceil', 'round']
        
        # 运算符和标点符号
        operators = ['+', '-', '*', '/', '^', '(', ')', ',', '=', '<', '>', '!']
        
        # 分割为token
        tokens = []
        i = 0
        
        while i < len(expression):
            # 跳过空白字符
            if expression[i].isspace():
                i += 1
                continue
            
            char = expression[i]
            
            # 检查是否是运算符或标点符号
            if char in operators:
                # 处理多个字符的运算符
                if i + 1 < len(expression):
                    two_char_op = expression[i:i+2]
                    if two_char_op in ['<=', '>=', '==', '!=']:
                        tokens.append(two_char_op)
                        i += 2
                        continue
                
                tokens.append(char)
                i += 1
                continue
            
            # 收集连续的非运算符字符
            j = i
            while j < len(expression) and not expression[j].isspace() and expression[j] not in operators:
                j += 1
            
            token = expression[i:j].strip()
            
            if token:
                tokens.append(token)
            
            i = j
        
        return tokens


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
        
        # 创建本地Transformer编码器
        self.token_embedding = nn.Embedding(self.get_vocab_size(), embedding_dim)
        
        # 位置编码
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            layer_norm_eps=1e-5
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer normalization
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        
        # 投影层
        self.projection = nn.Linear(embedding_dim, self.projection_dim)
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(self.projection_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        tokenizer = ExpressionTokenizer()
        return tokenizer.vocab_size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: tokenized expression ids
            attention_mask: attention mask
            
        Returns:
            表达式嵌入向量
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        token_embeddings = self.token_embedding(input_ids)
        
        # Position embedding
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Transformer encoding
        # For TransformerEncoder, we need src_key_padding_mask (batch_size, seq_len)
        src_key_padding_mask = attention_mask == 0  # True for padding positions
        
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Use CLS token (first token) as representation
        cls_token = encoded[:, 0]  # (batch_size, embedding_dim)
        
        # Apply final layer norm
        cls_token = self.final_layer_norm(cls_token)
        
        # Projection and normalization
        projected = self.projection(cls_token)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)
        
        return projected
    
    def encode(self, expression: str, training: bool = False) -> torch.Tensor:
        """
        编码单个表达式

        Args:
            expression: 数学表达式字符串
            training: 是否在训练模式下（影响是否计算梯度）

        Returns:
            归一化的嵌入向量
        """
        self.train()
        # 分词
        tokenizer = ExpressionTokenizer()
        encoded = tokenizer.tokenize_expression(expression)

        # 将张量移动到模型所在的设备
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].unsqueeze(0).to(device)
        attention_mask = encoded['attention_mask'].unsqueeze(0).to(device)

        # 动态扩展词汇表和embedding层以处理新的token
        tokenizer = ExpressionTokenizer()
        vocab = tokenizer.vocab
        vocab_size = len(vocab)
        
        # 检查是否有不在当前词汇表中的token
        token_ids = encoded['input_ids'].tolist()
        vocab_items = list(tokenizer.vocab.items())
        
        if hasattr(self, 'token_embedding'):
            current_vocab_size = self.token_embedding.num_embeddings
            if current_vocab_size < vocab_size:
                # 扩展 embedding 层
                old_state_dict = self.state_dict()
                self.token_embedding = nn.Embedding(vocab_size, self.embedding_dim).to(device)
                # 复制旧权重
                old_size = min(old_state_dict['token_embedding.weight'].shape[0], vocab_size)
                self.token_embedding.weight.data[:old_size] = old_state_dict['token_embedding.weight'][:old_size]
                # 重新初始化新权重
                nn.init.normal_(self.token_embedding.weight[old_size:], mean=0, std=0.02)
                
            # 重新映射 token IDs 到新的词汇表
            new_input_ids = []
            for token_id in token_ids:
                if token_id < len(tokenizer.id_to_token):
                    token = tokenizer.id_to_token[token_id]
                    new_id = vocab.get(token, tokenizer.unk_token_id)
                    new_input_ids.append(new_id)
                else:
                    new_input_ids.append(tokenizer.unk_token_id)
            
            input_ids = torch.tensor(new_input_ids, dtype=torch.long).unsqueeze(0).to(device)

        # 编码
        embedding = self.forward(input_ids, attention_mask)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0)  # 返回张量而不是numpy数组
    
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