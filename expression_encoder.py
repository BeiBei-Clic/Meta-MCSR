import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys
import os

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd


class ExpressionTokenizer:
    """表达式符号化器，将数学表达式转换为token序列"""
    
    def __init__(self):
        self.symbols = set()
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        
    def fit(self, expressions):
        """从表达式集合中构建词汇表"""
        for expr in expressions:
            tokens = self._tokenize_expression(expr)
            self.symbols.update(tokens)
        
        # 添加特殊token
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']
        self.symbols.update(special_tokens)
        
        # 构建映射
        sorted_symbols = sorted(self.symbols)
        self.token_to_id = {token: i for i, token in enumerate(sorted_symbols)}
        self.id_to_token = {i: token for i, token in enumerate(sorted_symbols)}
        self.vocab_size = len(self.token_to_id)
        
    def _tokenize_expression(self, expr):
        """将表达式转换为token序列"""
        expr_str = str(expr)
        tokens = []
        
        i = 0
        while i < len(expr_str):
            # 匹配函数名
            if expr_str[i:i+3] in ['sin', 'cos', 'tan', 'log', 'exp']:
                tokens.append(expr_str[i:i+3])
                i += 3
            # 匹配sqrt
            elif expr_str[i:i+4] == 'sqrt':
                tokens.append('sqrt')
                i += 4
            # 匹配变量 (x后跟数字)
            elif expr_str[i] == 'x' and i + 1 < len(expr_str) and expr_str[i+1].isdigit():
                j = i + 1
                while j < len(expr_str) and expr_str[j].isdigit():
                    j += 1
                tokens.append(expr_str[i:j])
                i = j
            # 匹配数字
            elif expr_str[i].isdigit() or (expr_str[i] == '.' and i + 1 < len(expr_str) and expr_str[i+1].isdigit()):
                j = i
                while j < len(expr_str) and (expr_str[j].isdigit() or expr_str[j] == '.'):
                    j += 1
                tokens.append(expr_str[i:j])
                i = j
            # 匹配运算符和括号
            else:
                tokens.append(expr_str[i])
                i += 1
        
        return tokens
    
    def encode(self, expression, max_length=None):
        """将表达式编码为token ID序列"""
        tokens = self._tokenize_expression(expression)
        token_ids = []
        
        # 添加SOS token
        token_ids.append(self.token_to_id['<SOS>'])
        
        for token in tokens:
            token_ids.append(self.token_to_id.get(token, self.token_to_id['<UNK>']))
        
        # 添加EOS token
        token_ids.append(self.token_to_id['<EOS>'])
        
        if max_length:
            # 截断或填充
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                token_ids[-1] = self.token_to_id['<EOS>']  # 确保以EOS结尾
            else:
                token_ids.extend([self.token_to_id['<PAD>']] * (max_length - len(token_ids)))
        
        return token_ids
    
    def decode(self, token_ids):
        """将token ID序列解码为字符串"""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '<UNK>')
            if token in ['<PAD>', '<SOS>', '<EOS>']:
                continue
            tokens.append(token)
        return ''.join(tokens)


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ExpressionEncoder(nn.Module):
    """基于Transformer的表达式编码器"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=512, max_seq_length=128, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 池化层
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, token_ids, attention_mask=None, return_sequence=False):
        """
        前向传播
        
        Args:
            token_ids: (batch_size, seq_len) token ID序列
            attention_mask: (batch_size, seq_len) 注意力掩码
            return_sequence: 是否返回序列级别的输出而不是池化后的嵌入
        
        Returns:
            如果return_sequence=False: embeddings: (batch_size, d_model) 表达式嵌入向量
            如果return_sequence=True: sequence_embeddings: (batch_size, seq_len, d_model) 序列嵌入向量
        """
        batch_size, seq_len = token_ids.size()
        
        # 嵌入
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Transformer编码
        if attention_mask is not None:
            # 将padding位置的注意力权重设为0
            attention_mask = attention_mask.bool()
            x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)
        else:
            x = self.transformer_encoder(x)
        
        if return_sequence:
            # 返回序列级别的输出
            return x
        else:
            # 全局平均池化
            if attention_mask is not None:
                # 只对非padding位置进行池化
                mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
                sum_mask = mask_expanded.sum(dim=1, keepdim=True)
                # 确保sum_mask的形状是[batch_size, 1]
                if sum_mask.dim() == 3 and sum_mask.size(-1) > 1:
                    sum_mask = sum_mask.sum(dim=-1, keepdim=True)
                # 池化操作
                x = (x * mask_expanded).sum(dim=1) / sum_mask.squeeze(-1).clamp(min=1e-9)
            else:
                x = x.mean(dim=1)
            
            return x


class ExpressionEmbedding:
    """表达式嵌入器包装类"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = ExpressionTokenizer()
        self.model = None
        self.max_seq_length = 128
        
        if model_path:
            self.load_model(model_path)
    
    def build_vocabulary(self, expressions):
        """构建词汇表"""
        self.tokenizer.fit(expressions)
        print(f"词汇表大小: {self.tokenizer.vocab_size}")
    
    def create_model(self, d_model=256, nhead=8, num_layers=6):
        """创建模型"""
        self.model = ExpressionEncoder(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_length=self.max_seq_length
        ).to(self.device)
    
    def encode_expressions(self, expressions, batch_size=32):
        """批量编码表达式为嵌入向量"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用 create_model()")
        
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(expressions), batch_size):
                batch_exprs = expressions[i:i + batch_size]
                
                # Token化
                token_ids_list = []
                attention_masks = []
                
                for expr in batch_exprs:
                    token_ids = self.tokenizer.encode(expr, self.max_seq_length)
                    token_ids_list.append(token_ids)
                    
                    # 创建注意力掩码 (非padding位置为True)
                    mask = [True] * len(token_ids)
                    attention_masks.append(mask)
                
                # 转换为tensor
                token_ids_tensor = torch.tensor(token_ids_list, device=self.device)
                attention_mask_tensor = torch.tensor(attention_masks, device=self.device)
                
                # 编码
                batch_embeddings = self.model(token_ids_tensor, attention_mask_tensor)
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def save_model(self, model_path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        # 保存tokenizer
        import pickle
        with open(model_path + '_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), model_path + '_model.pth')
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        import pickle
        
        # 加载tokenizer
        with open(model_path + '_tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # 创建模型
        self.create_model()
        
        # 加载权重
        self.model.load_state_dict(torch.load(model_path + '_model.pth', map_location=self.device))
        self.model.eval()
        
        print(f"模型已从 {model_path} 加载")
    
    def encode_single_expression(self, expression):
        """编码单个表达式"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        self.model.eval()
        token_ids = self.tokenizer.encode(expression, self.max_seq_length)
        attention_mask = [True] * len(token_ids)
        
        token_ids_tensor = torch.tensor([token_ids], device=self.device)
        attention_mask_tensor = torch.tensor([attention_mask], device=self.device)
        
        with torch.no_grad():
            embedding = self.model(token_ids_tensor, attention_mask_tensor)
            return embedding.cpu().numpy().flatten()


if __name__ == "__main__":
    # 测试代码
    embedding = ExpressionEmbedding()
    
    # 生成测试表达式
    test_expressions = [
        nd.Number(1.0),
        nd.Variable('x1'),
        nd.Add(nd.Variable('x1'), nd.Variable('x2')),
        nd.Mul(nd.Sin(nd.Variable('x1')), nd.Number(2.0)),
        nd.Div(nd.Add(nd.Variable('x1'), nd.Variable('x2')), nd.Number(3.0))
    ]
    
    print("测试表达式:")
    for expr in test_expressions:
        print(f"  {expr}")
    
    print("\n开始构建词汇表...")
    embedding.build_vocabulary(test_expressions)
    
    print("\n创建模型...")
    embedding.create_model()
    
    print("\n编码表达式...")
    embeddings = embedding.encode_expressions([str(expr) for expr in test_expressions])
    
    print(f"嵌入向量形状: {embeddings.shape}")
    print(f"第一个嵌入向量: {embeddings[0]}")
