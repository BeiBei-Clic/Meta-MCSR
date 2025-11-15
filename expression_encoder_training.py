import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
from collections import defaultdict

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from expression_encoder import ExpressionEmbedding, ExpressionTokenizer


class ExpressionGenerator:
    """表达式生成器，用于生成训练数据"""
    
    def __init__(self, max_depth=6, max_variables=3):
        self.max_depth = max_depth
        self.max_variables = max_variables
        self.binary_ops = [nd.Add, nd.Sub, nd.Mul, nd.Div]
        self.unary_ops = [nd.Sin, nd.Cos, nd.Sqrt, nd.Log, nd.Exp]
        self.constants = [nd.Number(1), nd.Number(2), nd.Number(0.5), nd.Number(-1), nd.Number(3)]
        self.variables = [nd.Variable(f'x{i+1}') for i in range(max_variables)]
        
    def generate_expression(self, depth=None):
        """生成随机表达式"""
        if depth is None:
            depth = random.randint(2, self.max_depth)
        
        return self._generate_recursive(depth)
    
    def _generate_recursive(self, depth):
        """递归生成表达式"""
        if depth == 0:
            # 叶子节点：变量或常数
            return random.choice(self.variables + self.constants)
        
        choice = random.choice(['variable', 'constant', 'unary', 'binary'])
        
        if choice == 'variable' and depth <= 2:
            return random.choice(self.variables)
        elif choice == 'constant' and depth <= 2:
            return random.choice(self.constants)
        elif choice == 'unary':
            operand = self._generate_recursive(depth - 1)
            op = random.choice(self.unary_ops)
            return op(operand)
        else:  # binary
            left = self._generate_recursive(depth - 1)
            right = self._generate_recursive(depth - 1)
            op = random.choice(self.binary_ops)
            return op(left, right)
    
    def generate_dataset(self, num_expressions=10000):
        """生成数据集"""
        expressions = []
        
        for i in range(num_expressions):
            expr = self.generate_expression()
            expressions.append(str(expr))
            
            if i % 1000 == 0:
                print(f"生成进度: {i}/{num_expressions}")
        
        return expressions
    
    def generate_curriculum_dataset(self, num_expressions=10000):
        """生成课程学习数据集（从简单到复杂）"""
        expressions = []
        
        # 阶段1：简单表达式 (40%)
        simple_count = int(num_expressions * 0.4)
        for i in range(simple_count):
            expr = self._generate_recursive(random.randint(1, 3))
            expressions.append(str(expr))
        
        # 阶段2：中等复杂度表达式 (40%)
        medium_count = int(num_expressions * 0.4)
        for i in range(medium_count):
            expr = self._generate_recursive(random.randint(3, 5))
            expressions.append(str(expr))
        
        # 阶段3：复杂表达式 (20%)
        complex_count = num_expressions - simple_count - medium_count
        for i in range(complex_count):
            expr = self._generate_recursive(random.randint(4, self.max_depth))
            expressions.append(str(expr))
        
        # 打乱数据
        random.shuffle(expressions)
        
        return expressions


class MaskedLanguageModel(nn.Module):
    """掩码语言模型，用于预训练表达式嵌入器"""
    
    def __init__(self, encoder, vocab_size, d_model):
        super().__init__()
        self.encoder = encoder
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, token_ids, attention_mask=None, masked_indices=None):
        """前向传播"""
        # 获取序列级别的嵌入
        sequence_embeddings = self.encoder(token_ids, attention_mask, return_sequence=True)
        
        # 应用MLM头部
        predictions = self.mlm_head(sequence_embeddings)
        
        if masked_indices is not None:
            # 只返回masked位置的预测
            masked_predictions = predictions[masked_indices]
            return masked_predictions
        
        return predictions


class ExpressionPreTrainer:
    """表达式嵌入器预训练器"""
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, 
                 batch_size=32, learning_rate=1e-4, max_seq_length=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        
        # 创建嵌入器和分词器
        self.embedding = ExpressionEmbedding()
        self.tokenizer = ExpressionTokenizer()
        
        # 预训练模型
        self.mlm_model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
    def prepare_training_data(self, expressions):
        """准备训练数据"""
        print("构建词汇表...")
        self.tokenizer.fit(expressions)
        print(f"词汇表大小: {self.tokenizer.vocab_size}")
        
        print("创建嵌入模型...")
        self.embedding.tokenizer = self.tokenizer
        self.embedding.create_model(d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers)
        
        # 创建MLM模型
        self.mlm_model = MaskedLanguageModel(
            self.embedding.model, 
            self.tokenizer.vocab_size, 
            self.d_model
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.mlm_model.parameters(), 
            lr=self.learning_rate
        )
        
    def create_masked_data(self, token_ids_list):
        """创建掩码数据"""
        masked_token_ids = []
        masked_indices = []
        attention_masks = []
        
        # 找到批次中最长的序列长度
        max_seq_len = max(len(ids) for ids in token_ids_list)
        
        for batch_idx, token_ids in enumerate(token_ids_list):
            masked_batch = token_ids.copy()
            
            # 随机选择15%的token进行掩码
            mask_ratio = 0.15
            seq_len = len(token_ids)
            num_masks = max(1, int(seq_len * mask_ratio))
            
            # 选择要掩码的位置（不包括特殊token）
            valid_positions = [i for i in range(1, seq_len - 1) 
                             if token_ids[i] not in [self.tokenizer.token_to_id['<SOS>'], 
                                                   self.tokenizer.token_to_id['<EOS>'], 
                                                   self.tokenizer.token_to_id['<PAD>']]]
            
            if len(valid_positions) > 0:
                masked_positions = random.sample(valid_positions, min(num_masks, len(valid_positions)))
                
                for pos in masked_positions:
                    original_token = token_ids[pos]
                    # 80%概率替换为MASK，10%概率替换为随机token，10%概率保持不变
                    rand = random.random()
                    if rand < 0.8:
                        masked_batch[pos] = self.tokenizer.token_to_id['<MASK>']
                        # 记录被掩码的位置（确保位置在有效范围内）
                        if pos < max_seq_len:  # 确保位置不超过批次中最长序列长度
                            masked_indices.append((batch_idx, pos))
                    elif rand < 0.9:
                        masked_batch[pos] = random.randint(1, self.tokenizer.vocab_size - 4)  # 排除特殊token
                        # 记录被掩码的位置（确保位置在有效范围内）
                        if pos < max_seq_len:  # 确保位置不超过批次中最长序列长度
                            masked_indices.append((batch_idx, pos))
                    # else: 保持不变，不记录
            
            masked_token_ids.append(masked_batch)
            attention_masks.append([True] * len(masked_batch))
        
        return masked_token_ids, attention_masks, masked_indices
    
    def train_epoch(self, expressions, epoch):
        """训练一个epoch"""
        self.mlm_model.train()
        total_loss = 0
        
        # 随机打乱表达式
        random.shuffle(expressions)
        
        # 分批处理
        for i in range(0, len(expressions), self.batch_size):
            batch_exprs = expressions[i:i + self.batch_size]
            
            # Token化
            token_ids_list = []
            for expr in batch_exprs:
                token_ids = self.tokenizer.encode(expr, self.max_seq_length)
                token_ids_list.append(token_ids)
            
            # 创建掩码数据
            masked_token_ids, attention_masks, masked_indices = self.create_masked_data(token_ids_list)
            
            # 转换为tensor
            input_ids = torch.tensor(masked_token_ids, device=self.device)
            attention_mask = torch.tensor(attention_masks, device=self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.mlm_model(input_ids, attention_mask)
            
            # 计算损失
            if len(masked_indices) > 0:
                # 提取被掩码位置的预测和真实标签
                batch_indices = [idx[0] for idx in masked_indices]
                pos_indices = [idx[1] for idx in masked_indices]
                
                # 提取被掩码位置的预测结果
                masked_outputs = outputs[batch_indices, pos_indices]
                masked_input_ids = torch.tensor([token_ids_list[idx[0]][idx[1]] for idx in masked_indices], device=self.device)
                
                loss = self.criterion(masked_outputs, masked_input_ids)
            else:
                # 如果没有掩码位置，跳过这个批次
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mlm_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if i % (self.batch_size * 10) == 0:
                print(f"Epoch {epoch}, Batch {i//self.batch_size}, Loss: {loss.item():.4f}")
        
        return total_loss / (len(expressions) // self.batch_size)
    
    def validate(self, expressions):
        """验证模型"""
        self.mlm_model.eval()
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(expressions), self.batch_size):
                batch_exprs = expressions[i:i + self.batch_size]
                
                token_ids_list = []
                for expr in batch_exprs:
                    token_ids = self.tokenizer.encode(expr, self.max_seq_length)
                    token_ids_list.append(token_ids)
                
                masked_token_ids, attention_masks, masked_indices = self.create_masked_data(token_ids_list)
                
                input_ids = torch.tensor(masked_token_ids, device=self.device)
                attention_mask = torch.tensor(attention_masks, device=self.device)
                
                outputs = self.mlm_model(input_ids, attention_mask)
                
                if len(masked_indices) > 0:
                    batch_indices = [idx[0] for idx in masked_indices]
                    pos_indices = [idx[1] for idx in masked_indices]
                    
                    # 提取被掩码位置的预测结果
                    masked_outputs = outputs[batch_indices, pos_indices]
                    masked_input_ids = torch.tensor([token_ids_list[idx[0]][idx[1]] for idx in masked_indices], device=self.device)
                    
                    loss = self.criterion(masked_outputs, masked_input_ids)
                    total_loss += loss.item()
                    valid_batches += 1
        
        return total_loss / valid_batches if valid_batches > 0 else 0
    
    def train(self, train_expressions, val_expressions=None, num_epochs=50, save_path='weights/expression_encoder'):
        """训练模型"""
        print("开始预训练表达式嵌入器...")
        print(f"训练数据: {len(train_expressions)} 表达式")
        if val_expressions:
            print(f"验证数据: {len(val_expressions)} 表达式")
        
        # 准备训练数据
        self.prepare_training_data(train_expressions)
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(train_expressions, epoch + 1)
            print(f"训练损失: {train_loss:.4f}")
            
            # 验证
            if val_expressions:
                val_loss = self.validate(val_expressions)
                print(f"验证损失: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # 保存最佳模型
                    self.embedding.save_model(save_path)
                    print(f"保存模型到: {save_path}")
            else:
                # 如果没有验证集，每个epoch都保存
                self.embedding.save_model(save_path)
                print(f"保存模型到: {save_path}")
        
        print(f"\n预训练完成！最佳验证损失: {best_val_loss:.4f}")


def main():
    """主函数"""
    print("表达式嵌入器预训练")
    print("=" * 50)
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建表达式生成器
    generator = ExpressionGenerator(max_depth=6, max_variables=3)
    
    # 生成训练数据
    print("生成训练数据...")
    train_expressions = generator.generate_curriculum_dataset(50000)
    
    # 生成验证数据
    print("生成验证数据...")
    val_expressions = generator.generate_curriculum_dataset(5000)
    
    print(f"训练表达式示例:")
    for i, expr in enumerate(train_expressions[:5]):
        print(f"  {i+1}. {expr}")
    
    # 创建预训练器
    trainer = ExpressionPreTrainer(
        d_model=256,
        nhead=8,
        num_layers=6,
        batch_size=64,
        learning_rate=1e-4
    )
    
    # 开始训练
    trainer.train(
        train_expressions=train_expressions,
        val_expressions=val_expressions,
        num_epochs=30,
        save_path='weights/expression_encoder'
    )
    
    print("=" * 50)
    print("预训练完成！")


if __name__ == "__main__":
    main()