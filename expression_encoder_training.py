import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
from collections import defaultdict

# 添加必要的路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from src.expression_encoder import ExpressionEmbedding, ExpressionTokenizer


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


class TripletLossModel(nn.Module):
    """三元组损失模型，用于对比学习表达式嵌入器"""
    
    def __init__(self, encoder, margin=0.5):
        super().__init__()
        self.encoder = encoder
        self.margin = margin
        
    def forward(self, anchor_ids, positive_ids, negative_ids, 
                anchor_mask=None, positive_mask=None, negative_mask=None):
        """前向传播"""
        # 编码anchor、positive和negative
        anchor_emb = self.encoder(anchor_ids, anchor_mask)
        positive_emb = self.encoder(positive_ids, positive_mask)
        negative_emb = self.encoder(negative_ids, negative_mask)
        
        return anchor_emb, positive_emb, negative_emb


class ExpressionPreTrainer:
    """表达式嵌入器预训练器（使用三元组损失）"""
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, 
                 batch_size=32, learning_rate=1e-4, max_seq_length=128, margin=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.margin = margin
        
        # 创建嵌入器和分词器
        self.embedding = ExpressionEmbedding()
        self.tokenizer = ExpressionTokenizer()
        
        # 预训练模型
        self.triplet_model = None
        self.optimizer = None
        self.criterion = nn.TripletMarginLoss(margin=margin, p=2)
        
    def prepare_training_data(self, expressions):
        """准备训练数据"""
        print("构建词汇表...")
        self.tokenizer.fit(expressions)
        print(f"词汇表大小: {self.tokenizer.vocab_size}")
        
        print("创建嵌入模型...")
        self.embedding.tokenizer = self.tokenizer
        self.embedding.create_model(d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers)
        
        # 创建三元组损失模型
        self.triplet_model = TripletLossModel(
            self.embedding.model, 
            margin=self.margin
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.triplet_model.parameters(), 
            lr=self.learning_rate
        )
        
    def generate_triplets(self, expressions):
        """生成三元组数据 (anchor, positive, negative)"""
        triplets = []
        
        for i, anchor_expr in enumerate(expressions):
            # 正样本：相同或相似的表达式
            # 这里简化处理：使用相同的表达式作为正样本
            # 实际应用中可以生成简化版本或代数变换版本
            if i + 1 < len(expressions):
                positive_expr = expressions[i + 1]
            else:
                positive_expr = anchor_expr
            
            # 负样本：随机选择不同的表达式
            neg_idx = random.randint(0, len(expressions) - 1)
            while neg_idx == i:
                neg_idx = random.randint(0, len(expressions) - 1)
            negative_expr = expressions[neg_idx]
            
            triplets.append((anchor_expr, positive_expr, negative_expr))
        
        return triplets
    
    def train_epoch(self, expressions, epoch):
        """训练一个epoch（使用三元组损失）"""
        self.triplet_model.train()
        total_loss = 0
        
        # 生成三元组
        triplets = self.generate_triplets(expressions)
        
        # 随机打乱
        random.shuffle(triplets)
        
        # 分批处理
        num_batches = 0
        for i in range(0, len(triplets), self.batch_size):
            batch_triplets = triplets[i:i + self.batch_size]
            
            # Token化
            anchor_ids_list = []
            positive_ids_list = []
            negative_ids_list = []
            anchor_masks = []
            positive_masks = []
            negative_masks = []
            
            for anchor, positive, negative in batch_triplets:
                # Anchor
                anchor_ids = self.tokenizer.encode(anchor, self.max_seq_length)
                anchor_ids_list.append(anchor_ids)
                anchor_masks.append([True] * len(anchor_ids))
                
                # Positive
                positive_ids = self.tokenizer.encode(positive, self.max_seq_length)
                positive_ids_list.append(positive_ids)
                positive_masks.append([True] * len(positive_ids))
                
                # Negative
                negative_ids = self.tokenizer.encode(negative, self.max_seq_length)
                negative_ids_list.append(negative_ids)
                negative_masks.append([True] * len(negative_ids))
            
            # 转换为tensor
            anchor_tensor = torch.tensor(anchor_ids_list, device=self.device)
            positive_tensor = torch.tensor(positive_ids_list, device=self.device)
            negative_tensor = torch.tensor(negative_ids_list, device=self.device)
            
            anchor_mask_tensor = torch.tensor(anchor_masks, device=self.device)
            positive_mask_tensor = torch.tensor(positive_masks, device=self.device)
            negative_mask_tensor = torch.tensor(negative_masks, device=self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            anchor_emb, positive_emb, negative_emb = self.triplet_model(
                anchor_tensor, positive_tensor, negative_tensor,
                anchor_mask_tensor, positive_mask_tensor, negative_mask_tensor
            )
            
            # 计算三元组损失
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.triplet_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % (self.batch_size * 10) == 0:
                print(f"Epoch {epoch}, Batch {i//self.batch_size}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, expressions):
        """验证模型（使用三元组损失）"""
        self.triplet_model.eval()
        total_loss = 0
        valid_batches = 0
        
        # 生成三元组
        triplets = self.generate_triplets(expressions)
        
        with torch.no_grad():
            for i in range(0, len(triplets), self.batch_size):
                batch_triplets = triplets[i:i + self.batch_size]
                
                # Token化
                anchor_ids_list = []
                positive_ids_list = []
                negative_ids_list = []
                anchor_masks = []
                positive_masks = []
                negative_masks = []
                
                for anchor, positive, negative in batch_triplets:
                    anchor_ids = self.tokenizer.encode(anchor, self.max_seq_length)
                    anchor_ids_list.append(anchor_ids)
                    anchor_masks.append([True] * len(anchor_ids))
                    
                    positive_ids = self.tokenizer.encode(positive, self.max_seq_length)
                    positive_ids_list.append(positive_ids)
                    positive_masks.append([True] * len(positive_ids))
                    
                    negative_ids = self.tokenizer.encode(negative, self.max_seq_length)
                    negative_ids_list.append(negative_ids)
                    negative_masks.append([True] * len(negative_ids))
                
                anchor_tensor = torch.tensor(anchor_ids_list, device=self.device)
                positive_tensor = torch.tensor(positive_ids_list, device=self.device)
                negative_tensor = torch.tensor(negative_ids_list, device=self.device)
                
                anchor_mask_tensor = torch.tensor(anchor_masks, device=self.device)
                positive_mask_tensor = torch.tensor(positive_masks, device=self.device)
                negative_mask_tensor = torch.tensor(negative_masks, device=self.device)
                
                anchor_emb, positive_emb, negative_emb = self.triplet_model(
                    anchor_tensor, positive_tensor, negative_tensor,
                    anchor_mask_tensor, positive_mask_tensor, negative_mask_tensor
                )
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
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
                    # 保存检查点模型
                    checkpoint_path = f'checkpoints/expression_encoder/best_epoch_{epoch+1}'
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    self.embedding.save_model(checkpoint_path)
                    print(f"保存检查点到: {checkpoint_path}")
            else:
                # 如果没有验证集，定期保存检查点
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = f'checkpoints/expression_encoder/epoch_{epoch+1}'
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    self.embedding.save_model(checkpoint_path)
                    print(f"保存检查点到: {checkpoint_path}")
        
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
        save_path='weights/expression_encoder'  # 这个参数现在只用于参考
    )
    
    # 保存最终模型到weights文件夹
    final_path = 'weights/expression_encoder'
    trainer.embedding.save_model(final_path)
    print(f"最终模型已保存到: {final_path}")
    
    # 清理weights文件夹，只保留推理必需的文件
    print("清理weights文件夹...")
    try:
        import subprocess
        import os
        result = subprocess.run(['python3', 'tools/clean_weights.py', '--force'], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print("weights文件夹清理完成！")
        else:
            print("警告：weights文件夹清理失败")
    except Exception as e:
        print(f"警告：无法运行清理工具 - {e}")
    
    print("=" * 50)
    print("预训练完成！")


if __name__ == "__main__":
    main()