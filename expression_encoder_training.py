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
    """扩展的表达式生成器，用于生成更多样化的训练数据"""
    
    def __init__(self, max_depth=8, max_variables=6):
        self.max_depth = max_depth
        self.max_variables = max_variables
        
        # 二元运算符
        self.binary_ops = [
            nd.Add, nd.Sub, nd.Mul, nd.Div, 
            nd.Pow  # 添加更多二元运算符
        ]
        
        # 一元运算符
        self.unary_ops = [
            nd.Sin, nd.Cos, nd.Tan, nd.Exp, nd.Log, nd.Sqrt,
            nd.Abs, nd.Neg  # 添加更多一元运算符
        ]
        
        # 扩展常数集合
        self.constants = [
            nd.Number(0), nd.Number(1), nd.Number(2), nd.Number(3), nd.Number(4),
            nd.Number(0.5), nd.Number(-1), nd.Number(0.25), nd.Number(0.75),
            nd.Number(1.5), nd.Number(2.5), nd.Number(3.14), nd.Number(-0.5),
            nd.Number(1.618), nd.Number(2.718)  # 添加数学常数
        ]
        
        # 变量
        self.variables = [nd.Variable(f'x{i+1}') for i in range(max_variables)]
        
        # 特殊函数（多参数函数）
        self.special_functions = [nd.Min, nd.Max]  # 根据nd2py的可用函数
        
    def generate_expression(self, depth=None):
        """生成随机表达式"""
        if depth is None:
            depth = random.randint(2, self.max_depth)
        
        return self._generate_recursive(depth)
    
    def _generate_recursive(self, depth, allow_special=False):
        """递归生成表达式（增强版）"""
        if depth == 0:
            # 叶子节点：变量或常数
            choice = random.choice(['variable', 'constant'])
            if choice == 'variable':
                return random.choice(self.variables)
            else:
                return random.choice(self.constants)
        
        # 权重化选择：根据深度调整选择概率
        if depth <= 2:
            choices = ['variable', 'constant', 'unary', 'binary', 'special']
            weights = [0.3, 0.3, 0.2, 0.15, 0.05]
        else:
            choices = ['unary', 'binary', 'special']
            weights = [0.4, 0.5, 0.1]
        
        if not allow_special:
            choices = [c for c in choices if c != 'special']
            weights = weights[:len(choices)]
        
        choice = random.choices(choices, weights=weights)[0]
        
        if choice == 'variable' and depth <= 2:
            return random.choice(self.variables)
        elif choice == 'constant' and depth <= 2:
            return random.choice(self.constants)
        elif choice == 'unary':
            operand = self._generate_recursive(depth - 1, allow_special)
            op = random.choice(self.unary_ops)
            return op(operand)
        elif choice == 'binary':
            left = self._generate_recursive(depth - 1, allow_special)
            right = self._generate_recursive(depth - 1, allow_special)
            op = random.choice(self.binary_ops)
            return op(left, right)
        else:  # special function
            if self.special_functions and allow_special:
                func = random.choice(self.special_functions)
                # 为特殊函数生成2-3个参数
                num_args = random.randint(2, 3)
                args = [self._generate_recursive(max(1, depth - 2), allow_special) for _ in range(num_args)]
                return func(*args)
            else:
                # 回退到二元操作
                left = self._generate_recursive(depth - 1, allow_special)
                right = self._generate_recursive(depth - 1, allow_special)
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
    
    def generate_curriculum_dataset(self, num_expressions=100000):
        """生成大规模课程学习数据集"""
        expressions = []
        
        print(f"开始生成 {num_expressions:,} 个表达式...")
        
        # 阶段1：简单表达式 (25%)
        simple_count = int(num_expressions * 0.25)
        print(f"生成简单表达式: {simple_count:,} 个")
        for i in range(simple_count):
            if i % 10000 == 0:
                print(f"简单表达式进度: {i:,}/{simple_count:,}")
            expr = self._generate_recursive(random.randint(1, 2), allow_special=False)
            expressions.append(str(expr))
        
        # 阶段2：中等复杂度表达式 (35%)
        medium_count = int(num_expressions * 0.35)
        print(f"生成中等复杂度表达式: {medium_count:,} 个")
        for i in range(medium_count):
            if i % 10000 == 0:
                print(f"中等复杂度表达式进度: {i:,}/{medium_count:,}")
            expr = self._generate_recursive(random.randint(2, 4), allow_special=True)
            expressions.append(str(expr))
        
        # 阶段3：复杂表达式 (25%)
        complex_count = int(num_expressions * 0.25)
        print(f"生成复杂表达式: {complex_count:,} 个")
        for i in range(complex_count):
            if i % 10000 == 0:
                print(f"复杂表达式进度: {i:,}/{complex_count:,}")
            expr = self._generate_recursive(random.randint(4, 6), allow_special=True)
            expressions.append(str(expr))
        
        # 阶段4：超复杂表达式 (15%)
        super_count = num_expressions - simple_count - medium_count - complex_count
        print(f"生成超复杂表达式: {super_count:,} 个")
        for i in range(super_count):
            if i % 10000 == 0:
                print(f"超复杂表达式进度: {i:,}/{super_count:,}")
            max_depth_for_super = max(5, self.max_depth)
            depth = random.randint(5, max_depth_for_super) if max_depth_for_super >= 5 else self.max_depth
            expr = self._generate_recursive(depth, allow_special=True)
            expressions.append(str(expr))
        
        # 打乱数据
        random.shuffle(expressions)
        
        print(f"数据集生成完成！总计 {len(expressions):,} 个表达式")
        return expressions
    
    def generate_specialized_datasets(self):
        """生成专门的表达式数据集"""
        datasets = {}
        
        # 线性表达式
        print("生成线性表达式数据集...")
        linear_exprs = []
        for _ in range(10000):
            expr = nd.Add(
                nd.Mul(nd.Variable('x1'), random.choice([nd.Number(i) for i in range(-5, 6) if i != 0])),
                nd.Mul(nd.Variable('x2'), random.choice([nd.Number(i) for i in range(-5, 6) if i != 0])),
                nd.Number(random.randint(-10, 10))
            )
            linear_exprs.append(str(expr))
        
        # 多项式表达式
        print("生成多项式表达式数据集...")
        polynomial_exprs = []
        for _ in range(10000):
            degree = random.randint(2, 4)
            expr = nd.Variable('x1')
            for i in range(2, degree + 1):
                coeff = nd.Number(random.randint(1, 5))
                power = nd.Pow(nd.Variable('x1'), nd.Number(i))
                term = nd.Mul(coeff, power)
                expr = nd.Add(expr, term)
            polynomial_exprs.append(str(expr))
        
        # 三角函数表达式
        print("生成三角函数表达式数据集...")
        trig_exprs = []
        for _ in range(10000):
            func = random.choice([nd.Sin, nd.Cos, nd.Tan])
            arg = nd.Add(
                nd.Mul(nd.Variable('x1'), nd.Number(random.randint(1, 5))),
                nd.Number(random.randint(-10, 10))
            )
            expr = func(arg)
            trig_exprs.append(str(expr))
        
        datasets['linear'] = linear_exprs
        datasets['polynomial'] = polynomial_exprs
        datasets['trigonometric'] = trig_exprs
        
        return datasets


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
    """表达式嵌入器预训练器（使用三元组损失，支持100M参数模型和多GPU）"""
    
    def __init__(self, d_model=768, nhead=12, num_layers=12, 
                 batch_size=64, learning_rate=1e-4, max_seq_length=128, margin=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.margin = margin
        
        # 多GPU设置 - 暂时禁用多GPU以避免设备不匹配问题
        self.num_gpus = 1  # torch.cuda.device_count()
        print(f"检测到 {torch.cuda.device_count()} 个GPU设备，但为避免设备问题暂时使用单GPU训练")
        
        # 创建嵌入器和分词器
        self.embedding = ExpressionEmbedding()
        self.tokenizer = ExpressionTokenizer()
        
        # 预训练模型
        self.triplet_model = None
        self.optimizer = None
        self.criterion = nn.TripletMarginLoss(margin=margin, p=2)
        self.scaler = torch.cuda.amp.GradScaler()  # 混合精度训练
        
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
        
        # 多GPU包装
        if self.num_gpus > 1:
            self.triplet_model = nn.DataParallel(self.triplet_model)
            print(f"已启用多GPU数据并行训练，batch_size自动调整为 {self.batch_size * self.num_gpus}")
        
        # 优化器 - 使用不同的学习率
        self.optimizer = torch.optim.AdamW(
            self.triplet_model.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
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
        """训练一个epoch（使用三元组损失和混合精度训练）"""
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
            
            # 混合精度训练
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                anchor_emb, positive_emb, negative_emb = self.triplet_model(
                    anchor_tensor, positive_tensor, negative_tensor,
                    anchor_mask_tensor, positive_mask_tensor, negative_mask_tensor
                )
                
                # 计算三元组损失
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.triplet_model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % (self.batch_size * 5) == 0:
                print(f"Epoch {epoch}, Batch {i//self.batch_size}, Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
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
    """主函数 - 100M参数表达式嵌入器预训练"""
    print("大规模表达式嵌入器预训练 (100M参数)")
    print("=" * 60)
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 设置GPU内存优化
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    # 创建表达式生成器
    generator = ExpressionGenerator(max_depth=8, max_variables=6)
    
    # 先生成小量测试数据验证代码
    print("\n=== 步骤1：小规模测试数据验证 ===")
    test_train_expressions = generator.generate_curriculum_dataset(1000)
    test_val_expressions = generator.generate_curriculum_dataset(200)
    
    print(f"测试训练表达式示例:")
    for i, expr in enumerate(test_train_expressions[:5]):
        print(f"  {i+1}. {expr}")
    
    # 创建测试预训练器
    test_trainer = ExpressionPreTrainer(
        d_model=768,
        nhead=12,
        num_layers=12,
        batch_size=16,  # 小batch size用于测试
        learning_rate=5e-5,  # 更小的学习率用于大模型
        margin=0.3
    )
    
    print("\n开始小规模测试训练...")
    test_trainer.prepare_training_data(test_train_expressions)
    
    # 运行一个测试epoch
    print("运行测试epoch...")
    test_loss = test_trainer.train_epoch(test_train_expressions, 1)
    print(f"测试epoch完成，平均损失: {test_loss:.4f}")
    
    if test_loss > 10.0:  # 如果损失异常高
        print("警告：测试损失过高，可能存在数值稳定性问题")
        return
    
    print("✅ 小规模测试通过！")
    
    # 如果测试通过，询问是否继续大规模训练
    print("\n=== 步骤2：大规模训练 ===")
    user_input = input("小规模测试通过！是否继续大规模训练？(y/N): ").strip().lower()
    if user_input not in ['y', 'yes', '是']:
        print("大规模训练已取消")
        return
    
    # 生成大规模训练数据
    print("\n生成大规模训练数据...")
    train_expressions = generator.generate_curriculum_dataset(200000)
    val_expressions = generator.generate_curriculum_dataset(20000)
    
    # 生成专门数据集
    print("\n生成专门化数据集...")
    specialized_datasets = generator.generate_specialized_datasets()
    for name, exprs in specialized_datasets.items():
        print(f"{name} 表达式数量: {len(exprs):,}")
        train_expressions.extend(exprs)
    
    print(f"总训练表达式数量: {len(train_expressions):,}")
    
    # 打乱所有训练数据
    random.shuffle(train_expressions)
    
    # 创建大规模预训练器
    trainer = ExpressionPreTrainer(
        d_model=768,
        nhead=12,
        num_layers=12,
        batch_size=128,  # 大batch size用于训练
        learning_rate=5e-5,
        margin=0.3
    )
    
    print(f"\n模型配置:")
    print(f"  - 隐藏维度: {trainer.d_model}")
    print(f"  - 注意力头数: {trainer.nhead}")
    print(f"  - 层数: {trainer.num_layers}")
    print(f"  - 批量大小: {trainer.batch_size}")
    print(f"  - 学习率: {trainer.learning_rate}")
    print(f"  - GPU数量: {trainer.num_gpus}")
    
    # 开始大规模训练
    print(f"\n开始大规模训练 (共 {len(train_expressions):,} 个表达式)...")
    trainer.train(
        train_expressions=train_expressions,
        val_expressions=val_expressions,
        num_epochs=50,
        save_path='weights/expression_encoder'
    )
    
    # 保存最终模型
    final_path = 'weights/expression_encoder'
    trainer.embedding.save_model(final_path)
    print(f"最终模型已保存到: {final_path}")
    
    # 显示模型参数统计
    if hasattr(trainer.embedding.model, 'module'):
        param_count = trainer.embedding.model.module.get_parameter_count()
    else:
        param_count = trainer.embedding.model.get_parameter_count()
    print(f"最终模型参数量: {param_count:,} ({param_count/1e6:.1f}M)")
    
    print("=" * 60)
    print("大规模预训练完成！")


if __name__ == "__main__":
    main()