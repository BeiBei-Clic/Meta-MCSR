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
            expr = self._generate_recursive(random.randint(1, 2), allow_special=False)
            expressions.append(str(expr))
        
        # 阶段2：中等复杂度表达式 (35%)
        medium_count = int(num_expressions * 0.35)
        print(f"生成中等复杂度表达式: {medium_count:,} 个")
        for i in range(medium_count):
            expr = self._generate_recursive(random.randint(2, 4), allow_special=True)
            expressions.append(str(expr))
        
        # 阶段3：复杂表达式 (25%)
        complex_count = int(num_expressions * 0.25)
        print(f"生成复杂表达式: {complex_count:,} 个")
        for i in range(complex_count):
            expr = self._generate_recursive(random.randint(4, 6), allow_special=True)
            expressions.append(str(expr))
        
        # 阶段4：超复杂表达式 (15%)
        super_count = num_expressions - simple_count - medium_count - complex_count
        print(f"生成超复杂表达式: {super_count:,} 个")
        for i in range(super_count):
            depth = random.randint(5, self.max_depth)
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
    
    def get_parameter_count(self):
        """获取总参数量"""
        count = sum(p.numel() for p in self.parameters())
        return count


class ExpressionPreTrainer:
    """表达式嵌入器预训练器（使用三元组损失，支持100M参数模型和多GPU）"""
    
    def __init__(self, d_model=768, nhead=12, num_layers=12, 
                 batch_size=64, learning_rate=1e-3, max_seq_length=128, margin=0.3,
                 use_distributed=False, rank=0, world_size=1):
        
        # 确保CUDA上下文正确初始化
        if torch.cuda.is_available():
            torch.cuda.init()
            # 预热CUDA
            dummy_tensor = torch.randn(1).cuda()
            del dummy_tensor
            torch.cuda.empty_cache()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # 分布式训练设置
        self.use_distributed = use_distributed
        self.rank = rank
        self.world_size = world_size
        
        # 多GPU训练优化
        self.num_gpus = torch.cuda.device_count()
        
        # 多GPU时自动调整batch_size - 更激进的调整确保并行
        if self.num_gpus > 1:
            # 大幅增加batch_size确保每个GPU都有足够数据处理
            self.batch_size = max(batch_size * self.num_gpus * 2, 64)  # 至少64
            print(f"🚀 多GPU优化: 原始batch_size {batch_size} → 总batch_size {self.batch_size}")
        else:
            self.batch_size = batch_size
            
        # 降低多GPU时的学习率放大倍数，避免梯度不稳定
        # 对于多GPU训练，使用更保守的设置
        lr_multiplier = 1.0 if self.num_gpus <= 1 else 0.5  # 多GPU时降低学习率
        self.learning_rate = learning_rate * lr_multiplier
        self.max_seq_length = max_seq_length
        self.margin = margin
        
        # 多GPU设置信息
        print(f"检测到 {self.num_gpus} 个GPU设备")
        
        if self.num_gpus > 1:
            print("✅ 检测到多GPU环境")
            print(f"  - GPU数量: {self.num_gpus}")
            print(f"  - 设备ID: 0-{self.num_gpus-1}")
            print("  - 将在prepare_training_data中启用数据并行训练")
        else:
            print("单GPU训练模式")
        
        # 创建嵌入器和分词器
        self.embedding = ExpressionEmbedding()
        self.tokenizer = ExpressionTokenizer()
        
        # 预训练模型
        self.triplet_model = None
        self.optimizer = None
        self.criterion = nn.TripletMarginLoss(margin=margin, p=2)
        self.scaler = torch.amp.GradScaler('cuda')  # 混合精度训练
        
        # 梯度监控
        self.grad_norms = []
        self.gradient_clip_val = 20.0  # 进一步提高梯度裁剪阈值
        
    def prepare_training_data(self, expressions):
        """准备训练数据"""
        print("构建词汇表...")
        self.tokenizer.fit(expressions)
        print(f"词汇表大小: {self.tokenizer.vocab_size}")
        
        print("创建嵌入模型...")
        self.embedding.tokenizer = self.tokenizer
        self.embedding.create_model(d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers)
        
        # 确保embedding模型在正确的设备上
        self.embedding.model = self.embedding.model.to(self.device)
        
        # 创建三元组损失模型
        self.triplet_model = TripletLossModel(
            self.embedding.model, 
            margin=self.margin
        ).to(self.device)
        
        print(f"TripletLossModel移动到设备: {self.device}")
        
        # 多GPU/分布式训练包装
        if self.use_distributed:
            # 分布式数据并行
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size)
            
            self.triplet_model = nn.parallel.DistributedDataParallel(
                self.triplet_model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )
            print(f"✅ 启用分布式训练 (rank {self.rank}/{self.world_size})")
            
        elif self.num_gpus > 1:
            # 数据并行训练
            device_ids = list(range(self.num_gpus))
            self.triplet_model = nn.DataParallel(self.triplet_model, device_ids=device_ids)
            print(f"✅ 成功启用多GPU数据并行训练")
            print(f"  - 设备ID: {device_ids}")
            print(f"  - 原始batch_size: {self.batch_size // self.num_gpus}")
            print(f"  - 总batch_size (多GPU): {self.batch_size}")
            print(f"  - 加速比: {self.num_gpus}x")
            
            # 验证设备一致性
            first_param_device = next(self.triplet_model.parameters()).device
            print(f"  - 模型参数设备: {first_param_device}")
            print(f"  - GPU利用率: 启用多GPU并行计算")
        else:
            print(f"单GPU训练，设备: {self.device}")
        
        # 优化器 - 使用AdamW for better training stability
        self.optimizer = torch.optim.AdamW(
            self.triplet_model.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def check_gradients(self):
        """检查梯度是否正常传播"""
        total_norm = 0
        param_count = 0
        
        for p in self.triplet_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.grad_norms.append(total_norm)
            
            # 只保留最近100个梯度范数
            if len(self.grad_norms) > 100:
                self.grad_norms.pop(0)
                
            return total_norm
        return 0
        
    def generate_triplets(self, expressions):
        """生成三元组数据 (anchor, positive, negative) - 改进版本"""
        triplets = []
        
        # 创建表达式复杂度映射，用于更好的负样本选择
        complexity_map = {}
        for expr in expressions:
            complexity = len(expr) + expr.count('(') * 2 + expr.count('*') + expr.count('/')
            complexity_map[expr] = complexity
        
        for i, anchor_expr in enumerate(expressions):
            # 正样本：生成与anchor相似的表达式
            positive_expr = self._generate_similar_expression(anchor_expr)
            
            # 负样本：选择复杂度差异较大的表达式
            anchor_complexity = complexity_map[anchor_expr]
            neg_candidates = []
            
            for j, candidate in enumerate(expressions):
                if j != i:  # 不选择自己
                    candidate_complexity = complexity_map[candidate]
                    # 选择复杂度差异至少20%的作为负样本
                    if abs(candidate_complexity - anchor_complexity) / max(anchor_complexity, 1) > 0.2:
                        neg_candidates.append(candidate)
            
            if neg_candidates:
                negative_expr = random.choice(neg_candidates)
            else:
                # 如果没有找到合适的负样本，使用随机选择
                neg_idx = random.randint(0, len(expressions) - 1)
                if neg_idx == i:
                    neg_idx = (neg_idx + 1) % len(expressions)
                negative_expr = expressions[neg_idx]
            
            triplets.append((anchor_expr, positive_expr, negative_expr))
        
        return triplets
    
    def _generate_similar_expression(self, base_expr):
        """生成与基准表达式相似的表达式"""
        base_str = str(base_expr)
        
        # 简单的相似性变换
        import random
        
        # 避免除零操作，只在安全的变换中使用除法
        # 变量替换
        if 'x1' in base_str and random.random() > 0.3:
            return base_str.replace('x1', 'x2' if 'x2' in base_str else 'x1')
        elif 'x2' in base_str and random.random() > 0.3:
            return base_str.replace('x2', 'x1')
        
        # 常数微调 - 避免产生0或非常小的数
        if '1.0' in base_str:
            new_val = '1.1' if random.random() > 0.5 else '0.8'  # 避免0.9可能导致的数值问题
            return base_str.replace('1.0', new_val)
        elif '2.0' in base_str:
            new_val = '2.1' if random.random() > 0.5 else '1.8'
            return base_str.replace('2.0', new_val)
        elif '3.0' in base_str:
            new_val = '3.1' if random.random() > 0.5 else '2.8'
            return base_str.replace('3.0', new_val)
        
        # 操作符变换 - 谨慎使用除法
        if '+' in base_str:
            return base_str.replace('+', '-', 1)
        elif '-' in base_str:
            return base_str.replace('-', '+', 1)
        # 只有在有足够大常数时才使用除法
        elif ('*' in base_str and '2.0' in base_str) or ('*' in base_str and '3.0' in base_str):
            if random.random() > 0.7:  # 降低除法变换概率
                return base_str.replace('*', '/', 1)
        
        # 如果没有找到可替换的，返回原表达式
        return base_expr
    
    def train_epoch(self, expressions, epoch):
        """训练一个epoch（使用三元组损失和混合精度训练，添加梯度监控）"""
        self.triplet_model.train()
        total_loss = 0
        total_grad_norm = 0
        skipped_batches = 0  # 记录跳过的批次
        
        # 验证输入数据
        if not expressions or len(expressions) == 0:
            print("❌ 错误：没有有效的表达式数据")
            return float('inf')
        
        # 过滤空值和无效表达式
        valid_expressions = [expr for expr in expressions if expr and expr.strip() and len(expr.strip()) > 0]
        if len(valid_expressions) < len(expressions):
            print(f"⚠️ 警告：过滤掉 {len(expressions) - len(valid_expressions)} 个空或无效表达式")
            expressions = valid_expressions
        
        # 检查是否有重复的表达式
        unique_expressions = list(set(expressions))
        if len(unique_expressions) < len(expressions) * 0.8:  # 80%唯一性
            print(f"⚠️ 警告：表达式重复率较高 ({len(expressions) - len(unique_expressions)} 重复)")
        
        # GPU状态监控 - 增强版
        gpu_usage = []
        for i in range(self.num_gpus):
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                gpu_usage.append(memory_allocated)
                print(f"  GPU {i} 内存: {memory_allocated:.2f}GB")
        
        if len(gpu_usage) > 1:
            # 检查GPU负载均衡
            max_memory = max(gpu_usage)
            min_memory = min(gpu_usage)
            if max_memory > min_memory * 2:
                print(f"  ⚠️ GPU负载不均衡: 最高({max_memory:.2f}GB) vs 最低({min_memory:.2f}GB)")
            else:
                print(f"  ✅ GPU负载相对均衡: 最高({max_memory:.2f}GB) vs 最低({min_memory:.2f}GB)")
            
            avg_memory = sum(gpu_usage) / len(gpu_usage)
            print(f"  GPU平均内存使用: {avg_memory:.2f}GB")
            
            # 计算负载差异百分比
            load_diff = (max_memory - min_memory) / max_memory * 100
            print(f"  负载差异: {load_diff:.1f}%")
        elif torch.cuda.is_available():
            print(f"  单GPU模式: {gpu_usage[0]:.2f}GB")
        
        # 生成三元组 - 进一步减少数据量并添加进度显示
        max_training_samples = min(len(expressions), 10000)  # 限制到1万条，5万太慢
        
        if len(expressions) > max_training_samples:
            print(f"⚠️ 数据量过大 ({len(expressions)}), 使用前 {max_training_samples} 条进行训练")
            training_expressions = expressions[:max_training_samples]
        else:
            training_expressions = expressions
        
        print("开始生成三元组...")
        print(f"  生成进度: 0/{len(training_expressions)}", end="")
        
        # 优化三元组生成 - 简化负样本选择算法
        triplets = []
        for i, anchor_expr in enumerate(training_expressions):
            # 正样本：生成与anchor相似的表达式
            positive_expr = self._generate_similar_expression(anchor_expr)
            
            # 简化负样本选择：随机选择但确保不重复
            if len(training_expressions) > 1:
                neg_idx = (i + 1) % len(training_expressions)  # 选择下一个，避免循环
                if neg_idx == i:
                    neg_idx = (i + 2) % len(training_expressions)
            else:
                neg_idx = i  # 只有一个样本时
            
            negative_expr = training_expressions[neg_idx]
            
            triplets.append((anchor_expr, positive_expr, negative_expr))
            
            # 显示进度
            if (i + 1) % 1000 == 0:
                print(f"\r  生成进度: {i+1}/{len(training_expressions)}", end="")
        
        print(f"\r  生成进度: {len(training_expressions)}/{len(training_expressions)} ✅")
        
        # 验证三元组质量 - 简化验证
        valid_triplets = []
        skipped = 0
        
        for anchor, positive, negative in triplets:
            # 简化的有效性检查
            if (anchor != positive and anchor != negative and positive != negative):
                valid_triplets.append((anchor, positive, negative))
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"  过滤无效三元组: {skipped} 个")
        print(f"  ✅ 有效三元组: {len(valid_triplets)} 个")
        
        triplets = valid_triplets
        
        # 随机打乱
        random.shuffle(triplets)
        
        if len(triplets) == 0:
            print("❌ 错误：没有有效的三元组数据")
            return float('inf')
        
        print(f"开始训练: {len(triplets)} 个三元组，批次大小: {self.batch_size}，预计批次: {(len(triplets) + self.batch_size - 1) // self.batch_size}")
        print("📊 准备开始训练...（首次batch可能需要较长时间进行DataParallel初始化）")
        
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
            
            try:
                # 在forward前检查GPU使用情况
                gpu_usage_before = []
                for j in range(self.num_gpus):
                    memory_allocated = torch.cuda.memory_allocated(j) / 1024**3
                    gpu_usage_before.append(memory_allocated)
                
                with torch.amp.autocast('cuda'):
                    anchor_emb, positive_emb, negative_emb = self.triplet_model(
                        anchor_tensor, positive_tensor, negative_tensor,
                        anchor_mask_tensor, positive_mask_tensor, negative_mask_tensor
                    )
                    
                    # 在forward后检查GPU使用情况
                    gpu_usage_after = []
                    for j in range(self.num_gpus):
                        memory_allocated = torch.cuda.memory_allocated(j) / 1024**3
                        gpu_usage_after.append(memory_allocated)
                    
                    # 检查是否有GPU真正参与计算
                    if num_batches % 10 == 0:  # 每10个批次显示一次
                        print(f"  GPU负载检查: {[f'{m:.2f}' for m in gpu_usage_after]}")
                    
                    # 计算三元组损失
                    loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                    
                    # 检查损失有效性
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ 损失无效，skip批次: {loss}")
                        self.optimizer.zero_grad()
                        skipped_batches += 1
                        continue
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 检查梯度 - 在裁剪前检查
                grad_norm = self.check_gradients()
                
                # 检查是否有梯度爆炸 - 更严格的检查
                if grad_norm != grad_norm or grad_norm == float('inf') or grad_norm > 50000:  # 提高阈值到5万
                    print(f"⚠️ 梯度异常: {grad_norm:.4f}, 跳过此批次")
                    self.optimizer.zero_grad()  # 清零梯度，避免累积
                    skipped_batches += 1
                    continue
                
                # 梯度裁剪 - 使用更严格的裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.triplet_model.parameters(), max_norm=min(self.gradient_clip_val, grad_norm * 0.8))
                
                # 只在梯度正常时更新
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # 记录有效的损失和梯度
                total_loss += loss.item()
                total_grad_norm += grad_norm
                num_batches += 1
                
                if i % (self.batch_size * 20) == 0:  # 更频繁的日志更新
                    avg_grad = total_grad_norm / num_batches if num_batches > 0 else 0
                    
                    # 实时GPU内存监控
                    if hasattr(self.triplet_model, 'module') and torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated(0) / 1024**3
                        print(f"Epoch {epoch}, Batch {i//self.batch_size}, Loss: {loss.item():.4f}, "
                              f"Grad Norm: {avg_grad:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                              f"Memory: {current_memory:.2f}GB")
                    else:
                        print(f"Epoch {epoch}, Batch {i//self.batch_size}, Loss: {loss.item():.4f}, "
                              f"Grad Norm: {avg_grad:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                    
                    # 内存警告
                    if torch.cuda.is_available() and torch.cuda.memory_allocated(0) / 1024**3 > 8:
                        print(f"⚠️ 内存使用过高 ({current_memory:.2f}GB)，建议降低batch_size")
                          
            except Exception as e:
                print(f"批次训练出错: {e}")
                skipped_batches += 1
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0
        
        if num_batches > 0:
            print(f"  有效训练批次: {num_batches}")
            print(f"  跳过的异常批次: {skipped_batches}")
            print(f"  平均损失: {avg_loss:.4f}")
            print(f"  平均梯度范数: {avg_grad_norm:.4f}")
        else:
            print("❌ 没有有效的训练批次")
        
        return avg_loss
    
    def validate(self, expressions):
        """验证模型（使用三元组损失）"""
        self.triplet_model.eval()
        total_loss = 0
        
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
        
        return total_loss / (len(triplets) // self.batch_size)
    
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
            
            # 每次epoch前检查GPU状态
            if epoch == 0:
                print(f"🎯 多GPU训练状态检查:")
                if hasattr(self.triplet_model, 'module'):
                    print(f"  ✅ DataParallel已激活")
                    print(f"  GPU数量: {self.num_gpus}")
                    print(f"  当前设备: {next(self.triplet_model.parameters()).device}")
                else:
                    print(f"  ⚠️ 未检测到DataParallel")
            
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
        
        if torch.cuda.device_count() > 1:
            print("🔥 多GPU检测成功！将使用所有可用GPU进行训练")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("📱 单GPU模式")
        
    # 创建表达式生成器
    generator = ExpressionGenerator(max_depth=8, max_variables=6)
    
    # 先生成小量测试数据验证代码
    print("\n=== 步骤1：小规模测试数据验证 ===")
    test_train_expressions = generator.generate_curriculum_dataset(1000)
    test_val_expressions = generator.generate_curriculum_dataset(200)
    
    print(f"测试训练表达式示例:")
    for i, expr in enumerate(test_train_expressions[:5]):
        print(f"  {i+1}. {expr}")
    
    # 创建测试预训练器 - 与大规模训练保持一致
    test_trainer = ExpressionPreTrainer(
        d_model=256,   # 与大规模训练保持一致
        nhead=8,       # 与大规模训练保持一致
        num_layers=6,  # 与大规模训练保持一致
        batch_size=16, # 小batch size用于测试
        learning_rate=5e-4,  # 与大规模训练保持一致
        margin=0.3,
        use_distributed=False,
        rank=0,
        world_size=1
    )
    
    print("\n开始小规模测试训练...")
    test_trainer.prepare_training_data(test_train_expressions)
    
    # 运行一个测试epoch
    print("运行测试epoch...")
    test_loss = test_trainer.train_epoch(test_train_expressions, 1)
    print(f"测试epoch完成，平均损失: {test_loss:.4f}")
    
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
    
    # 创建大规模预训练器 - 简化模型避免梯度爆炸
    trainer = ExpressionPreTrainer(
        d_model=256,   # 降低模型维度
        nhead=8,       # 减少注意力头
        num_layers=6,  # 减少层数
        batch_size=16, # 进一步减少batch size
        learning_rate=5e-4,  # 显著降低学习率
        margin=0.3,
        use_distributed=False,
        rank=0,
        world_size=1
    )
    
    print(f"\n模型配置:")
    print(f"  - 隐藏维度: {trainer.d_model}")
    print(f"  - 注意力头数: {trainer.nhead}")
    print(f"  - 层数: {trainer.num_layers}")
    print(f"  - 批量大小: {trainer.batch_size}")
    print(f"  - 学习率: {trainer.learning_rate:.2e}")
    print(f"  - 检测到的GPU数量: {trainer.num_gpus}")
    print(f"  - 预期参数量: ~{trainer.d_model * trainer.nhead * trainer.num_layers * 4 / 1e6:.1f}M")
    
    if trainer.num_gpus > 1:
        print(f"🚀 多GPU并行训练配置")
        print(f"  - GPU数量: {trainer.num_gpus}")
        print(f"  - 数据并行: nn.DataParallel")
        print(f"  - 预期加速比: {trainer.num_gpus}x")
    else:
        print(f"📱 单GPU训练")
    
    # 开始大规模训练
    print(f"\n开始大规模训练 (共 {len(train_expressions):,} 个表达式)...")
    
    # 验证多GPU状态
    if hasattr(trainer.triplet_model, 'module'):
        print(f"✅ 多GPU训练已激活 - DataParallel模式")
        print(f"  - 主设备: {next(trainer.triplet_model.parameters()).device}")
        print(f"  - 副本数量: {len(trainer.triplet_model.device_ids)}")
        
        # 强制检查GPU负载
        print(f"  - 强制GPU负载检查:")
        for i in range(trainer.num_gpus):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"    GPU {i}: {memory_allocated:.2f}GB")
            
    else:
        print(f"📱 单GPU训练模式")
    
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
    
    # 显示多GPU性能统计
    if trainer.num_gpus > 1:
        print(f"🚀 多GPU训练性能总结:")
        print(f"  - GPU数量: {trainer.num_gpus}")
        print(f"  - 训练类型: 数据并行 (DataParallel)")
        if hasattr(trainer.triplet_model, 'module'):
            print(f"  - GPU利用率: ✅ 已激活")
            print(f"  - 副本设备: {trainer.triplet_model.device_ids}")
    
    print("=" * 60)
    print("大规模预训练完成！")


if __name__ == "__main__":
    main()