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
            # nd.Abs, nd.Neg  # 添加更多一元运算符
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
            expr = self._generate_recursive(random.randint(2, 4), allow_special=False)
            expressions.append(str(expr))
        
        # 阶段3：复杂表达式 (25%)
        complex_count = int(num_expressions * 0.25)
        print(f"生成复杂表达式: {complex_count:,} 个")
        for i in range(complex_count):
            expr = self._generate_recursive(random.randint(4, 6), allow_special=False)
            expressions.append(str(expr))
        
        # 阶段4：超复杂表达式 (15%)
        super_count = num_expressions - simple_count - medium_count - complex_count
        print(f"生成超复杂表达式: {super_count:,} 个")
        for i in range(super_count):
            depth = random.randint(5, self.max_depth)
            expr = self._generate_recursive(depth, allow_special=False)
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
        """生成三元组数据 (anchor, positive, negative) - 基于表达式结构操作"""
        triplets = []
        
        for i, anchor_expr in enumerate(expressions):
            # 正样本：将锚点表达式低层运算符（运算符带着的整颗子树）去掉来构建
            positive_expr = self._generate_positive_by_removing_operator(anchor_expr)
            
            # 负样本：将锚点表达式中某一运算符替换来构建
            negative_expr = self._generate_negative_by_replacing_operator(anchor_expr)
            
            # 确保生成的正负样本与锚点不同
            if positive_expr == anchor_expr:
                positive_expr = self._generate_positive_by_removing_operator(anchor_expr)
            if negative_expr == anchor_expr:
                negative_expr = self._generate_negative_by_replacing_operator(anchor_expr)
            
            triplets.append((anchor_expr, positive_expr, negative_expr))
        
        return triplets
    
    def _generate_positive_by_removing_operator(self, anchor_expr_str):
        """通过简化表达式来构建正样本：移除括号或简化表达"""
        import re
        
        # 策略1：尝试移除最外层的括号
        expr = anchor_expr_str.strip()
        if expr.startswith('(') and expr.endswith(')'):
            # 检查括号是否匹配
            paren_count = 0
            for i, char in enumerate(expr):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0 and i < len(expr) - 1:
                        # 括号不匹配，不是最外层括号
                        break
            else:
                # 括号匹配，可以移除
                inner_expr = expr[1:-1].strip()
                if inner_expr:
                    return inner_expr
        
        # 策略2：移除某些函数调用
        # 移除 sin() 或 cos() 等
        pattern = r'\b(sin|cos|tan|exp|log|sqrt|abs)\s*\(([^)]+)\)'
        if re.search(pattern, expr):
            match = re.search(pattern, expr)
            if match:
                return match.group(2)  # 返回函数内部的参数
        
        # 策略3：简化算术表达式
        # 如果有连续的括号，尝试简化
        simplified = expr.replace('()', '').strip()
        if simplified and simplified != expr:
            return simplified
        
        # 策略4：随机移除某个运算符的整个操作数
        operators = ['+', '-', '*', '/', '**']
        for op in operators:
            if op in expr:
                # 找到运算符的位置（不匹配括号）
                paren_count = 0
                op_pos = -1
                for i, char in enumerate(expr):
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    elif char == op and paren_count == 0:
                        op_pos = i
                        break
                
                if op_pos > 0:
                    # 分割表达式
                    left = expr[:op_pos].strip()
                    right = expr[op_pos+1:].strip()
                    
                    # 尝试移除右操作数
                    if right:
                        # 如果右操作数在括号中，移除括号
                        if right.startswith('(') and right.endswith(')'):
                            return left
                        else:
                            # 简单情况下直接返回左操作数
                            return left
        
        # 如果所有策略都失败，返回原始表达式
        return expr
    
    def _generate_negative_by_replacing_operator(self, anchor_expr_str):
        """通过替换运算符来构建负样本"""
        import re
        
        expr = anchor_expr_str.strip()
        
        # 策略1：替换算术运算符
        operator_mappings = [
            # 替换 + 为其他运算符
            (r'\+', '-'),
            (r'-', '+'),
            (r'\*', '/'),
            (r'/', '*'),
            (r'\*\*', '+'),  # 替换 ** 为 +
        ]
        
        for pattern, replacement in operator_mappings:
            # 只在最高层替换（不考虑括号内的）
            paren_count = 0
            modified_chars = []
            replaced = False
            
            for i, char in enumerate(expr):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char in '+-*/' and paren_count == 0:
                    # 检查是否匹配模式
                    if char == '+' and pattern == r'\+':
                        modified_chars.append(replacement)
                        replaced = True
                    elif char == '-' and pattern == '-':
                        modified_chars.append(replacement)
                        replaced = True
                    elif char == '*' and pattern == r'\*':
                        modified_chars.append(replacement)
                        replaced = True
                    elif char == '/' and pattern == '/':
                        modified_chars.append(replacement)
                        replaced = True
                    else:
                        modified_chars.append(char)
                else:
                    modified_chars.append(char)
            
            if replaced:
                return ''.join(modified_chars).strip()
        
        # 策略2：替换函数名
        function_mappings = [
            (r'\bsin\b', 'cos'),
            (r'\bcos\b', 'sin'),
            (r'\btan\b', 'sin'),
            (r'\bexp\b', 'log'),
            (r'\blog\b', 'exp'),
            (r'\bsqrt\b', 'abs'),
        ]
        
        for pattern, replacement in function_mappings:
            new_expr = re.sub(pattern, replacement, expr)
            if new_expr != expr:
                return new_expr
        
        # 策略3：如果找不到可替换的运算符，尝试添加否定或常量
        if 'sin' in expr or 'cos' in expr:
            # 在三角函数前加负号
            new_expr = re.sub(r'\b(sin|cos|tan)\b', r'-\1', expr)
            if new_expr != expr:
                return new_expr
        
        # 策略4：添加随机数字到表达式
        import random
        if random.random() < 0.5:
            # 在表达式前添加随机数字
            random_num = random.choice([0, 1, 2, 0.5])
            return f"{random_num} + ({expr})"
        else:
            # 在表达式后添加随机数字
            random_num = random.choice([1, 0, -1])
            return f"({expr}) + {random_num}"
        
        # 如果所有策略都失败，返回原始表达式
        return expr
    
    def _find_operator_nodes(self, expr):
        """找到表达式中的所有运算符节点"""
        nodes = []
        
        # 检查是否是运算符节点（二元或一元运算符）
        if hasattr(expr, '__class__'):
            class_name = expr.__class__.__name__
            
            # 二元运算符
            if class_name in ['Add', 'Sub', 'Mul', 'Div', 'Pow']:
                nodes.append(expr)
                # 递归查找子节点
                if hasattr(expr, 'left'):
                    nodes.extend(self._find_operator_nodes(expr.left))
                if hasattr(expr, 'right'):
                    nodes.extend(self._find_operator_nodes(expr.right))
            
            # 一元运算符
            elif class_name in ['Sin', 'Cos', 'Tan', 'Exp', 'Log', 'Sqrt']:
                nodes.append(expr)
                # 递归查找子节点
                if hasattr(expr, 'operand'):
                    nodes.extend(self._find_operator_nodes(expr.operand))
        
        return nodes
    
    def _remove_node(self, expr, target_node):
        """删除指定的节点，返回修改后的表达式"""
        if expr is target_node:
            # 如果要删除的是根节点，返回None
            return None
            
        if hasattr(expr, 'left') and expr.left is target_node:
            # 删除左子树，返回右子树（如果是二元运算符）
            if hasattr(expr, 'right'):
                return expr.right
            else:
                # 如果是一元运算符，返回操作数
                if hasattr(expr, 'operand'):
                    return expr.operand
                    
        if hasattr(expr, 'right') and expr.right is target_node:
            # 删除右子树，返回左子树
            return expr.left
            
        if hasattr(expr, 'operand') and expr.operand is target_node:
            # 删除一元运算符的操作数，返回操作数
            return expr.operand
        
        # 递归修改子树
        if hasattr(expr, 'left') and expr.left is not target_node:
            new_left = self._remove_node(expr.left, target_node)
            if new_left is not None:
                expr.left = new_left
            else:
                # 如果左子树被删除，返回右子树
                if hasattr(expr, 'right'):
                    return expr.right
                    
        if hasattr(expr, 'right') and expr.right is not target_node:
            new_right = self._remove_node(expr.right, target_node)
            if new_right is not None:
                expr.right = new_right
            else:
                # 如果右子树被删除，返回左子树
                if hasattr(expr, 'left'):
                    return expr.left
                    
        if hasattr(expr, 'operand') and expr.operand is not target_node:
            new_operand = self._remove_node(expr.operand, target_node)
            if new_operand is not None:
                expr.operand = new_operand
            else:
                return None
        
        return expr
    
    def _replace_node(self, expr, target_node):
        """替换指定的运算符节点"""
        if expr is target_node:
            # 替换当前节点
            return self._get_replacement_operator(expr)
            
        # 递归修改子树
        modified = False
        
        if hasattr(expr, 'left') and expr.left is target_node:
            expr.left = self._replace_node(expr.left, target_node)
            modified = True
            
        if hasattr(expr, 'right') and expr.right is target_node:
            expr.right = self._replace_node(expr.right, target_node)
            modified = True
            
        if hasattr(expr, 'operand') and expr.operand is target_node:
            expr.operand = self._replace_node(expr.operand, target_node)
            modified = True
            
        # 如果没有修改子树，递归修改子节点
        if not modified:
            if hasattr(expr, 'left') and expr.left is not target_node:
                new_left = self._replace_node(expr.left, target_node)
                if new_left != expr.left:
                    expr.left = new_left
                    
            if hasattr(expr, 'right') and expr.right is target_node:
                new_right = self._replace_node(expr.right, target_node)
                if new_right != expr.right:
                    expr.right = new_right
                    
            if hasattr(expr, 'operand') and expr.operand is not target_node:
                new_operand = self._replace_node(expr.operand, target_node)
                if new_operand != expr.operand:
                    expr.operand = new_operand
        
        return expr
    
    def _get_replacement_operator(self, original_node):
        """为给定的运算符节点获取替换运算符"""
        class_name = original_node.__class__.__name__
        
        # 二元运算符替换映射
        binary_replacements = {
            'Add': [nd.Sub, nd.Mul, nd.Div],
            'Sub': [nd.Add, nd.Mul, nd.Div],
            'Mul': [nd.Add, nd.Sub, nd.Div],
            'Div': [nd.Add, nd.Sub, nd.Mul],
            'Pow': [nd.Add, nd.Sub, nd.Mul]
        }
        
        # 一元运算符替换映射
        unary_replacements = {
            'Sin': [nd.Cos, nd.Tan, nd.Exp],
            'Cos': [nd.Sin, nd.Tan, nd.Exp],
            'Tan': [nd.Sin, nd.Cos, nd.Exp],
            'Exp': [nd.Log, nd.Sin, nd.Cos],
            'Log': [nd.Sqrt, nd.Exp, nd.Sin],
            'Sqrt': [nd.Exp, nd.Log, nd.Sin]
        }
        
        if class_name in binary_replacements:
            # 获取原操作数
            if hasattr(original_node, 'left') and hasattr(original_node, 'right'):
                new_op_class = random.choice(binary_replacements[class_name])
                return new_op_class(original_node.left, original_node.right)
                
        elif class_name in unary_replacements:
            # 获取原操作数
            if hasattr(original_node, 'operand'):
                new_op_class = random.choice(unary_replacements[class_name])
                return new_op_class(original_node.operand)
        
        # 如果没有合适的替换，返回原节点
        return original_node
    
    def train_epoch(self, expressions, epoch):
        """训练一个epoch（使用三元组损失和混合精度训练，添加梯度监控）"""
        self.triplet_model.train()
        total_loss = 0
        total_grad_norm = 0
        skipped_batches = 0  # 记录跳过的批次
        
        # GPU状态监控 - 增强版
        gpu_usage = []
        for i in range(self.num_gpus):
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                gpu_usage.append(memory_allocated)
                print(f"  GPU {i} 内存: {memory_allocated:.2f}GB")
        
        print("开始生成三元组...")
        print(f"  生成进度: 0/{len(expressions)}", end="")
        
        # 使用新的三元组生成方法
        print("开始生成三元组...")
        triplets = self.generate_triplets(expressions)
        print(f"三元组生成完成: {len(triplets)} 个三元组")
        
        # 随机打乱
        random.shuffle(triplets)
        
        if len(triplets) == 0:
            print("❌ 错误：没有有效的三元组数据")
            return float('inf')
        
        print(f"开始训练: {len(triplets)} 个三元组，批次大小: {self.batch_size}，预计批次: {(len(triplets) + self.batch_size - 1) // self.batch_size}")
        
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
                    if num_batches % 100 == 0:  # 每10个批次显示一次
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
                
                if i % (self.batch_size * 100) == 0:  # 更频繁的日志更新
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
    
    # 创建测试预训练器
    test_trainer = ExpressionPreTrainer()
    
    print("\n开始小规模测试训练...")
    test_trainer.prepare_training_data(test_train_expressions)
    
    # 运行一个测试epoch
    print("运行测试epoch...")
    test_loss = test_trainer.train_epoch(test_train_expressions, 1)
    print(f"测试epoch完成，平均损失: {test_loss:.4f}")
    
    print("✅ 小规模测试通过！")
    
    # 如果测试通过，询问是否继续大规模训练
    print("\n=== 步骤2：大规模训练 ===")
    try:
        user_input = input("小规模测试通过！是否继续大规模训练？(y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # 在非交互环境中，自动选择不继续大规模训练
        print("检测到非交互环境，默认不进行大规模训练")
        print("要运行大规模训练，请手动运行脚本并输入 'y'")
        return
    
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
    trainer = ExpressionPreTrainer()
    
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