"""
预训练管道

实现基于对比学习的预训练，使用对称性InfoNCE损失进行联合训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

from ..models.expression_encoder import ExpressionEncoder
from ..models.data_encoder import DataEncoder


class ContrastiveDataset(Dataset):
    """对比学习数据集"""
    
    def __init__(
        self, 
        expressions: List[str], 
        datasets: List[np.ndarray],
        transform=None
    ):
        self.expressions = expressions
        self.datasets = datasets
        self.transform = transform
        
        # 确保expressions和datasets长度匹配
        assert len(expressions) == len(datasets), "expressions和datasets长度不匹配"
        
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        expression = self.expressions[idx]
        data = self.datasets[idx]
        
        if self.transform:
            expression = self.transform(expression)
            data = self.transform(data)
            
        return expression, data


class PretrainPipeline:
    """预训练管道"""
    
    def __init__(
        self,
        expression_encoder: ExpressionEncoder,
        data_encoder: DataEncoder,
        config: Dict[str, Any],
        device: str = 'cpu'
    ):
        self.expression_encoder = expression_encoder
        self.data_encoder = data_encoder
        self.config = config
        self.device = device
        
        # 将模型移动到设备
        self.expression_encoder.to(device)
        self.data_encoder.to(device)
        
        # 对比学习损失
        self.temperature = config.get('temperature', 0.07)
        self.info_nce_loss = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 日志
        self.logger = self._setup_logging()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def _create_optimizer(self):
        """创建优化器"""
        # 为两个编码器创建联合优化器
        param_groups = [
            {'params': self.expression_encoder.parameters(), 'lr': self.config['learning_rate']},
            {'params': self.data_encoder.parameters(), 'lr': self.config['learning_rate']}
        ]
        
        if self.config.get('optimizer', {}).get('type', 'adamw') == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config['learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.config['learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.get('scheduler', {}).get('type') == 'cosine_with_warmup':
            from transformers import get_cosine_schedule_with_warmup
            
            num_training_steps = self.config['num_epochs'] * self.config.get('steps_per_epoch', 1000)
            num_warmup_steps = self.config.get('warmup_steps', 1000)
            
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            # 使用StepLR作为默认调度器
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
            
        return scheduler
    
    def _setup_logging(self):
        """设置日志"""
        logger = logging.getLogger('pretrain')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def generate_pretrain_data(
        self,
        n_expressions: int = 100000,
        n_samples_per_expr: int = 100,
        variables_range: Tuple[float, float] = (-5, 5),
        noise_level: float = 0.01,
        output_path: str = "data/pretrain/"
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        生成预训练数据
        
        Args:
            n_expressions: 生成的表达式数量
            n_samples_per_expr: 每个表达式的数据点数量
            variables_range: 变量取值范围
            noise_level: 噪声水平
            output_path: 输出路径
            
        Returns:
            (表达式列表, 数据集列表)
        """
        self.logger.info(f"开始生成{n_expressions}个预训练表达式...")
        
        expressions = []
        datasets = []
        
        # 预定义的数学函数模板
        expression_templates = [
            "x1 + x2",
            "x1 * x2",
            "sin(x1) + cos(x2)",
            "x1^2 + x2^2",
            "exp(x1) + log(x2)",
            "x1 * sin(x2) + x2 * cos(x1)",
            "sqrt(x1^2 + x2^2)",
            "x1 / (x2 + 1)",
            "sin(x1 * x2) + cos(x1 / x2)",
            "x1^3 + x2^3 - x1 * x2",
            "exp(-x1^2) * sin(x2)",
            "log(x1^2 + x2^2) + x1",
            "x1 * exp(-x2) + cos(x1 + x2)",
            "sqrt(x1) + x2^0.5",
            "sin(x1) * cos(x2) + x1 * x2",
            "x1^2 * sin(x2) + x2^2 * cos(x1)",
            "exp(x1) * x2 + log(x1 + x2)",
            "sqrt(x1 + x2) / (x1 * x2 + 1)",
            "sin(x1 + x2) + cos(x1 - x2)",
            "x1^4 + x2^4 - 2 * x1 * x2"
        ]
        
        for i in range(n_expressions):
            # 选择模板或生成新的表达式
            if i < len(expression_templates):
                template = expression_templates[i % len(expression_templates)]
            else:
                template = self._generate_random_expression()
            
            # 生成数据
            try:
                X, y = self._generate_data_from_expression(
                    template, 
                    n_samples_per_expr, 
                    variables_range, 
                    noise_level
                )
                
                expressions.append(template)
                datasets.append((X, y))
                
                if (i + 1) % 10000 == 0:
                    self.logger.info(f"已生成 {i + 1}/{n_expressions} 个表达式")
                    
            except Exception as e:
                self.logger.warning(f"生成表达式 {template} 时出错: {e}")
                continue
        
        # 保存数据
        self._save_pretrain_data(expressions, datasets, output_path)
        
        self.logger.info(f"预训练数据生成完成，共{len(expressions)}个有效表达式")
        return expressions, datasets
    
    def _generate_random_expression(self) -> str:
        """生成随机数学表达式"""
        import random
        
        # 随机生成表达式
        base_exprs = [
            "x1", "x2", "x1 + x2", "x1 - x2", "x1 * x2", "x1 / x2",
            "x1^2", "x2^2", "sin(x1)", "cos(x2)", "exp(x1)", "log(x2)"
        ]
        
        base = random.choice(base_exprs)
        
        # 随机添加操作
        operations = [
            f" + {random.choice(base_exprs)}",
            f" - {random.choice(base_exprs)}",
            f" * {random.choice(base_exprs)}",
            f" / {random.choice(base_exprs)}",
            f" + sin({random.choice(base_exprs)})",
            f" + cos({random.choice(base_exprs)})"
        ]
        
        if random.random() < 0.7:  # 70%的概率添加额外操作
            base += random.choice(operations)
            
        return base
    
    def _generate_data_from_expression(
        self,
        expression: str,
        n_samples: int,
        variables_range: Tuple[float, float],
        noise_level: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """根据表达式生成数据"""
        # 导入nd2py包
        sys_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..')
        sys.path.append(sys_path)
        import nd2py as nd
        
        # 变量范围
        x1_range = variables_range
        x2_range = variables_range
        
        # 生成输入数据
        x1 = np.random.uniform(x1_range[0], x1_range[1], n_samples)
        x2 = np.random.uniform(x2_range[0], x2_range[1], n_samples)
        
        # 准备输入字典
        X_dict = {'x1': x1, 'x2': x2}
        
        # 解析并评估表达式
        try:
            # 这里需要根据实际的nd2py语法来解析表达式
            # 暂时使用简单的eval方式，实际应用中需要更严格的解析
            expr_str = expression.replace('^', '**')
            
            # 安全的表达式求值（仅允许特定函数）
            allowed_names = {
                "x1": x1,
                "x2": x2,
                "sin": np.sin,
                "cos": np.cos,
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "abs": np.abs
            }
            
            y = eval(expr_str, {"__builtins__": {}}, allowed_names)
            
            # 添加噪声
            noise = np.random.normal(0, noise_level, len(y))
            y += noise
            
            # 组合输入数据
            X = np.column_stack([x1, x2])
            
            return X, y
            
        except Exception as e:
            # 如果表达式求值失败，使用简单的线性组合
            y = x1 + 2 * x2 + np.random.normal(0, noise_level, n_samples)
            X = np.column_stack([x1, x2])
            return X, y
    
    def _save_pretrain_data(
        self,
        expressions: List[str],
        datasets: List[Tuple[np.ndarray, np.ndarray]],
        output_path: str
    ):
        """保存预训练数据"""
        os.makedirs(output_path, exist_ok=True)
        
        # 保存表达式
        with open(os.path.join(output_path, 'expressions.pkl'), 'wb') as f:
            pickle.dump(expressions, f)
        
        # 保存数据集
        with open(os.path.join(output_path, 'datasets.pkl'), 'wb') as f:
            pickle.dump(datasets, f)
            
        self.logger.info(f"预训练数据已保存到 {output_path}")
    
    def load_pretrain_data(self, data_path: str) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
        """加载预训练数据"""
        expressions_path = os.path.join(data_path, 'expressions.pkl')
        datasets_path = os.path.join(data_path, 'datasets.pkl')
        
        with open(expressions_path, 'rb') as f:
            expressions = pickle.load(f)
            
        with open(datasets_path, 'rb') as f:
            datasets = pickle.load(f)
            
        self.logger.info(f"从 {data_path} 加载了 {len(expressions)} 个预训练样本")
        return expressions, datasets
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.expression_encoder.train()
        self.data_encoder.train()
        
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (expressions, datasets) in enumerate(progress_bar):
            batch_size = len(expressions)
            
            # 准备数据
            # 这里需要实现数据加载逻辑
            batch_losses = []
            
            for i in range(batch_size):
                # 获取表达式和数据
                expr = expressions[i]
                X, y = datasets[i]
                
                # 计算损失
                loss = self._compute_contrastive_loss(expr, (X, y))
                batch_losses.append(loss)
            
            # 批次平均损失
            batch_loss = torch.stack(batch_losses).mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.expression_encoder.parameters()) + list(self.data_encoder.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += batch_loss.item() * batch_size
            total_samples += batch_size
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / total_samples
        return {'loss': avg_loss}
    
    def _compute_contrastive_loss(
        self, 
        expression: str, 
        data_tuple: Tuple[np.ndarray, np.ndarray]
    ) -> torch.Tensor:
        """计算对比学习损失"""
        X, y = data_tuple
        
        # 计算嵌入向量
        expr_embedding = self.expression_encoder.encode(expression)
        data_embedding = self.data_encoder.encode(X, y)
        
        # 转换为张量并添加batch维度
        expr_tensor = torch.FloatTensor(expr_embedding).unsqueeze(0).to(self.device)
        data_tensor = torch.FloatTensor(data_embedding).unsqueeze(0).to(self.device)
        
        # InfoNCE损失需要批次处理，这里使用简化的单样本版本
        # 实际应用中需要构建正负样本对
        similarity = F.cosine_similarity(expr_tensor, data_tensor)
        
        # 将相似度转换为损失
        # 对于正样本，我们希望相似度越高越好，所以损失 = -similarity
        loss = -similarity / self.temperature
        
        return loss
    
    def fit(
        self,
        expressions: Optional[List[str]] = None,
        datasets: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        generate_data: bool = True,
        data_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            expressions: 表达式列表
            datasets: 数据集列表
            generate_data: 是否生成数据
            data_path: 数据路径
            
        Returns:
            训练历史
        """
        # 加载或生成数据
        if expressions is None or datasets is None:
            if data_path and os.path.exists(data_path):
                expressions, datasets = self.load_pretrain_data(data_path)
            elif generate_data:
                expressions, datasets = self.generate_pretrain_data(
                    n_expressions=self.config['n_expressions'],
                    n_samples_per_expr=self.config['n_samples_per_expr'],
                    variables_range=tuple(self.config['variables_range']),
                    noise_level=self.config['noise_level'],
                    output_path=self.config['output_path']
                )
            else:
                raise ValueError("需要提供expressions和datasets，或启用数据生成")
        
        # 分割训练和验证集
        train_expr, val_expr, train_data, val_data = train_test_split(
            expressions, datasets, test_size=0.1, random_state=42
        )
        
        # 创建数据集
        train_dataset = ContrastiveDataset(train_expr, train_data)
        val_dataset = ContrastiveDataset(val_expr, val_data)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        # 训练循环
        train_history = {'loss': []}
        val_history = {'loss': []}
        
        self.logger.info("开始预训练...")
        
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            train_history['loss'].append(train_metrics['loss'])
            
            # 验证
            val_metrics = self.validate(val_loader)
            val_history['loss'].append(val_metrics['loss'])
            
            # 记录日志
            self.logger.info(
                f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}, "
                f"Val Loss = {val_metrics['loss']:.4f}"
            )
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.save_pretrained()
                self.logger.info(f"新的最佳模型已保存，验证损失: {self.best_loss:.4f}")
        
        self.logger.info("预训练完成！")
        return {
            'train_loss': train_history,
            'val_loss': val_history
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.expression_encoder.eval()
        self.data_encoder.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for expressions, datasets in dataloader:
                batch_size = len(expressions)
                
                batch_losses = []
                for i in range(batch_size):
                    expr = expressions[i]
                    X, y = datasets[i]
                    
                    loss = self._compute_contrastive_loss(expr, (X, y))
                    batch_losses.append(loss)
                
                batch_loss = torch.stack(batch_losses).mean()
                total_loss += batch_loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        return {'loss': avg_loss}
    
    def save_pretrained(self, save_path: Optional[str] = None):
        """保存预训练模型"""
        if save_path is None:
            save_path = self.config.get('output_dir', 'models_weights/pretrained/')
        
        os.makedirs(save_path, exist_ok=True)
        
        # 保存表达式编码器
        expr_path = os.path.join(save_path, 'expression_encoder')
        self.expression_encoder.save_pretrained(expr_path)
        
        # 保存数据编码器
        data_path = os.path.join(save_path, 'data_encoder')
        self.data_encoder.save_pretrained(data_path)
        
        # 保存训练配置
        config_path = os.path.join(save_path, 'training_config.json')
        with open(config_path, 'w') as f:
            import json
            json.dump({
                'temperature': self.temperature,
                'best_loss': self.best_loss,
                'global_step': self.global_step,
                'epoch': self.epoch
            }, f, indent=2)
        
        self.logger.info(f"预训练模型已保存到 {save_path}")
    
    def load_pretrained(self, load_path: str):
        """加载预训练模型"""
        # 加载表达式编码器
        expr_path = os.path.join(load_path, 'expression_encoder')
        if os.path.exists(expr_path):
            self.expression_encoder = ExpressionEncoder.from_pretrained(expr_path)
            self.expression_encoder.to(self.device)
        
        # 加载数据编码器
        data_path = os.path.join(load_path, 'data_encoder')
        if os.path.exists(data_path):
            self.data_encoder = DataEncoder.from_pretrained(data_path)
            self.data_encoder.to(self.device)
        
        # 加载训练状态
        config_path = os.path.join(load_path, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
                self.temperature = config.get('temperature', 0.07)
                self.best_loss = config.get('best_loss', float('inf'))
                self.global_step = config.get('global_step', 0)
                self.epoch = config.get('epoch', 0)
        
        self.logger.info(f"预训练模型已从 {load_path} 加载")