"""
预训练管道

实现基于对比学习的预训练，使用对称性InfoNCE损失进行联合训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os
import math
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

from ..models.expression_encoder import ExpressionEncoder
from ..models.data_encoder import DataEncoder


class ContrastiveDataset(Dataset):
    """对比学习数据集"""
    
    def __init__(self, expressions: List[str], datasets: List[Tuple[np.ndarray, np.ndarray]]):
        self.expressions = expressions
        self.datasets = datasets
        
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        return self.expressions[idx], self.datasets[idx]
    
    @staticmethod
    def collate_fn(batch):
        expressions = [item[0] for item in batch]
        data_tuples = [item[1] for item in batch]
        
        X_list = [data[0] for data in data_tuples]
        y_list = [data[1] for data in data_tuples]
        
        return expressions, X_list, y_list


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
        learning_rate = float(self.config.get('learning_rate', 1e-4))
        weight_decay = float(self.config.get('weight_decay', 1e-4))
        
        param_groups = [
            {'params': self.expression_encoder.parameters(), 'lr': learning_rate},
            {'params': self.data_encoder.parameters(), 'lr': learning_rate}
        ]
        
        return torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )
    
    def _setup_logging(self):
        """设置日志"""
        # 确保日志目录存在
        os.makedirs('results/logs', exist_ok=True)

        logger = logging.getLogger('pretrain')
        logger.setLevel(logging.INFO)

        # 阻止消息传播到父级logger，避免重复输出
        logger.propagate = False

        # 清除已有的handlers
        logger.handlers.clear()

        # 只添加文件handler，避免与tqdm进度条冲突
        handler = logging.FileHandler('results/logs/pretrain.log')
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
        """生成预训练数据"""
        expressions = []
        datasets = []
        
        templates = [
            "x1 + x2", "x1 * x2", "sin(x1) + cos(x2)", "x1^2 + x2^2",
            "exp(x1) + log(x2)", "x1 * sin(x2) + x2 * cos(x1)",
            "sqrt(x1^2 + x2^2)", "x1 / (x2 + 1)", "sin(x1 * x2)",
            "x1^3 + x2^3 - x1 * x2", "exp(-x1^2) * sin(x2)",
            "log(x1^2 + x2^2) + x1", "sqrt(x1) + x2^0.5"
        ]
        
        for i in range(n_expressions):
            template = templates[i % len(templates)]
            X, y = self._generate_data_from_expression(template, n_samples_per_expr, variables_range, noise_level)
            expressions.append(template)
            datasets.append((X, y))
        
        self._save_pretrain_data(expressions, datasets, output_path)
        return expressions, datasets
    
    
    
    def _generate_data_from_expression(
        self,
        expression: str,
        n_samples: int,
        variables_range: Tuple[float, float],
        noise_level: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """根据表达式生成数据"""
        x1 = np.random.uniform(variables_range[0], variables_range[1], n_samples)
        x2 = np.random.uniform(variables_range[0], variables_range[1], n_samples)
        
        expr_str = expression.replace('^', '**')
        allowed_names = {
            "x1": x1, "x2": x2, "sin": np.sin, "cos": np.cos,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs
        }
        
        y = eval(expr_str, {"__builtins__": {}}, allowed_names)
        y += np.random.normal(0, noise_level, len(y))
        
        return np.column_stack([x1, x2]), y
    
    
    
    def train_epoch(self, dataloader: TorchDataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.expression_encoder.train()
        self.data_encoder.train()
        
        total_loss = 0.0
        total_samples = 0
        
        for expressions, X_list, y_list in dataloader:
            batch_size = len(expressions)
            
            batch_loss = self._compute_contrastive_loss(expressions, X_list, y_list)
            
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.expression_encoder.parameters()) + list(self.data_encoder.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += batch_loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1
        
        return {'loss': total_loss / total_samples}
    
    def _compute_contrastive_loss(
        self,
        expressions: List[str],
        X_list: List[np.ndarray],
        y_list: List[np.ndarray]
    ) -> torch.Tensor:
        """计算对称性对比学习损失"""
        batch_size = len(expressions)

        # 计算嵌入向量
        expr_embeddings = torch.stack([
            self.expression_encoder.encode(expr, training=True) for expr in expressions
        ])
        
        data_embeddings = torch.stack([
            self.data_encoder.encode(
                torch.FloatTensor(X_list[i]).to(self.device),
                torch.FloatTensor(y_list[i]).to(self.device)
            ) for i in range(batch_size)
        ])
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(expr_embeddings, data_embeddings.transpose(0, 1)) / self.temperature
        labels = torch.arange(batch_size).to(self.device)
        
        # 对称性InfoNCE损失
        loss_expr_to_data = self.info_nce_loss(similarity_matrix, labels)
        loss_data_to_expr = self.info_nce_loss(similarity_matrix.transpose(0, 1), labels)
        
        return 0.5 * (loss_expr_to_data + loss_data_to_expr)
    
    def fit(
        self,
        expressions: Optional[List[str]] = None,
        datasets: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Dict[str, List[float]]:
        """训练模型"""
        # 数据分割
        combined_data = list(zip(expressions, datasets))
        train_combined, val_combined = train_test_split(combined_data, test_size=0.2, random_state=42)
        train_expr, train_data = zip(*train_combined)
        val_expr, val_data = zip(*val_combined)
        
        # 创建数据加载器
        train_loader = TorchDataLoader(
            ContrastiveDataset(train_expr, train_data),
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=ContrastiveDataset.collate_fn
        )
        
        val_loader = TorchDataLoader(
            ContrastiveDataset(val_expr, val_data),
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=ContrastiveDataset.collate_fn
        )
        
        # 训练循环
        train_history = {'loss': []}
        val_history = {'loss': []}
        
        epoch_pbar = tqdm(range(self.config['num_epochs']), desc="Training Epochs")

        for epoch in epoch_pbar:
            self.epoch = epoch
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            train_history['loss'].append(train_metrics['loss'])
            val_history['loss'].append(val_metrics['loss'])
            
            self.scheduler.step(val_metrics['loss'])
            
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.save_pretrained()
            
            epoch_pbar.set_postfix({
                'Train Loss': f"{train_metrics['loss']:.4f}",
                'Val Loss': f"{val_metrics['loss']:.4f}"
            })
        
        epoch_pbar.close()
        return {'train_loss': train_history, 'val_loss': val_history}
    
    def validate(self, dataloader: TorchDataLoader) -> Dict[str, float]:
        """验证"""
        self.expression_encoder.eval()
        self.data_encoder.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for expressions, X_list, y_list in dataloader:
                batch_size = len(expressions)
                batch_loss = self._compute_contrastive_loss(expressions, X_list, y_list)
                total_loss += batch_loss.item() * batch_size
                total_samples += batch_size
        
        return {'loss': total_loss / total_samples}
    
    def save_pretrained(self, save_path: Optional[str] = None):
        """保存预训练模型"""
        save_path = save_path or self.config.get('output_dir', 'models_weights/pretrained/')
        os.makedirs(save_path, exist_ok=True)

        self.expression_encoder.save_pretrained(os.path.join(save_path, 'expression_encoder'))
        self.data_encoder.save_pretrained(os.path.join(save_path, 'data_encoder'))

        with open(os.path.join(save_path, 'training_config.json'), 'w') as f:
            import json
            json.dump({
                'temperature': self.temperature,
                'best_loss': self.best_loss,
                'global_step': self.global_step,
                'epoch': self.epoch
            }, f, indent=2)
    
    def load_pretrained(self, load_path: str):
        """加载预训练模型"""
        expr_path = os.path.join(load_path, 'expression_encoder')
        data_path = os.path.join(load_path, 'data_encoder')
        
        self.expression_encoder = ExpressionEncoder.from_pretrained(expr_path)
        self.data_encoder = DataEncoder.from_pretrained(data_path)
        self.expression_encoder.to(self.device)
        self.data_encoder.to(self.device)
        
        config_path = os.path.join(load_path, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
                self.temperature = config.get('temperature', 0.07)
                self.best_loss = config.get('best_loss', float('inf'))
                self.global_step = config.get('global_step', 0)
                self.epoch = config.get('epoch', 0)