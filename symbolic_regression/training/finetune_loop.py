"""
在线微调循环

实现基于真实解引导的专家微调阶段，使用MCTS+三元组损失
对编码器进行精细打磨。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from collections import deque
import os
import pickle
import copy
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

from ..models.expression_encoder import ExpressionEncoder
from ..models.data_encoder import DataEncoder
from ..core.mcts_engine import MCTSEngine
from ..core.reward_calculator import RewardCalculator


class HardNegativeSample(NamedTuple):
    """难负样本三元组"""
    anchor_expr: str  # 锚点（真实解）
    negative_expr: str  # 难负样本
    data_X: np.ndarray  # 数据特征
    data_y: np.ndarray  # 数据标签
    target_embedding: Optional[torch.Tensor] = None  # 目标数据嵌入（可选缓存）


class OnlineFinetuneLoop:
    """在线微调循环"""

    def __init__(
        self,
        expression_encoder: ExpressionEncoder,
        data_encoder: DataEncoder,
        config: Dict[str, Any],
        device: str = 'cpu'
    ):
        """
        初始化在线微调循环

        Args:
            expression_encoder: 表达式编码器
            data_encoder: 数据编码器
            config: 配置字典
            device: 设备
        """
        self.expression_encoder = expression_encoder
        self.data_encoder = data_encoder
        self.config = config
        self.device = device

        # 将模型移动到设备
        self.expression_encoder.to(device)
        self.data_encoder.to(device)

        # 奖励计算器
        self.reward_calculator = RewardCalculator(
            reward_weights=config.get('reward_weights', {
                'accuracy': 0.5,
                'data_alignment': 0.3,
                'structure_alignment': 0.2,
            }),
            temperature=0.07
        )

        # MCTS引擎
        self.mcts_engine = self._create_mcts_engine()

        # 难负样本池
        self.hard_negative_pool = deque(maxlen=config.get('hard_negative_pool_size', 1000))
        self.hard_negative_threshold = config.get('hard_negative_threshold', 0.8)

        # 三元组损失
        self.triplet_loss = nn.TripletMarginLoss(
            margin=config.get('triplet_margin', 1.0),
            p=2  # 欧氏距离
        )
        self.alignment_loss_weight = config.get('alignment_loss_weight', 0.5)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 日志
        self.logger = self._setup_logging()

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float('-inf')

        # 统计信息
        self.statistics = {
            'hard_negatives_found': 0,
            'triplet_losses': [],
            'alignment_losses': [],
            'total_losses': [],
            'mcts_rewards': [],
        }

    def _create_mcts_engine(self) -> MCTSEngine:
        """创建MCTS引擎"""
        return MCTSEngine(
            expression_encoder=self.expression_encoder,
            data_encoder=self.data_encoder,
            reward_calculator=self.reward_calculator,
            config=self.config.get('mcts', {}),
            device=self.device,
            freeze_encoders=False  # 在线微调阶段不冻结编码器
        )

    def _create_optimizer(self):
        """创建优化器"""
        learning_rate = float(self.config.get('learning_rate', 1e-6))  # 极低学习率

        param_groups = [
            {'params': self.expression_encoder.parameters(), 'lr': learning_rate},
            {'params': self.data_encoder.parameters(), 'lr': learning_rate}
        ]

        if self.config.get('optimizer', {}).get('type', 'adamw') == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=float(self.config.get('weight_decay', 1e-4))
            )
        else:
            optimizer = torch.optim.Adam(
                param_groups,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=float(self.config.get('weight_decay', 1e-4))
            )

        return optimizer

    def _create_scheduler(self):
        """创建学习率调度器"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 监控奖励，奖励越高越好
            factor=0.5,
            patience=5,
            min_lr=1e-8
        )

    def _setup_logging(self):
        """设置日志"""
        os.makedirs('results/logs', exist_ok=True)

        logger = logging.getLogger('finetune')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        handler = logging.FileHandler('results/logs/finetune.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def finetune(
        self,
        benchmark_tasks: List[Tuple[str, np.ndarray, np.ndarray]],
        mcts_epochs: int = 50,
        eval_steps: int = 10,
        save_steps: int = 10,
        output_dir: str = "models_weights/finetuned/",
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        执行在线微调

        Args:
            benchmark_tasks: 基准任务列表，每个元素为 (真实表达式, X, y)
            mcts_epochs: MCTS微调轮数
            eval_steps: 评估步数
            save_steps: 保存步数
            output_dir: 输出目录
            verbose: 是否打印详细信息

        Returns:
            训练历史
        """
        self.logger.info(f"开始在线微调，共{len(benchmark_tasks)}个基准任务")

        # 分割训练和验证任务
        train_tasks, val_tasks = train_test_split(
            benchmark_tasks,
            test_size=0.2,
            random_state=42
        )

        # 主微调循环
        train_history = {
            'triplet_loss': [],
            'total_loss': [],
            'mcts_reward': []
        }

        for epoch in range(mcts_epochs):
            self.epoch = epoch

            # 为每个基准任务执行MCTS+微调
            epoch_metrics = {
                'triplet_loss': 0.0,
                'total_loss': 0.0,
                'mcts_reward': 0.0,
                'count': 0
            }

            # 随机打乱任务顺序
            tasks_order = np.random.permutation(len(train_tasks))

            pbar = tqdm(tasks_order, desc=f"Epoch {epoch+1}/{mcts_epochs}", leave=False) if verbose else tasks_order

            for task_idx in pbar:
                # 获取任务
                true_expr, X, y = train_tasks[task_idx]

                # 执行MCTS探索
                best_expr, reward_dict , best_reward = self.mcts_engine.search(
                    task_data=(X, y),
                    target_expression=true_expr,
                    verbose=False
                )

                print(f'best_expr: {best_expr}\nreward_dict: {reward_dict}\nbest_reward: {best_reward}')

                # 检查是否为难负样本
                self._check_and_add_hard_negative(
                    true_expr, best_expr, X, y, true_expr
                )

                # 记录MCTS奖励
                epoch_metrics['mcts_reward'] += best_reward

                # 检查是否有难负样本，如果没有则跳过微调
                if len(self.hard_negative_pool) == 0:
                    epoch_metrics['count'] += 1
                    # 更新进度条
                    if verbose:
                        pbar.set_postfix({
                            'Total Loss': f"0.0000 (skip)",
                            'Hard Neg Pool': f"{len(self.hard_negative_pool)}"
                        })
                    continue

                # 有难负样本，进行微调
                # 使用当前难负样本池中的样本进行训练
                triplet_loss, _ = self._compute_batch_finetune_loss(
                    list(self.hard_negative_pool)
                )

                # 使用三元组损失作为总损失
                total_loss = triplet_loss

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    list(self.expression_encoder.parameters()) +
                    list(self.data_encoder.parameters()),
                    max_norm=1.0
                )

                self.optimizer.step()

                # 记录指标
                epoch_metrics['triplet_loss'] += triplet_loss.item()
                epoch_metrics['total_loss'] += total_loss.item()
                epoch_metrics['count'] += 1

                # 更新进度条
                if verbose:
                    pbar.set_postfix({
                        'Total Loss': f"{total_loss.item():.4f}",
                        'Hard Neg Pool': f"{len(self.hard_negative_pool)}"
                    })

            # 计算epoch平均指标
            if epoch_metrics['count'] > 0:
                for key in epoch_metrics:
                    if key != 'count':
                        epoch_metrics[key] /= epoch_metrics['count']

                train_history['triplet_loss'].append(epoch_metrics['triplet_loss'])
                train_history['total_loss'].append(epoch_metrics['total_loss'])
                train_history['mcts_reward'].append(epoch_metrics['mcts_reward'])

            # 验证
            if (epoch + 1) % eval_steps == 0:
                val_metrics = self._validate(val_tasks)

                # 更新学习率调度器
                self.scheduler.step(val_metrics['reward'])

                # 保存最佳模型
                if val_metrics['reward'] > self.best_reward:
                    self.best_reward = val_metrics['reward']
                    self.save_pretrained(output_dir)
                    self.logger.info(f"新的最佳模型已保存，验证奖励: {self.best_reward:.4f}")

                self.logger.info(
                    f"Epoch {epoch+1}: Val Reward = {val_metrics['reward']:.4f}, "
                    f"Hard Neg Pool = {len(self.hard_negative_pool)}"
                )

            # 定期保存
            if (epoch + 1) % save_steps == 0:
                self.save_pretrained(os.path.join(output_dir, f"epoch_{epoch+1}"))

            # 更新统计
            self.statistics['hard_negatives_found'] += 0  # 实际计数在_check_and_add_hard_negative中
            self.global_step += 1

        pbar.close() if verbose else None

        self.logger.info("在线微调完成！")

        return train_history

    def _check_and_add_hard_negative(
        self,
        true_expr: str,
        candidate_expr: str,
        X: np.ndarray,
        y: np.ndarray,
        target_expr: str
    ):
        """检查候选表达式是否为难负样本，如果是则加入样本池"""
        true_embedding = None
        candidate_embedding = None
        true_tensor = None
        candidate_tensor = None

        try:
            # 计算嵌入
            true_embedding = self.expression_encoder.encode(true_expr, training=False)
            candidate_embedding = self.expression_encoder.encode(candidate_expr, training=False)

            # 计算结构相似度
            true_tensor = true_embedding.unsqueeze(0)
            candidate_tensor = candidate_embedding.unsqueeze(0)
            structure_sim = F.cosine_similarity(true_tensor, candidate_tensor).item()

            # 放宽条件：如果结构相似度中等，且准确度有一定差异
            if structure_sim >= 0.5:  # 降低阈值，从0.8降到0.3
                # 计算准确度差异
                true_r2 = self._evaluate_r2(true_expr, X, y)
                candidate_r2 = self._evaluate_r2(candidate_expr, X, y)

                # 放宽准确度差异要求
                if true_r2 > candidate_r2 + 0.05:  # 从10%降到5%
                    # 添加到难负样本池
                    sample = HardNegativeSample(
                        anchor_expr=true_expr,
                        negative_expr=candidate_expr,
                        data_X=X,
                        data_y=y
                    )
                    print(f"真实表达式：{true_expr}，准确度：{true_r2:.4f} \n添加难负样本:{candidate_expr}，准确度：{candidate_r2:.4f}\n难负样本结构相似度：{structure_sim:.4f}")

                    self.hard_negative_pool.append(sample)
                    self.statistics['hard_negatives_found'] += 1

        except Exception as e:
            self.logger.warning(f"检查难负样本时出错: {e}")
        finally:
            # 清理临时张量
            del true_embedding, candidate_embedding, true_tensor, candidate_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _evaluate_r2(self, expression: str, X: np.ndarray, y: np.ndarray) -> float:
        """评估表达式的R²分数"""
        try:
            y_pred = self._safe_eval_expression(expression, X)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                return 0.0 if np.allclose(y_pred, y) else -np.inf

            return 1 - (ss_res / ss_tot)

        except Exception:
            return -np.inf

    def _safe_eval_expression(self, expression: str, X: np.ndarray) -> np.ndarray:
        """安全地求值表达式"""
        n_vars = X.shape[1]

        var_dict = {}
        for i in range(n_vars):
            var_dict[f'x{i + 1}'] = X[:, i]

        var_dict.update({
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
        })

        expr_str = expression.replace('^', '**')
        safe_dict = {"__builtins__": {}}

        return eval(expr_str, safe_dict, var_dict)

    def _compute_finetune_loss(
        self,
        true_expr: str,
        best_expr: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算微调损失（三元组损失 + 对齐损失）

        Returns:
            (triplet_loss, alignment_loss)
        """
        # 确保模型在训练模式
        self.expression_encoder.train()
        self.data_encoder.train()

        # 计算嵌入
        true_embedding = self.expression_encoder.encode(true_expr, training=True)
        candidate_embedding = self.expression_encoder.encode(best_expr, training=True)

        # 计算数据嵌入
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        data_embedding = self.data_encoder.encode(X_tensor, y_tensor)

        # 修复：使用数据嵌入作为正样本
        # anchor: 真实解表达式 (true_expr)
        # positive: 数据嵌入 (期望模型生成与数据匹配的表达式)
        # negative: MCTS候选表达式 (best_expr)
        positive_embedding = data_embedding

        # 计算三元组损失
        # 目标：d(anchor, positive) - d(anchor, negative) + margin > 0
        # 这鼓励模型生成更接近真实解且符合数据特征的表达式
        triplet_loss = self.triplet_loss(
            anchor=true_embedding,
            positive=positive_embedding,
            negative=candidate_embedding
        )

        # 调试输出（仅在前几步）
        if self.global_step < 5:
            self.logger.info(f"Step {self.global_step}:")
            self.logger.info(f"  Triplet Loss: {triplet_loss.item():.6f}")
            self.logger.info(f"  True Expr: {true_expr}")
            self.logger.info(f"  Best Expr: {best_expr}")

        return triplet_loss

    def _compute_alignment_loss(self, expression: str, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """计算表达式与数据的对齐损失"""
        try:
            # 计算表达式嵌入
            expr_embedding = self.expression_encoder.encode(expression, training=True)

            # 计算数据嵌入
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            data_embedding = self.data_encoder.encode(X_tensor, y_tensor)

            # 计算余弦相似度损失（距离为1 - 相似度）
            cosine_sim = torch.nn.functional.cosine_similarity(
                expr_embedding.unsqueeze(0),
                data_embedding.unsqueeze(0)
            )

            alignment_loss = 1 - cosine_sim  # 转换为距离

            return alignment_loss

        except Exception:
            return torch.tensor(0.0, device=self.device)

    def _compute_batch_finetune_loss(self, batch_samples: List[HardNegativeSample]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算批次级别的微调损失"""
        if not batch_samples:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        total_triplet_loss = 0.0
        total_alignment_loss = 0.0

        for sample in batch_samples:
            triplet_loss, alignment_loss = self._compute_finetune_loss(
                sample.anchor_expr,
                sample.negative_expr,
                sample.data_X,
                sample.data_y
            )

            total_triplet_loss += triplet_loss
            total_alignment_loss += alignment_loss

        # 计算平均损失
        avg_triplet_loss = total_triplet_loss / len(batch_samples)
        avg_alignment_loss = total_alignment_loss / len(batch_samples)

        return avg_triplet_loss, avg_alignment_loss

    def _validate(self, val_tasks: List[Tuple[str, np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """验证模型性能"""
        self.expression_encoder.eval()
        self.data_encoder.eval()

        total_reward = 0.0
        total_count = 0

        with torch.no_grad():
            for true_expr, X, y in val_tasks:
                # 执行MCTS搜索
                best_expr, mcts_reward = self.mcts_engine.search(
                    task_data=(X, y),
                    target_expression=true_expr,
                    verbose=False
                )

                total_reward += mcts_reward
                total_count += 1

        avg_reward = total_reward / max(1, total_count)

        return {'reward': avg_reward}

    def train_with_hard_negatives(
        self,
        batch_size: int = 8,
        epochs: int = 10
    ) -> Dict[str, float]:
        """使用难负样本池训练模型"""
        if len(self.hard_negative_pool) == 0:
            self.logger.warning("难负样本池为空，跳过训练")
            return {'triplet_loss': 0.0, 'total_loss': 0.0}

        self.logger.info(f"使用{len(self.hard_negative_pool)}个难负样本进行训练")

        self.expression_encoder.train()
        self.data_encoder.train()

        total_triplet_loss = 0.0

        # 随机采样训练
        for epoch in range(epochs):
            # 随机采样一个批次
            samples = np.random.choice(
                list(self.hard_negative_pool),
                size=min(batch_size, len(self.hard_negative_pool)),
                replace=False
            )

            # 计算损失
            triplet_loss, _ = self._compute_batch_finetune_loss(list(samples))

            # 使用三元组损失作为总损失
            total_loss = triplet_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.expression_encoder.parameters()) +
                list(self.data_encoder.parameters()),
                max_norm=1.0
            )

            self.optimizer.step()

            total_triplet_loss += triplet_loss.item()

        avg_triplet_loss = total_triplet_loss / epochs

        self.logger.info(
            f"难负样本训练完成: Triplet Loss = {avg_triplet_loss:.4f}"
        )

        return {
            'triplet_loss': avg_triplet_loss,
            'total_loss': avg_triplet_loss
        }

    def save_pretrained(self, save_path: str):
        """保存微调后的模型"""
        os.makedirs(save_path, exist_ok=True)

        # 保存表达式编码器
        expr_path = os.path.join(save_path, 'expression_encoder')
        self.expression_encoder.save_pretrained(expr_path)

        # 保存数据编码器
        data_path = os.path.join(save_path, 'data_encoder')
        self.data_encoder.save_pretrained(data_path)

        # 保存难负样本池
        hard_neg_path = os.path.join(save_path, 'hard_negative_pool.pkl')
        with open(hard_neg_path, 'wb') as f:
            pickle.dump(list(self.hard_negative_pool), f)

        # 保存训练配置
        config_path = os.path.join(save_path, 'finetune_config.json')
        with open(config_path, 'w') as f:
            import json
            json.dump({
                'best_reward': self.best_reward,
                'global_step': self.global_step,
                'epoch': self.epoch,
                'statistics': self.statistics
            }, f, indent=2)

        self.logger.info(f"微调模型已保存到 {save_path}")

    def load_pretrained(self, load_path: str):
        """加载微调后的模型"""
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

        # 加载难负样本池
        hard_neg_path = os.path.join(load_path, 'hard_negative_pool.pkl')
        if os.path.exists(hard_neg_path):
            with open(hard_neg_path, 'rb') as f:
                self.hard_negative_pool = deque(
                    pickle.load(f),
                    maxlen=self.config.get('hard_negative_pool_size', 1000)
                )

        # 加载训练状态
        config_path = os.path.join(load_path, 'finetune_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
                self.best_reward = config.get('best_reward', float('-inf'))
                self.global_step = config.get('global_step', 0)
                self.epoch = config.get('epoch', 0)
                self.statistics = config.get('statistics', {})

        self.logger.info(f"微调模型已从 {load_path} 加载")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'hard_negative_pool_size': len(self.hard_negative_pool),
            'hard_negatives_found': self.statistics['hard_negatives_found'],
            'best_reward': self.best_reward,
            'global_step': self.global_step,
            'epoch': self.epoch
        }
