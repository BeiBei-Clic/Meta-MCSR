"""
在线微调循环

实现MCTS探索与真实解引导微调的集成算法。
这是算法的核心部分，将MCTS探索与编码器微调结合。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import logging
from tqdm import tqdm
from collections import deque

from ..core.mcts_engine import EnhancedMCTSEngine, EnhancedNode
from ..core.reward_calculator import RewardCalculator
from ..models.expression_encoder import ExpressionEncoder
from ..models.data_encoder import DataEncoder


class FinetuneLoop:
    """在线微调循环，实现MCTS探索与真实解引导微调的集成"""
    
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
        
        # 创建优化器（极低学习率）
        self.optimizer = self._create_optimizer()
        
        # 奖励计算器
        self.reward_calculator = RewardCalculator(
            reward_weights=config.get('reward_weights', None)
        )
        
        # MCTS引擎
        self.mcts_engine = self._create_mcts_engine()
        
        # 经验回放池
        self.experience_buffer = deque(maxlen=config.get('buffer_size', 10000))
        
        # 日志
        self.logger = self._setup_logging()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_performance = {
            'r2': -np.inf,
            'expression': None,
            'epoch': -1
        }
        
        # 性能记录
        self.performance_history = {
            'r2_scores': [],
            'best_expressions': [],
            'finetune_losses': []
        }
    
    def _create_optimizer(self):
        """创建优化器，使用极低学习率"""
        param_groups = [
            {'params': self.expression_encoder.parameters(), 'lr': self.config['learning_rate']},
            {'params': self.data_encoder.parameters(), 'lr': self.config['learning_rate']}
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        return optimizer
    
    def _create_mcts_engine(self) -> EnhancedMCTSEngine:
        """创建MCTS引擎"""
        return EnhancedMCTSEngine(
            expression_encoder=self.expression_encoder,
            data_encoder=self.data_encoder,
            max_depth=self.config.get('max_depth', 12),
            max_iterations=self.config.get('max_iterations', 1000),
            max_vars=self.config.get('max_variables', 5),
            exploration_constant=self.config.get('exploration_constant', 1.4),
            simulation_count=self.config.get('simulation_count', 10),
            reward_weights=self.config.get('reward_weights', None),
            device=self.device
        )
    
    def _setup_logging(self):
        """设置日志"""
        logger = logging.getLogger('finetune_loop')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        true_expression: str,
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        运行在线微调循环
        
        Args:
            X: 输入数据
            y: 目标值
            true_expression: 真实表达式
            variables: 变量列表
            
        Returns:
            训练结果
        """
        self.logger.info(f"开始在线微调循环，真实表达式: {true_expression}")
        
        # 阶段0：初始化目标嵌入
        target_expr_embedding = self.expression_encoder.encode(true_expression)
        target_data_embedding = self.data_encoder.encode(X, y)
        
        self.logger.info("计算目标嵌入完成")
        
        # 主循环训练
        for epoch in range(self.config.get('mcts_epochs', 50)):
            self.epoch = epoch
            
            self.logger.info(f"=== Epoch {epoch + 1}/{self.config['mcts_epochs']} ===")
            
            # 阶段一：MCTS探索与经验收集
            best_expr = self._mcts_exploration_phase(
                X, y, target_expr_embedding, target_data_embedding, true_expression
            )
            
            # 阶段二：网络微调（基于真实解）
            finetune_loss = self._finetune_phase(
                X, y, true_expression, target_data_embedding
            )
            
            # 评估性能
            performance = self._evaluate_performance(X, y, best_expr)
            
            # 记录性能
            self.performance_history['r2_scores'].append(performance['r2'])
            self.performance_history['best_expressions'].append(str(best_expr))
            self.performance_history['finetune_losses'].append(finetune_loss)
            
            # 更新最佳性能
            if performance['r2'] > self.best_performance['r2']:
                self.best_performance.update({
                    'r2': performance['r2'],
                    'expression': str(best_expr),
                    'epoch': epoch
                })
            
            # 记录日志
            self.logger.info(
                f"Epoch {epoch + 1}: R2 = {performance['r2']:.4f}, "
                f"Finetune Loss = {finetune_loss:.4f}, "
                f"Best Expression = {best_expr}"
            )
            
            # 定期保存
            if (epoch + 1) % self.config.get('save_steps', 10) == 0:
                self.save_checkpoint()
            
            # 提前停止条件
            if performance['r2'] > 0.95:  # 如果R2超过95%，可以提前停止
                self.logger.info(f"达到优秀性能 (R2={performance['r2']:.4f})，提前停止")
                break
        
        self.logger.info("在线微调循环完成！")
        return {
            'best_performance': self.best_performance,
            'performance_history': self.performance_history,
            'final_expression': best_expr
        }
    
    def _mcts_exploration_phase(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_expr_embedding: np.ndarray,
        target_data_embedding: np.ndarray,
        true_expression: str
    ) -> Any:
        """阶段一：MCTS探索与经验收集"""
        
        # 使用真实解引导的MCTS进行探索
        best_expr = self.mcts_engine.fit(
            X, y,
            true_expression=true_expression,
            target_data_embedding=target_data_embedding
        )
        
        # 收集经验到回放池
        mcts_experiences = self.mcts_engine.get_experience_buffer()
        for experience in mcts_experiences:
            experience['epoch'] = self.epoch
            experience['target_expr_embedding'] = target_expr_embedding
            experience['target_data_embedding'] = target_data_embedding
            self.experience_buffer.append(experience)
        
        # 清理MCTS经验池
        self.mcts_engine.clear_experience_buffer()
        
        return best_expr
    
    def _finetune_phase(
        self,
        X: np.ndarray,
        y: np.ndarray,
        true_expression: str,
        target_data_embedding: np.ndarray
    ) -> float:
        """阶段二：网络微调（基于真实解的精准学习）"""
        
        self.expression_encoder.train()
        self.data_encoder.train()
        
        # 使用唯一的、完美的正样本对 (E_true, D_task)
        self.optimizer.zero_grad()
        
        # 计算真实解和数据集的嵌入
        true_expr_embedding = self.expression_encoder.encode(true_expression)
        data_embedding = target_data_embedding
        
        # 转换为张量
        expr_tensor = torch.FloatTensor(true_expr_embedding).to(self.device)
        data_tensor = torch.FloatTensor(data_embedding).to(self.device)
        
        # 计算协同嵌入损失
        # 损失函数目标：最大化真实解对的余弦相似度
        cosine_similarity = F.cosine_similarity(expr_tensor.unsqueeze(0), data_tensor.unsqueeze(0))
        
        # 对比学习损失：1 - cosine_similarity
        # 这样当相似度为1时，损失为0；当相似度为-1时，损失为2
        finetune_loss = 1 - cosine_similarity
        
        # 反向传播
        finetune_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.expression_encoder.parameters()) + list(self.data_encoder.parameters()),
            max_norm=1.0
        )
        
        # 更新权重
        self.optimizer.step()
        
        return finetune_loss.item()
    
    def _evaluate_performance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expression: Any
    ) -> Dict[str, float]:
        """评估当前表达式的性能"""
        try:
            # 使用MCTS引擎的预测功能
            y_pred = self.mcts_engine.predict(X)
            
            # 计算R2分数
            from nd2py.utils import R2_score, RMSE_score
            r2 = R2_score(y, y_pred)
            rmse = RMSE_score(y, y_pred)
            
            return {
                'r2': r2,
                'rmse': rmse,
                'expression': expression
            }
            
        except Exception as e:
            self.logger.warning(f"性能评估时出错: {e}")
            return {
                'r2': -np.inf,
                'rmse': np.inf,
                'expression': expression
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的模型进行预测"""
        return self.mcts_engine.predict(X)
    
    def get_best_expression(self) -> Any:
        """获取最佳表达式"""
        return self.best_performance.get('expression')
    
    def get_performance_history(self) -> Dict[str, List]:
        """获取性能历史"""
        return self.performance_history
    
    def analyze_experience_buffer(self) -> Dict[str, Any]:
        """分析经验回放池"""
        if not self.experience_buffer:
            return {}
        
        experiences = list(self.experience_buffer)
        
        # 统计信息
        r2_scores = [exp.get('r2', 0) for exp in experiences]
        rewards = [exp.get('reward', 0) for exp in experiences]
        expressions = [exp.get('expression') for exp in experiences]
        
        analysis = {
            'total_experiences': len(experiences),
            'r2_stats': {
                'mean': np.mean(r2_scores),
                'std': np.std(r2_scores),
                'min': np.min(r2_scores),
                'max': np.max(r2_scores),
                'median': np.median(r2_scores)
            },
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'median': np.median(rewards)
            },
            'best_expressions': sorted(
                [(str(exp.get('expression', '')), exp.get('r2', -np.inf)) 
                 for exp in experiences],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        
        return analysis
    
    def save_checkpoint(self, checkpoint_path: Optional[str] = None):
        """保存检查点"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.config.get('output_dir', 'models_weights/finetuned/'),
                f'checkpoint_epoch_{self.epoch}'
            )
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存模型权重
        torch.save({
            'expression_encoder_state_dict': self.expression_encoder.state_dict(),
            'data_encoder_state_dict': self.data_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_performance': self.best_performance,
            'performance_history': self.performance_history,
            'config': self.config
        }, os.path.join(checkpoint_path, 'finetune_checkpoint.pt'))
        
        # 保存经验回放池
        experience_data = {
            'buffer': list(self.experience_buffer),
            'analysis': self.analyze_experience_buffer()
        }
        
        import pickle
        with open(os.path.join(checkpoint_path, 'experience_buffer.pkl'), 'wb') as f:
            pickle.dump(experience_data, f)
        
        # 保存配置
        config_path = os.path.join(checkpoint_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"检查点已保存到 {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        # 加载模型权重
        checkpoint = torch.load(
            os.path.join(checkpoint_path, 'finetune_checkpoint.pt'),
            map_location=self.device
        )
        
        self.expression_encoder.load_state_dict(checkpoint['expression_encoder_state_dict'])
        self.data_encoder.load_state_dict(checkpoint['data_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载训练状态
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_performance = checkpoint['best_performance']
        self.performance_history = checkpoint['performance_history']
        
        # 加载经验回放池
        try:
            import pickle
            with open(os.path.join(checkpoint_path, 'experience_buffer.pkl'), 'rb') as f:
                experience_data = pickle.load(f)
                self.experience_buffer = deque(
                    experience_data['buffer'],
                    maxlen=self.config.get('buffer_size', 10000)
                )
        except:
            self.logger.warning("无法加载经验回放池")
        
        self.logger.info(f"检查点已从 {checkpoint_path} 加载")
    
    def save_final_model(self, save_path: str):
        """保存最终模型"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存预训练好的编码器
        expr_path = os.path.join(save_path, 'final_expression_encoder')
        data_path = os.path.join(save_path, 'final_data_encoder')
        
        self.expression_encoder.save_pretrained(expr_path)
        self.data_encoder.save_pretrained(data_path)
        
        # 保存最佳性能
        final_results = {
            'best_performance': self.best_performance,
            'performance_history': self.performance_history,
            'final_expression': str(self.get_best_expression()),
            'experience_analysis': self.analyze_experience_buffer()
        }
        
        with open(os.path.join(save_path, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # 保存MCTS引擎
        import pickle
        with open(os.path.join(save_path, 'mcts_engine.pkl'), 'wb') as f:
            pickle.dump(self.mcts_engine, f)
        
        self.logger.info(f"最终模型已保存到 {save_path}")
    
    def reset(self):
        """重置训练状态"""
        self.global_step = 0
        self.epoch = 0
        self.best_performance = {
            'r2': -np.inf,
            'expression': None,
            'epoch': -1
        }
        self.performance_history = {
            'r2_scores': [],
            'best_expressions': [],
            'finetune_losses': []
        }
        self.experience_buffer.clear()
        self.reward_calculator.reset_history()
        
        self.logger.info("训练状态已重置")