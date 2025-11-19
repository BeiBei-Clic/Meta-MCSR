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
import gc
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
        
        # 内存管理
        self.memory_management_enabled = config.get('memory_management', True)
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        
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
            'finetune_losses': [],
            'diversity_scores': [],
            'composite_scores': []
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
        
        # 检查并记录初始内存状态
        if self.memory_management_enabled and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free_memory = total_memory - memory_allocated
            self.logger.info(f"初始GPU内存状态: {memory_allocated:.2f} GB / {total_memory:.2f} GB (可用: {free_memory:.2f} GB)")
            
            # 如果可用内存少于1GB，发出警告
            if free_memory < 1.0:
                self.logger.warning(f"可用GPU内存较少 ({free_memory:.2f} GB)，可能需要进一步优化")
                # 立即清理一次
                torch.cuda.empty_cache()
                gc.collect()
                
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
                X, y, true_expression, target_data_embedding.detach().cpu().numpy()  # 确保是numpy格式
            )
            
            # 多样性评估：检查当前表达式与历史最佳表达式的差异
            diversity_score = self._evaluate_diversity(best_expr)
            
            # 评估性能
            performance = self._evaluate_performance(X, y, best_expr)
            
            # 综合评分：R2 + 多样性奖励 - 复杂度惩罚
            complexity_penalty = len(str(best_expr)) * 0.001
            composite_score = performance['r2'] + 0.1 * diversity_score - complexity_penalty
            
            # 记录性能
            self.performance_history['r2_scores'].append(performance['r2'])
            self.performance_history['best_expressions'].append(str(best_expr))
            self.performance_history['finetune_losses'].append(finetune_loss)
            self.performance_history['diversity_scores'].append(diversity_score)
            self.performance_history['composite_scores'].append(composite_score)
            
            # 更新最佳性能（基于综合评分）
            if composite_score > getattr(self.best_performance, 'composite_score', -np.inf):
                self.best_performance.update({
                    'r2': performance['r2'],
                    'expression': str(best_expr),
                    'epoch': epoch,
                    'diversity_score': diversity_score,
                    'composite_score': composite_score
                })
            
            # 记录日志
            self.logger.info(
                f"Epoch {epoch + 1}: R2 = {performance['r2']:.4f}, "
                f"Diversity = {diversity_score:.4f}, "
                f"Composite = {composite_score:.4f}, "
                f"Finetune Loss = {finetune_loss:.4f}, "
                f"Best Expression = {best_expr}"
            )
            
            # 内存管理
            if self.memory_management_enabled and torch.cuda.is_available():
                # 清理CUDA缓存
                torch.cuda.empty_cache()
                # 垃圾回收
                gc.collect()
                
                # 记录内存使用
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self.logger.info(
                        f"GPU内存使用: {memory_allocated:.2f}GB / {total_memory:.2f}GB "
                        f"(保留: {memory_reserved:.2f}GB, 空闲: {total_memory - memory_allocated:.2f}GB)"
                    )
            
            # 定期保存
            if (epoch + 1) % self.config.get('save_steps', 10) == 0:
                self.save_checkpoint()
            
            # 更合理的提前停止条件
            if len(self.performance_history['r2_scores']) >= 10:
                recent_r2 = self.performance_history['r2_scores'][-10:]
                r2_std = np.std(recent_r2)
                r2_trend = np.mean(recent_r2[-5:]) - np.mean(recent_r2[:5])
                
                # 如果R2变化很小且标准差很小，可能已经收敛到局部最优
                if r2_std < 0.001 and abs(r2_trend) < 0.001:
                    self.logger.info(f"R2收敛过早 (标准差={r2_std:.6f})，可能存在过拟合，提前停止")
                    break
            
            # 如果多样性过低，也可能存在过拟合
            if hasattr(self, 'performance_history') and len(self.performance_history['diversity_scores']) >= 5:
                recent_diversity = self.performance_history['diversity_scores'][-5:]
                avg_diversity = np.mean(recent_diversity)
                if avg_diversity < 0.01:
                    self.logger.info(f"多样性过低 ({avg_diversity:.4f})，可能存在过拟合，提前停止")
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
        """阶段二：网络微调（内存优化版本）"""

        self.expression_encoder.train()
        self.data_encoder.train()

        self.optimizer.zero_grad()

        # 使用with torch.no_grad()来避免额外的梯度计算
        with torch.no_grad():
            # 计算真实解和数据集的嵌入
            true_expr_embedding = self.expression_encoder.encode(true_expression)
            data_embedding = target_data_embedding

            # 确保数据在正确的设备上，并且是张量格式
            if isinstance(true_expr_embedding, np.ndarray):
                true_expr_embedding = torch.FloatTensor(true_expr_embedding)

            if isinstance(data_embedding, np.ndarray):
                data_embedding = torch.FloatTensor(data_embedding)

            # 将张量移动到设备
            expr_tensor = true_expr_embedding.to(self.device)
            data_tensor = data_embedding.to(self.device)

            # 计算余弦相似度
            cosine_similarity = F.cosine_similarity(expr_tensor.unsqueeze(0), data_tensor.unsqueeze(0))

            # 减少噪声生成的内存使用
            noise_factor = 0.1 * (1.0 - self.epoch / self.config['mcts_epochs'])  # 随训练进度减少噪声
            
            # 原地操作减少内存分配
            expr_tensor.add_(torch.randn_like(expr_tensor) * noise_factor)
            data_tensor.add_(torch.randn_like(data_tensor) * noise_factor)

            # 重新计算带噪声的余弦相似度
            noisy_cosine = F.cosine_similarity(expr_tensor.unsqueeze(0), data_tensor.unsqueeze(0))

        # 分离的损失计算
        finetune_loss = 1 - noisy_cosine.detach()

        # 内存优化的正则化计算
        if self.epoch % 5 == 0:  # 每5个epoch计算一次正则化损失
            # 添加L2正则化
            l2_reg = 0.01
            reg_loss = 0.0
            for param in list(self.expression_encoder.parameters()) + list(self.data_encoder.parameters()):
                reg_loss += torch.norm(param, p=2)
            finetune_loss += l2_reg * reg_loss

            # 添加熵正则化鼓励多样性
            entropy_reg = -0.001 * torch.mean(cosine_similarity**2)
            finetune_loss += entropy_reg

        # 反向传播
        finetune_loss.backward(retain_graph=False)  # 避免保留计算图

        # 更严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.expression_encoder.parameters()) + list(self.data_encoder.parameters()),
            max_norm=0.5
        )

        # 更新权重
        self.optimizer.step()

        # 立即清理临时变量
        if self.memory_management_enabled and torch.cuda.is_available():
            del expr_tensor, data_tensor, cosine_similarity, noisy_cosine
            if 'reg_loss' in locals():
                del reg_loss
            if 'entropy_reg' in locals():
                del entropy_reg
        
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
    
    def _evaluate_diversity(self, expression: Any) -> float:
        """评估当前表达式与历史最佳表达式之间的多样性"""
        try:
            current_expr_str = str(expression)
            
            # 与历史表达式比较
            diversity_scores = []
            for prev_expr in self.performance_history['best_expressions']:
                if prev_expr and prev_expr != current_expr_str:
                    # 简单的字符串差异度量
                    current_len = len(current_expr_str)
                    prev_len = len(prev_expr)
                    len_diff = abs(current_len - prev_len)
                    
                    # 字符级别的差异
                    common_chars = set(current_expr_str) & set(prev_expr)
                    char_diversity = (len(set(current_expr_str) | set(prev_expr)) - len(common_chars)) / max(current_len, prev_len)
                    
                    diversity_scores.append(char_diversity)
            
            # 返回平均多样性分数
            return np.mean(diversity_scores) if diversity_scores else 1.0
            
        except Exception as e:
            self.logger.warning(f"多样性评估时出错: {e}")
            return 0.0
    
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
            'finetune_losses': [],
            'diversity_scores': [],
            'composite_scores': []
        }
        self.experience_buffer.clear()
        self.reward_calculator.reset_history()
        
        self.logger.info("训练状态已重置")
    
    def check_memory_status(self) -> Dict[str, float]:
        """检查GPU内存状态"""
        if not torch.cuda.is_available():
            return {'cuda_available': False}
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_memory = total_memory - memory_allocated
        
        return {
            'cuda_available': True,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'total_memory_gb': total_memory,
            'free_memory_gb': free_memory,
            'memory_usage_percent': (memory_allocated / total_memory) * 100
        }
    
    def clear_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("GPU内存已清理")