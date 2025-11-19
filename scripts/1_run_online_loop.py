#!/usr/bin/env python3
"""
阶段一：在线主循环训练

此脚本实现MCTS探索与真实解引导微调的集成算法。
在加载预训练权重的基础上，针对已知真实解的任务进行深度优化。
"""

import os
import sys
import torch
import numpy as np
import logging
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

# 设置CUDA内存管理
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from symbolic_regression.models.expression_encoder import ExpressionEncoder
from symbolic_regression.models.data_encoder import DataEncoder
from symbolic_regression.training.finetune_loop import FinetuneLoop
from symbolic_regression.utils.data_loader import DataLoader, generate_data


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/logs/finetune.log')
        ]
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 确保数值类型转换，特别是学习率等关键参数
        if 'training' in config:
            for training_type in config['training'].values():
                if 'learning_rate' in training_type:
                    training_type['learning_rate'] = float(training_type['learning_rate'])
                if 'weight_decay' in training_type:
                    training_type['weight_decay'] = float(training_type['weight_decay'])
                if 'batch_size' in training_type:
                    training_type['batch_size'] = int(training_type['batch_size'])
        
        return config
    except ImportError:
        print("请安装pyyaml: pip install pyyaml")
        raise
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return get_default_config()


def load_pretrain_data(pretrain_data_path: str = "data/pretrain/progress_100000/datasets.txt") -> Optional[Dict[str, Any]]:
    """加载预训练数据"""
    if not os.path.exists(pretrain_data_path):
        print(f"警告：预训练数据文件 {pretrain_data_path} 不存在")
        return None
    
    try:
        # 加载预训练数据，解析特定格式
        pretrain_data = []
        current_expression = None
        
        with open(pretrain_data_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        max_expressions = 3  # 限制只处理前3个表达式用于测试
        count = 0
        
        while i < len(lines) and count < max_expressions:
            line = lines[i].strip()
            
            # 查找表达式定义
            if line.startswith('Expression: '):
                current_expression = line.replace('Expression: ', '').strip()
                print(f"找到表达式 {count+1}: {current_expression}")
                
                # 跳过Sample input data行
                i += 1
                if i < len(lines) and lines[i].strip() == 'Sample input data:':
                    i += 1
                    
                    # 收集这个表达式的样本数据
                    samples = []
                    while i < len(lines):
                        sample_line = lines[i].strip()
                        
                        # 检查是否到达下一个表达式或文件结尾
                        if lines[i].strip().startswith('===') or lines[i].strip().startswith('Expression:'):
                            break
                            
                        # 解析样本数据
                        if sample_line.startswith('Sample ') and 'X=' in sample_line and 'y=' in sample_line:
                            try:
                                # 使用正则表达式解析
                                import re
                                match = re.search(r'X=\[(.*?)\], y=(.*)', sample_line)
                                if match:
                                    x_value = float(match.group(1))
                                    y_value = float(match.group(2))
                                    
                                    samples.append({'X': x_value, 'y': y_value})
                                else:
                                    print(f"  正则解析失败: {sample_line[:50]}...")
                            except (ValueError, IndexError) as e:
                                # 如果解析失败，跳过这个样本
                                print(f"  解析样本失败: {sample_line[:50]}... (错误: {e})")
                                pass
                        
                        i += 1
                    
                    # 如果有表达式和数据，添加到pretrain_data
                    if current_expression and samples:
                        pretrain_data.append({
                            'expression': current_expression,
                            'samples': samples
                        })
                        print(f"  加载样本数量: {len(samples)}")
                        count += 1
            else:
                i += 1
        
        print(f"成功加载 {len(pretrain_data)} 个预训练样本")
        return pretrain_data
    except Exception as e:
        print(f"加载预训练数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'model': {
            'expression_encoder': {
                'embedding_dim': 512,
                'n_heads': 8,
                'n_layers': 6,
                'dropout': 0.1,
                'max_seq_length': 128
            },
            'data_encoder': {
                'embedding_dim': 512,
                'n_heads': 8,
                'n_layers': 6,
                'dropout': 0.1,
                'max_features': 100
            }
        },
        'training': {
            'online_finetune': {
                'learning_rate': 1e-6,  # 极低学习率
                'mcts_epochs': 20,      # 减少epochs以便快速测试
                'max_depth': 12,
                'max_iterations': 500,  # 减少迭代次数
                'max_variables': 5,
                'exploration_constant': 1.4,
                'simulation_count': 10,
                'reward_weights': {
                    'structure_alignment': 0.3,
                    'data_alignment': 0.4,
                    'accuracy': 0.3
                },
                'eval_steps': 5,
                'save_steps': 5,
                'output_dir': 'models_weights/finetuned/',
                'buffer_size': 5000,
                # 内存管理选项
                'memory_management': True,
                'gradient_checkpointing': False
            }
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_seed': 42
    }





def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="在线微调循环训练")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("在线微调循环 - 阶段一")
    print("MCTS探索与真实解引导微调的集成算法")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 加载配置
    try:
        config = load_config(args.config)
        logger.info("配置加载成功")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return 1
    
    # 设置随机种子
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    logger.info(f"随机种子设置为: {config['random_seed']}")
    
    # 设置设备
    device = config['device']
    logger.info(f"使用设备: {device}")
    
    try:
        # 创建模型
        logger.info("初始化模型...")
        expression_encoder = ExpressionEncoder(
            **config['model']['expression_encoder']
        )
        data_encoder = DataEncoder(
            **config['model']['data_encoder']
        )
        
        # 自动加载预训练权重
        pretrained_path = 'models_weights/pretrained/'
        if 'training' in config and 'pretrain' in config['training']:
            # 从配置文件获取预训练权重路径
            pretrained_path = config['training']['pretrain'].get('output_dir', pretrained_path)
        
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"自动加载预训练权重: {pretrained_path}")
            expr_path = os.path.join(pretrained_path, 'expression_encoder')
            data_path = os.path.join(pretrained_path, 'data_encoder')
            
            if os.path.exists(expr_path):
                expression_encoder = ExpressionEncoder.from_pretrained(expr_path)
                expression_encoder.to(device)
                logger.info("表达式编码器预训练权重加载成功")
            
            if os.path.exists(data_path):
                data_encoder = DataEncoder.from_pretrained(data_path)
                data_encoder.to(device)
                logger.info("数据编码器预训练权重加载成功")
        else:
            logger.warning("未找到预训练权重，使用随机初始化")
        
        # 自动加载预训练数据
        pretrain_data = None
        pretrain_data_path = "data/pretrain/progress_100000/datasets.txt"
        
        pretrain_data = load_pretrain_data(pretrain_data_path)
        if pretrain_data:
            logger.info(f"从 {pretrain_data_path} 加载了预训练数据")
        
        # 创建在线微调循环
        logger.info("创建在线微调循环...")
        finetune_loop = FinetuneLoop(
            expression_encoder=expression_encoder,
            data_encoder=data_encoder,
            config=config['training']['online_finetune'],
            device=device
        )
        
        # 准备数据
        if pretrain_data:
            # 选择不重复的表达式进行测试，避免过拟合
            selected_sample = None
            target_expression = None
            
            # 查找不重复的表达式
            expression_counts = {}
            for sample in pretrain_data:
                expr = sample['expression']
                expression_counts[expr] = expression_counts.get(expr, 0) + 1
            
            # 选择重复次数较少的表达式
            candidates = [sample for sample in pretrain_data 
                         if expression_counts[sample['expression']] <= 5]
            
            if candidates:
                # 选择第二个表达式（避免第一个总是exp(x1)）
                if len(candidates) > 1:
                    selected_sample = candidates[1]
                else:
                    selected_sample = candidates[0]
            else:
                # 如果都重复，使用第一个
                selected_sample = pretrain_data[0]
            
            true_expression = selected_sample['expression']
            samples = selected_sample['samples']
            logger.info(f"使用表达式进行测试: {true_expression} (在预训练数据中出现{expression_counts[true_expression]}次)")
            
            # 使用预训练数据中的实际样本
            X = np.array([[sample['X']] for sample in samples]).astype(np.float32)
            y = np.array([sample['y'] for sample in samples]).astype(np.float32)

            problem_data = {
                'true_expression': true_expression,
                'X': X,
                'y': y,
                'variables': ['x1']  # 单变量表达式
            }
        else:
            logger.error("预训练数据文件不存在或加载失败")
            return 1
        
        print(f"\n问题设置:")
        print(f"真实表达式: {problem_data['true_expression']}")
        print(f"数据形状: {problem_data['X'].shape}")
        print(f"变量: {problem_data['variables']}")
        
        # 显示预训练状态
        if pretrain_data:
            print(f"已加载预训练数据样本数量: {len(pretrain_data)}")
            print(f"预训练数据来源: {pretrain_data_path}")
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"预训练权重来源: {pretrained_path}")
        else:
            print("使用随机初始化的模型权重")
        
        # 开始在线微调
        logger.info("开始在线微调循环...")
        results = finetune_loop.fit(
            X=problem_data['X'],
            y=problem_data['y'],
            true_expression=problem_data['true_expression'],
            variables=problem_data['variables']
        )
        
        # 保存结果
        final_model_path = os.path.join(
            config['training']['online_finetune']['output_dir'],
            'final_model'
        )
        finetune_loop.save_final_model(final_model_path)
        
        # 打印最终结果
        print("\n" + "=" * 60)
        print("在线微调完成！")
        print("=" * 60)
        print(f"最佳R2分数: {results['best_performance']['r2']:.4f}")
        print(f"最佳表达式: {results['best_performance']['expression']}")
        print(f"真实表达式: {problem_data['true_expression']}")
        print(f"最终模型保存路径: {final_model_path}")
        
        # 性能对比
        true_r2 = calculate_r2_score(problem_data['X'], problem_data['y'], problem_data['true_expression'])
        print(f"真实表达式R2分数: {true_r2:.4f}")
        
        # 保存结果到文件
        results_summary = {
            'problem': {
                'true_expression': problem_data['true_expression'],
                'data_shape': problem_data['X'].shape,
                'variables': problem_data['variables']
            },
            'results': {
                'best_performance': results['best_performance'],
                'true_expression_r2': true_r2,
                'convergence_epoch': results['best_performance']['epoch']
            }
        }
        
        import json
        with open(os.path.join(final_model_path, 'results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        return 0
        
    except Exception as e:
        logger.error(f"在线微调过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


def calculate_r2_score(X, y, expression):
    """计算表达式的R2分数"""
    try:
        # 导入nd2py包
        sys_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..')
        sys.path.append(sys_path)
        import nd2py as nd
        from nd2py.utils import R2_score
        
        # 解析表达式
        expr_str = expression.replace('^', '**')
        
        # 安全的表达式求值
        allowed_names = {
            'x1': X[:, 0], 'x2': X[:, 1], 'sin': np.sin, 'cos': np.cos,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs
        }
        
        y_pred = eval(expr_str, {"__builtins__": {}}, allowed_names)
        
        return R2_score(y, y_pred)
        
    except Exception:
        return -np.inf


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
