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

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from src.symbolic_regression.models.expression_encoder import ExpressionEncoder
from src.symbolic_regression.models.data_encoder import DataEncoder
from src.symbolic_regression.training.finetune_loop import FinetuneLoop
from src.symbolic_regression.utils.data_loader import DataLoader, generate_data


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
        return config
    except ImportError:
        print("请安装pyyaml: pip install pyyaml")
        raise
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return get_default_config()


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
                'buffer_size': 5000
            }
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_seed': 42
    }


def create_test_problem() -> Dict[str, Any]:
    """创建测试问题"""
    # 定义一些测试表达式
    test_expressions = [
        "x1 + 2*x2",
        "x1 * sin(x2) + x2 * cos(x1)",
        "x1^2 + x2^2",
        "exp(-x1) * sin(x2)",
        "log(x1^2 + x2^2)"
    ]
    
    print("\n请选择测试表达式:")
    for i, expr in enumerate(test_expressions, 1):
        print(f"{i}. {expr}")
    
    while True:
        try:
            choice = int(input("\n请输入选择 (1-5): ")) - 1
            if 0 <= choice < len(test_expressions):
                selected_expression = test_expressions[choice]
                break
            else:
                print("无效选择，请输入1-5之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    # 生成数据
    data_loader = DataLoader()
    dataset = data_loader.generate_synthetic_data(
        expression=selected_expression,
        n_samples=1000,
        n_features=2,
        variables_range=(-5, 5),
        noise_level=0.01
    )
    
    return {
        'true_expression': selected_expression,
        'X': dataset.X,
        'y': dataset.y,
        'variables': dataset.variables
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="在线微调循环训练")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--expression', type=str, help='真实表达式（可选）')
    parser.add_argument('--generate-data', action='store_true', help='生成测试数据')
    parser.add_argument('--pretrained-path', type=str, help='预训练模型路径')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    
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
        
        # 加载预训练权重（如果提供）
        if args.pretrained_path:
            logger.info(f"加载预训练权重: {args.pretrained_path}")
            expr_path = os.path.join(args.pretrained_path, 'expression_encoder')
            data_path = os.path.join(args.pretrained_path, 'data_encoder')
            
            if os.path.exists(expr_path):
                expression_encoder = ExpressionEncoder.from_pretrained(expr_path)
                expression_encoder.to(device)
            
            if os.path.exists(data_path):
                data_encoder = DataEncoder.from_pretrained(data_path)
                data_encoder.to(device)
        
        # 创建在线微调循环
        logger.info("创建在线微调循环...")
        finetune_loop = FinetuneLoop(
            expression_encoder=expression_encoder,
            data_encoder=data_encoder,
            config=config['training']['online_finetune'],
            device=device
        )
        
        # 准备数据
        if args.expression:
            true_expression = args.expression
            data_loader = DataLoader()
            dataset = data_loader.generate_synthetic_data(
                expression=true_expression,
                n_samples=1000,
                n_features=2,
                variables_range=(-5, 5),
                noise_level=0.01
            )
            
            problem_data = {
                'true_expression': true_expression,
                'X': dataset.X,
                'y': dataset.y,
                'variables': dataset.variables
            }
        elif args.generate_data or args.test:
            problem_data = create_test_problem()
        else:
            logger.error("请提供--expression参数或使用--generate-data标志")
            return 1
        
        print(f"\n问题设置:")
        print(f"真实表达式: {problem_data['true_expression']}")
        print(f"数据形状: {problem_data['X'].shape}")
        print(f"变量: {problem_data['variables']}")
        
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
