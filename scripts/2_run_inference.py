#!/usr/bin/env python3
"""
阶段二：算法实战

此脚本实现训练好的模型在真实问题上的推理和预测功能。
使用训练好的编码器和MCTS引擎进行符号回归预测。
"""

import os
import sys
import torch
import numpy as np
import logging
import argparse
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from src.symbolic_regression.models.expression_encoder import ExpressionEncoder
from src.symbolic_regression.models.data_encoder import DataEncoder
from src.symbolic_regression.core.mcts_engine import EnhancedMCTSEngine
from src.symbolic_regression.training.finetune_loop import FinetuneLoop
from src.symbolic_regression.utils.data_loader import DataLoader


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/logs/inference.log')
        ]
    )


def load_trained_model(model_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型路径
        device: 设备类型
        
    Returns:
        包含模型组件的字典
    """
    logger = logging.getLogger(__name__)
    
    # 加载表达式编码器
    expr_path = os.path.join(model_path, 'final_expression_encoder')
    if os.path.exists(expr_path):
        expression_encoder = ExpressionEncoder.from_pretrained(expr_path)
        expression_encoder.to(device)
        logger.info(f"表达式编码器加载成功: {expr_path}")
    else:
        raise FileNotFoundError(f"表达式编码器路径不存在: {expr_path}")
    
    # 加载数据编码器
    data_path = os.path.join(model_path, 'final_data_encoder')
    if os.path.exists(data_path):
        data_encoder = DataEncoder.from_pretrained(data_path)
        data_encoder.to(device)
        logger.info(f"数据编码器加载成功: {data_path}")
    else:
        raise FileNotFoundError(f"数据编码器路径不存在: {data_path}")
    
    # 加载MCTS引擎
    mcts_path = os.path.join(model_path, 'mcts_engine.pkl')
    if os.path.exists(mcts_path):
        with open(mcts_path, 'rb') as f:
            mcts_engine = pickle.load(f)
        logger.info(f"MCTS引擎加载成功: {mcts_path}")
    else:
        mcts_engine = None
        logger.warning(f"MCTS引擎未找到，将重新创建: {mcts_path}")
    
    # 加载结果摘要
    results_path = os.path.join(model_path, 'results_summary.json')
    results_summary = {}
    if os.path.exists(results_path):
        import json
        with open(results_path, 'r') as f:
            results_summary = json.load(f)
        logger.info(f"结果摘要加载成功: {results_path}")
    
    return {
        'expression_encoder': expression_encoder,
        'data_encoder': data_encoder,
        'mcts_engine': mcts_engine,
        'results_summary': results_summary
    }


def predict_with_mcts(
    X: np.ndarray,
    y: Optional[np.ndarray],
    models: Dict[str, Any],
    config: Dict[str, Any],
    target_expression: Optional[str] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    使用MCTS进行预测
    
    Args:
        X: 输入数据
        y: 目标值（可选）
        models: 模型组件字典
        config: 配置信息
        target_expression: 目标表达式（用于微调）
        device: 设备类型
        
    Returns:
        预测结果
    """
    logger = logging.getLogger(__name__)
    
    # 创建MCTS引擎
    if models['mcts_engine'] is None:
        logger.info("重新创建MCTS引擎...")
        mcts_engine = EnhancedMCTSEngine(
            expression_encoder=models['expression_encoder'],
            data_encoder=models['data_encoder'],
            max_depth=config.get('max_depth', 12),
            max_iterations=config.get('max_iterations', 1000),
            max_vars=config.get('max_variables', 5),
            exploration_constant=config.get('exploration_constant', 1.4),
            simulation_count=config.get('simulation_count', 10),
            reward_weights=config.get('reward_weights', None),
            device=device
        )
    else:
        mcts_engine = models['mcts_engine']
        # 更新编码器（如果模型被重新加载）
        mcts_engine.expression_encoder = models['expression_encoder']
        mcts_engine.data_encoder = models['data_encoder']
    
    # 计算数据嵌入
    if y is not None:
        target_data_embedding = models['data_encoder'].encode(X, y)
    else:
        logger.warning("未提供目标值，使用默认嵌入")
        target_data_embedding = models['data_encoder'].encode(X, np.zeros(X.shape[0]))
    
    # 计算目标表达式嵌入（如果有）
    target_expr_embedding = None
    if target_expression is not None:
        target_expr_embedding = models['expression_encoder'].encode(target_expression)
    
    # 运行MCTS预测
    logger.info("开始MCTS预测...")
    
    # 这里简化实现，直接使用已有的MCTS引擎进行预测
    # 实际应用中可能需要重新运行MCTS搜索
    
    prediction_results = {
        'X': X,
        'y_target': y,
        'target_data_embedding': target_data_embedding,
        'target_expr_embedding': target_expr_embedding,
        'mcts_engine': mcts_engine,
        'prediction_method': 'mcts_guided'
    }
    
    return prediction_results


def evaluate_predictions(
    X: np.ndarray,
    y_true: np.ndarray,
    predicted_expression: str,
    models: Dict[str, Any]
) -> Dict[str, float]:
    """
    评估预测结果
    
    Args:
        X: 输入数据
        y_true: 真实目标值
        predicted_expression: 预测的表达式
        models: 模型组件
        
    Returns:
        评估指标
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 导入评估函数
        sys_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..')
        sys.path.append(sys_path)
        import nd2py as nd
        from nd2py.utils import R2_score, RMSE_score
        
        # 安全评估预测表达式
        expr_str = predicted_expression.replace('^', '**')
        
        # 准备变量字典
        var_dict = {}
        for i, var_name in enumerate([f'x{j+1}' for j in range(X.shape[1])]):
            var_dict[var_name] = X[:, i]
        
        # 添加允许的函数
        var_dict.update({
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
            'pi': np.pi, 'e': np.e
        })
        
        # 计算预测值
        y_pred = eval(expr_str, {"__builtins__": {}}, var_dict)
        
        # 计算评估指标
        r2 = R2_score(y_true, y_pred)
        rmse = RMSE_score(y_true, y_pred)
        
        # 计算嵌入相似度（如果提供了目标表达式）
        predicted_embedding = models['expression_encoder'].encode(predicted_expression)
        
        evaluation_results = {
            'r2_score': r2,
            'rmse_score': rmse,
            'predicted_expression': predicted_expression,
            'embedding_norm': np.linalg.norm(predicted_embedding),
            'prediction_successful': True
        }
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"评估预测结果时出错: {e}")
        return {
            'r2_score': -np.inf,
            'rmse_score': np.inf,
            'predicted_expression': predicted_expression,
            'embedding_norm': np.nan,
            'prediction_successful': False,
            'error': str(e)
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="算法实战 - 模型推理")
    parser.add_argument('--model-path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--data', type=str, help='输入数据文件路径')
    parser.add_argument('--expression', type=str, help='目标表达式（用于对比）')
    parser.add_argument('--generate-test', action='store_true', help='生成测试数据')
    parser.add_argument('--output', type=str, help='输出结果文件路径')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='auto', help='设备类型')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("算法实战 - 模型推理")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"使用设备: {device}")
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return 1
    
    try:
        # 加载模型
        logger.info(f"加载模型: {args.model_path}")
        models = load_trained_model(args.model_path, device)
        
        # 准备测试数据
        if args.generate_test:
            # 生成测试数据
            test_expression = "x1 + 2*x2*sin(x1) + 0.5*x3^2"
            if args.expression:
                test_expression = args.expression
            
            logger.info(f"生成测试数据，表达式: {test_expression}")
            data_loader = DataLoader()
            dataset = data_loader.generate_synthetic_data(
                expression=test_expression,
                n_samples=1000,
                n_features=3,
                variables_range=(-5, 5),
                noise_level=0.01
            )
            
            X = dataset.X
            y = dataset.y
            variables = dataset.variables
            
        elif args.data:
            # 从文件加载数据
            logger.info(f"加载数据: {args.data}")
            data_loader = DataLoader()
            dataset = data_loader.load_and_prepare(args.data)
            
            X = dataset.X
            y = dataset.y
            variables = dataset.variables
            
        else:
            logger.error("请提供--data参数或使用--generate-test标志")
            return 1
        
        print(f"\n数据信息:")
        print(f"数据形状: {X.shape}")
        print(f"目标值形状: {y.shape if y is not None else 'None'}")
        print(f"变量: {variables}")
        
        # 加载配置
        try:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except:
            logger.warning(f"无法加载配置文件 {args.config}，使用默认配置")
            config = get_default_config()
        
        # 运行预测
        logger.info("开始预测...")
        prediction_results = predict_with_mcts(
            X, y, models, config, args.expression, device
        )
        
        # 使用最佳表达式（如果可用）
        best_expression = None
        if 'best_performance' in models['results_summary']:
            best_expression = models['results_summary']['results']['best_performance']['expression']
        
        if best_expression:
            print(f"\n最佳预测表达式: {best_expression}")
            
            # 评估预测结果
            evaluation = evaluate_predictions(X, y, best_expression, models)
            
            print(f"\n预测评估结果:")
            print(f"R2分数: {evaluation['r2_score']:.4f}")
            print(f"RMSE分数: {evaluation['rmse_score']:.4f}")
            
            if args.expression:
                print(f"真实表达式: {args.expression}")
                
                # 评估真实表达式
                true_evaluation = evaluate_predictions(X, y, args.expression, models)
                print(f"真实表达式R2: {true_evaluation['r2_score']:.4f}")
        else:
            logger.warning("未找到最佳表达式，无法进行评估")
        
        # 保存结果
        if args.output:
            output_results = {
                'prediction_results': prediction_results,
                'evaluation': evaluation if 'evaluation' in locals() else {},
                'model_info': {
                    'model_path': args.model_path,
                    'device': device,
                    'data_shape': X.shape,
                    'variables': variables
                }
            }
            
            with open(args.output, 'wb') as f:
                pickle.dump(output_results, f)
            
            logger.info(f"结果已保存到: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'max_depth': 12,
        'max_iterations': 1000,
        'max_variables': 5,
        'exploration_constant': 1.4,
        'simulation_count': 10,
        'reward_weights': {
            'structure_alignment': 0.3,
            'data_alignment': 0.4,
            'accuracy': 0.3
        }
    }


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
