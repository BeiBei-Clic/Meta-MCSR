import torch
import numpy as np
import sys
import os
import time

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from nd2py.utils import R2_score, RMSE_score

from mcts_enhanced import MCTSWithRewardNetwork
from reward_network import RewardNetwork


def load_trained_models(expr_encoder_path='weights/expression_encoder', 
                       reward_network_path='weights/reward_network_final'):
    """加载训练好的模型"""
    models = {}
    
    # 检查表达式嵌入器
    if os.path.exists(expr_encoder_path + '_tokenizer.pkl'):
        print(f"加载表达式嵌入器: {expr_encoder_path}")
        models['expr_encoder'] = expr_encoder_path
    else:
        print(f"警告：未找到表达式嵌入器 {expr_encoder_path}")
    
    # 检查奖励网络
    if os.path.exists(reward_network_path + '_reward_network.pth'):
        print(f"加载奖励网络: {reward_network_path}")
        models['reward_network'] = reward_network_path
    else:
        print(f"警告：未找到奖励网络 {reward_network_path}")
        print("将使用原始MCTS（基于R2分数）")
    
    return models


def create_symbolic_regression_problem():
    """创建符号回归问题示例"""
    problems = [
        {
            'name': '简单线性函数',
            'X': np.random.uniform(-5, 5, (100, 2)),
            'true_function': lambda X: X[:, 0] + 2 * X[:, 1],
            'true_expression': "x1 + 2*x2",
            'description': "y = x1 + 2*x2 + 噪声"
        },
        {
            'name': '多项式函数',
            'X': np.random.uniform(-3, 3, (100, 2)),
            'true_function': lambda X: X[:, 0]**2 + X[:, 1]**2,
            'true_expression': "x1^2 + x2^2",
            'description': "y = x1² + x2² + 噪声"
        },
        {
            'name': '三角函数',
            'X': np.random.uniform(-np.pi, np.pi, (100, 2)),
            'true_function': lambda X: np.sin(X[:, 0]) + np.cos(X[:, 1]),
            'true_expression': "sin(x1) + cos(x2)",
            'description': "y = sin(x1) + cos(x2) + 噪声"
        },
        {
            'name': '复合函数',
            'X': np.random.uniform(-3, 3, (100, 3)),
            'true_function': lambda X: X[:, 0] + 2 * X[:, 1] * np.sin(X[:, 0]) + 0.5 * X[:, 2]**2,
            'true_expression': "x1 + 2*x2*sin(x1) + 0.5*x3^2",
            'description': "y = x1 + 2*x2*sin(x1) + 0.5*x3² + 噪声"
        },
        {
            'name': '指数对数函数',
            'X': np.random.uniform(0.1, 3, (100, 2)),
            'true_function': lambda X: np.exp(X[:, 0]) + np.log(X[:, 1]),
            'true_expression': "exp(x1) + log(x2)",
            'description': "y = exp(x1) + log(x2) + 噪声"
        }
    ]
    
    return problems


def generate_noisy_data(true_function, X, noise_level=0.1):
    """生成带噪声的数据"""
    y_clean = true_function(X)
    y_noisy = y_clean + np.random.normal(0, noise_level, len(y_clean))
    return y_noisy


def run_symbolic_regression(X, y, true_expression=None, models=None, 
                           mcts_params=None, verbose=True):
    """运行符号回归"""
    
    if mcts_params is None:
        mcts_params = {
            'max_depth': 8,
            'max_iterations': 500,
            'max_vars': min(X.shape[1], 3),
            'eta': 0.999,
            'alpha_hybrid': 0.7
        }
    
    start_time = time.time()
    
    # 创建奖励网络
    reward_network = RewardNetwork(
        expr_encoder_path=models.get('expr_encoder'),
        fusion_type='attention'
    )
    
    # 设置数据维度
    input_dim = X.shape[1] if len(X.shape) > 1 else 1
    reward_network.set_data_encoder_dim(input_dim)
    
    # 创建增强MCTS
    mcts = MCTSWithRewardNetwork(
        reward_network=reward_network,
        **mcts_params
    )
    
    # 设置神谕目标
    if true_expression:
        try:
            mcts.set_oracle_target(str(true_expression))
        except:
            print("警告：无法设置神谕目标")
    
    # 训练模型
    if verbose:
        print("开始训练...")
    
    best_expr = mcts.fit(X, y)
    
    training_time = time.time() - start_time
    
    # 评估结果
    r2, rmse = mcts.get_score(X, y)
    
    results = {
        'best_expression': str(best_expr),
        'r2_score': r2,
        'rmse_score': rmse,
        'training_time': training_time,
        'mcts_instance': mcts
    }
    
    return results











def main():
    """主函数"""
    print("基于自学习奖励网络的MCTS符号回归系统")
    print("=" * 60)
    
    # 加载模型
    models = load_trained_models()
    
    # 创建问题集
    problems = create_symbolic_regression_problem()
    
    # 选择第一个问题（简单线性函数）作为示例
    problem = problems[0]
    
    print(f"\n{'='*60}")
    print(f"问题: {problem['name']}")
    print(f"描述: {problem['description']}")
    print(f"真实解: {problem['true_expression']}")
    print(f"{'='*60}")
    
    # 生成数据
    X = problem['X']
    y = generate_noisy_data(problem['true_function'], X)
    
    # 设置MCTS参数
    mcts_params = {
        'max_depth': 8,
        'max_iterations': 500,
        'max_vars': min(X.shape[1], 3),
        'eta': 0.999,
        'alpha_hybrid': 0.7
    }
    
    # 运行增强MCTS
    print("\n使用增强MCTS（奖励网络）")
    print("-" * 30)
    
    result = run_symbolic_regression(
        X, y,
        true_expression=problem['true_expression'],
        models=models,
        mcts_params=mcts_params
    )
    
    # 显示结果
    print(f"\n结果:")
    print(f"找到的解: {result['best_expression']}")
    print(f"R2: {result['r2_score']:.4f}")
    print(f"RMSE: {result['rmse_score']:.4f}")
    print(f"训练时间: {result['training_time']:.2f}s")
    
    # 与真实解比较
    print(f"\n真实解性能（参考）")
    print("-" * 30)
    y_true = problem['true_function'](X)
    r2_true = R2_score(y, y_true)
    rmse_true = RMSE_score(y, y_true)
    print(f"真实解: {problem['true_expression']}")
    print(f"R2: {r2_true:.4f}")
    print(f"RMSE: {rmse_true:.4f}")
    
    # 性能对比
    print(f"\n性能对比")
    print("-" * 30)
    print(f"增强MCTS R2: {result['r2_score']:.4f}")
    print(f"真实解   R2: {r2_true:.4f}")
    
    difference = result['r2_score'] - r2_true
    print(f"与真实解差距: {abs(difference):.4f}")


if __name__ == "__main__":
    main()