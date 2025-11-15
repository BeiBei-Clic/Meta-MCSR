import numpy as np
import torch
import sys
import os
import time

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from nd2py.utils import R2_score, RMSE_score
from self_learning_mcts import SelfLearningMCTS


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
        }
    ]
    
    return problems


def generate_noisy_data(true_function, X, noise_level=0.1):
    """生成带噪声的数据"""
    y_clean = true_function(X)
    y_noisy = y_clean + np.random.normal(0, noise_level, len(y_clean))
    return y_noisy


def main():
    """主函数"""
    print("基于自学习奖励网络的MCTS符号回归系统")
    print("=" * 60)
    
    # 检查表达式嵌入器模型是否存在
    expr_encoder_path = 'weights/expression_encoder'
    if not os.path.exists(expr_encoder_path + '_tokenizer.pkl'):
        print(f"错误：未找到表达式嵌入器模型 {expr_encoder_path}")
        print("请先运行 expression_encoder_training.py 进行预训练")
        return
    
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
    
    # 创建自学习MCTS实例
    print("\n初始化自学习MCTS...")
    mcts = SelfLearningMCTS(
        max_depth=6,
        max_iterations=200,
        max_vars=min(X.shape[1], 3),
        alpha_hybrid=0.7,
        experience_buffer_size=2000
    )
    
    # 运行自学习训练
    print("\n开始自学习训练...")
    start_time = time.time()
    
    best_expr = mcts.fit(
        X, y, 
        true_expression=problem['true_expression'],
        num_epochs=3,
        mcts_iterations_per_epoch=200,
        training_epochs_per_epoch=1
    )
    
    training_time = time.time() - start_time
    
    # 评估结果
    r2, rmse = mcts.get_score(X, y)
    
    # 显示结果
    print(f"\n结果:")
    print(f"找到的解: {best_expr}")
    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"训练时间: {training_time:.2f}s")
    
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
    print(f"自学习MCTS R2: {r2:.4f}")
    print(f"真实解     R2: {r2_true:.4f}")
    
    difference = r2 - r2_true
    print(f"与真实解差距: {abs(difference):.4f}")
    
    # 显示训练统计
    if hasattr(mcts, 'epoch_stats') and 'epochs' in mcts.epoch_stats:
        print(f"\n训练统计:")
        print("-" * 30)
        for i, epoch_info in enumerate(mcts.epoch_stats['epochs']):
            print(f"Epoch {i+1}: R2={epoch_info['best_r2']:.4f}, "
                  f"经验={epoch_info['experience_count']}, "
                  f"损失={epoch_info['training_loss']:.4f}")


if __name__ == "__main__":
    main()