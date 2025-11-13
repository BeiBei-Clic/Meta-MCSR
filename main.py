#!/usr/bin/env python3
"""
蒙特卡洛树搜索符号回归主程序
"""

import numpy as np
import sys
import os

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from nd2py.utils import R2_score, RMSE_score

# 导入核心模块
from mcts_core import MCTSSymbolicRegression

def main():
    """主函数"""
    print("蒙特卡洛树搜索符号回归测试")
    print("=" * 50)
    
    # 生成示例数据：y = x1 + 2*x2*sin(x1) + 0.5*x3^2
    np.random.seed(42)
    n_samples = 100
    X = np.random.uniform(-5, 5, (n_samples, 3))  # 3个特征
    y = X[:, 0] + 2 * X[:, 1] * np.sin(X[:, 0]) + 0.5 * X[:, 2]**2 + np.random.normal(0, 0.1, n_samples)
    
    # 创建并训练模型
    model = MCTSSymbolicRegression(max_depth=5, max_iterations=1000, max_vars=5)
    
    print("开始训练...")
    best_expr = model.fit(X, y)
    print(f"最佳表达式: {best_expr}")
    
    # 评估模型
    r2, rmse = model.get_score(X, y)
    print(f"R2 分数: {r2:.4f}")
    print(f"RMSE 分数: {rmse:.4f}")
    
    print("=" * 50)
    return 0

if __name__ == "__main__":
    main()