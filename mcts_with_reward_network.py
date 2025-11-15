#!/usr/bin/env python3
"""
Meta-MCSR: 基于自学习奖励网络的符号回归系统

支持多种测试模式：
- composite: 复合函数测试
- dataset-file: 自定义数据集测试
"""

import sys
import os
import argparse

def check_dependencies():
    """检查依赖是否满足"""
    missing_deps = []
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    if missing_deps:
        print(f"错误：缺少依赖库: {', '.join(missing_deps)}")
        print("请安装依赖: pip install torch numpy")
        return False
    
    # 添加项目路径
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
    
    try:
        import nd2py as nd
        from nd2py.utils import R2_score, RMSE_score
        from src.mcts import MCTSWithRewardNetwork
        from src.reward_network import RewardNetwork
    except ImportError as e:
        print(f"错误：项目依赖导入失败 - {e}")
        print("请确保src目录和nd2py_package目录存在")
        return False
    
    return True

def create_composite_function_problem():
    """创建复合函数测试问题"""
    import numpy as np
    
    problem = {
        'name': '复合函数',
        'X': np.random.uniform(-3, 3, (100, 3)),
        'true_function': lambda X: X[:, 0] + 2 * X[:, 1] * np.sin(X[:, 0]) + 0.5 * X[:, 2]**2,
        'true_expression': "x1 + 2*x2*sin(x1) + 0.5*x3^2",
        'description': "y = x1 + 2*x2*sin(x1) + 0.5*x3² + 噪声"
    }
    return problem


def load_dataset_from_file(file_path):
    """从文件加载数据集"""
    import numpy as np
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    try:
        data = np.load(file_path)
        if 'X' in data and 'y' in data:
            X = data['X']
            y = data['y']
            return {
                'X': X,
                'y': y,
                'name': os.path.basename(file_path),
                'description': f"从文件加载的数据集: {file_path}",
                'true_expression': None  # 用户数据没有真实表达式
            }
        else:
            raise ValueError(f"数据集文件必须包含'X'和'y'数组")
    except Exception as e:
        raise ValueError(f"加载数据集失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Meta-MCSR: 基于自学习奖励网络的符号回归系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python mcts_with_reward_network.py composite                # 运行复合函数测试
  python mcts_with_reward_network.py dataset-file --dataset-path data.npz  # 运行指定数据集文件测试
  python mcts_with_reward_network.py --help                  # 显示帮助信息

数据集文件格式:
  - 支持numpy .npz格式
  - 必须包含'X'（输入数据）和'y'（输出数据）数组
  - X的形状应该是 (n_samples, n_features)
  - y的形状应该是 (n_samples,)
        """
    )
    
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['composite', 'dataset-file'],
        help='测试模式: composite (复合函数测试) 或 dataset-file (指定数据集文件测试)'
    )
    
    parser.add_argument(
        '--dataset-path',
        help='dataset-file模式下的数据集文件路径 (numpy .npz格式，需要包含X和y数组)'
    )
    
    parser.add_argument(
        '--expr-encoder',
        default='weights/expression_encoder',
        help='表达式编码器模型路径'
    )
    
    parser.add_argument(
        '--reward-network',
        default='weights/reward_network_final',
        help='奖励网络模型路径'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=500,
        help='MCTS最大迭代次数'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=8,
        help='MCTS最大深度'
    )
    
    # 如果没有提供参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    print("Meta-MCSR: 基于自学习奖励网络的符号回归系统")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装所需的依赖库后再运行程序。")
        print("依赖安装命令: pip install torch numpy")
        return
    
    # 导入所需模块
    import torch
    import numpy as np
    import time
    from nd2py.utils import R2_score, RMSE_score
    from src.mcts import MCTSWithRewardNetwork
    from src.reward_network import RewardNetwork
    
    def load_trained_models():
        """加载训练好的模型"""
        models = {}
        
        # 检查表达式嵌入器
        expr_path = args.expr_encoder
        if os.path.exists(expr_path + '_tokenizer.pkl'):
            print(f"加载表达式嵌入器: {expr_path}")
            models['expr_encoder'] = expr_path
        else:
            print(f"警告：未找到表达式嵌入器 {expr_path}")
        
        # 检查奖励网络
        reward_path = args.reward_network
        if os.path.exists(reward_path + '_reward_network.pth'):
            print(f"加载奖励网络: {reward_path}")
            models['reward_network'] = reward_path
        else:
            print(f"警告：未找到奖励网络 {reward_path}")
            print("将使用原始MCTS（基于R2分数）")
        
        return models
    
    def generate_noisy_data(true_function, X, noise_level=0.1):
        """生成带噪声的数据"""
        y_clean = true_function(X)
        y_noisy = y_clean + np.random.normal(0, noise_level, len(y_clean))
        return y_noisy
    
    def run_symbolic_regression(X, y, true_expression=None, models=None, verbose=True):
        """运行符号回归"""
        # 设置MCTS参数
        mcts_params = {
            'max_depth': args.max_depth,
            'max_iterations': args.max_iterations,
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
    
    def run_composite_function_test():
        """运行复合函数测试"""
        print("复合函数符号回归测试")
        print("=" * 50)
        
        # 加载模型
        models = load_trained_models()
        
        # 创建复合函数问题
        problem = create_composite_function_problem()
        
        print(f"\n问题: {problem['name']}")
        print(f"描述: {problem['description']}")
        print(f"真实解: {problem['true_expression']}")
        
        # 生成数据
        X = problem['X']
        y = generate_noisy_data(problem['true_function'], X)
        
        # 运行增强MCTS
        print("\n使用增强MCTS（奖励网络）")
        print("-" * 30)
        
        result = run_symbolic_regression(
            X, y,
            true_expression=problem['true_expression'],
            models=models
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
    
    def run_dataset_file_test():
        """运行指定数据集文件测试"""
        if not args.dataset_path:
            print("错误：使用dataset-file模式必须指定 --dataset-path 参数")
            print("示例: python mcts_with_reward_network.py dataset-file --dataset-path data.npz")
            return
            
        print("数据集文件符号回归测试")
        print("=" * 50)
        
        # 加载模型
        models = load_trained_models()
        
        try:
            # 加载数据集
            dataset = load_dataset_from_file(args.dataset_path)
            X = dataset['X']
            y = dataset['y']
            
            print(f"\n数据集: {dataset['name']}")
            print(f"描述: {dataset['description']}")
            print(f"数据形状: X={X.shape}, y={y.shape}")
            
            # 检查数据维度是否合适
            if X.shape[1] > 5:
                print(f"警告：特征维度较高 ({X.shape[1]})，可能影响符号回归效果")
                
            # 运行增强MCTS
            print("\n使用增强MCTS...")
            
            result = run_symbolic_regression(
                X, y,
                true_expression=None,  # 用户数据没有真实表达式
                models=models
            )
            
            # 显示结果
            print(f"\n结果:")
            print(f"找到的解: {result['best_expression']}")
            print(f"R2: {result['r2_score']:.4f}")
            print(f"RMSE: {result['rmse_score']:.4f}")
            print(f"训练时间: {result['training_time']:.2f}s")
            
            # 数据集统计信息
            print(f"\n数据集统计信息")
            print("-" * 30)
            print(f"样本数: {X.shape[0]}")
            print(f"特征数: {X.shape[1]}")
            print(f"目标值范围: [{y.min():.4f}, {y.max():.4f}]")
            print(f"目标值均值: {y.mean():.4f}")
            print(f"目标值标准差: {y.std():.4f}")
            
            # 模型性能评估
            if result['r2_score'] > 0.8:
                print(f"\n模型拟合效果: 优秀 (R2 > 0.8)")
            elif result['r2_score'] > 0.6:
                print(f"\n模型拟合效果: 良好 (R2 > 0.6)")
            elif result['r2_score'] > 0.4:
                print(f"\n模型拟合效果: 一般 (R2 > 0.4)")
            else:
                print(f"\n模型拟合效果: 较差 (R2 ≤ 0.4)")
                
        except Exception as e:
            print(f"错误：{e}")
            print("请确保数据集文件格式正确且包含X和y数组")
    
    # 根据模式运行测试
    if args.mode == 'composite':
        run_composite_function_test()
    elif args.mode == 'dataset-file':
        run_dataset_file_test()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()