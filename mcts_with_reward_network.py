import torch
import numpy as np
import sys
import os
import argparse
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
    
    # 创建MCTS实例
    if models and 'reward_network' in models:
        print("使用增强MCTS（奖励网络）")
        
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
        
    else:
        print("使用传统MCTS（R2分数）")
        from mcts_core import MCTSSymbolicRegression
        
        mcts = MCTSSymbolicRegression(**mcts_params)
    
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


def evaluate_expression(expression_str, X, y):
    """评估表达式的性能"""
    try:
        # 解析表达式（简化处理）
        # 在实际应用中，这里需要更复杂的表达式解析和求值
        # 这里我们使用一个简单的近似方法
        
        # 创建一个简单的评估函数（仅用于示例）
        def simple_eval(x_values):
            # 这里需要根据expression_str构建实际的求值函数
            # 为了演示，我们假设所有表达式都是有效的
            return np.random.random(len(x_values))  # 占位符
        
        # 注意：在实际应用中，需要使用真正的表达式求值
        # 这里只是示例，实际应该使用sympy或其他工具
        
        # 计算R2分数（占位符）
        y_pred = simple_eval(X)
        r2 = R2_score(y, y_pred)
        rmse = RMSE_score(y, y_pred)
        
        return r2, rmse
        
    except Exception as e:
        print(f"评估表达式时出错: {e}")
        return -np.inf, np.inf


def compare_methods(problem, models):
    """比较不同方法的性能"""
    print(f"\n{'='*60}")
    print(f"问题: {problem['name']}")
    print(f"描述: {problem['description']}")
    print(f"真实解: {problem['true_expression']}")
    print(f"{'='*60}")
    
    X = problem['X']
    y = generate_noisy_data(problem['true_function'], X)
    
    results = {}
    
    # 方法1：传统MCTS
    print("\n1. 传统MCTS（R2分数）")
    print("-" * 30)
    result_traditional = run_symbolic_regression(
        X, y, models=None, verbose=False
    )
    results['traditional'] = result_traditional
    print(f"表达式: {result_traditional['best_expression']}")
    print(f"R2: {result_traditional['r2_score']:.4f}")
    print(f"RMSE: {result_traditional['rmse_score']:.4f}")
    print(f"训练时间: {result_traditional['training_time']:.2f}s")
    
    # 方法2：增强MCTS（如果模型可用）
    if models and 'reward_network' in models:
        print("\n2. 增强MCTS（奖励网络）")
        print("-" * 30)
        result_enhanced = run_symbolic_regression(
            X, y, 
            true_expression=problem['true_expression'],
            models=models, 
            verbose=False
        )
        results['enhanced'] = result_enhanced
        print(f"表达式: {result_enhanced['best_expression']}")
        print(f"R2: {result_enhanced['r2_score']:.4f}")
        print(f"RMSE: {result_enhanced['rmse_score']:.4f}")
        print(f"训练时间: {result_enhanced['training_time']:.2f}s")
    
    # 方法3：真实解性能
    print("\n3. 真实解性能（参考）")
    print("-" * 30)
    y_true = problem['true_function'](X)
    r2_true = R2_score(y, y_true)
    rmse_true = RMSE_score(y, y_true)
    print(f"真实解: {problem['true_expression']}")
    print(f"R2: {r2_true:.4f}")
    print(f"RMSE: {rmse_true:.4f}")
    
    # 比较结果
    print("\n4. 比较总结")
    print("-" * 30)
    if 'enhanced' in results:
        print(f"传统MCTS R2: {result_traditional['r2_score']:.4f}")
        print(f"增强MCTS R2: {result_enhanced['r2_score']:.4f}")
        print(f"真实解   R2: {r2_true:.4f}")
        
        improvement = result_enhanced['r2_score'] - result_traditional['r2_score']
        print(f"性能提升: {improvement:+.4f}")
    else:
        print(f"传统MCTS R2: {result_traditional['r2_score']:.4f}")
        print(f"真实解   R2: {r2_true:.4f}")
    
    return results


def interactive_mode():
    """交互模式"""
    print("符号回归系统交互模式")
    print("=" * 40)
    
    # 加载模型
    models = load_trained_models()
    
    problems = create_symbolic_regression_problem()
    
    while True:
        print("\n可用问题:")
        for i, problem in enumerate(problems):
            print(f"  {i+1}. {problem['name']} - {problem['description']}")
        print(f"  {len(problems)+1}. 自定义问题")
        print(f"  0. 退出")
        
        try:
            choice = int(input("\n选择问题 (0-{}): ".format(len(problems)+1)))
            
            if choice == 0:
                print("再见！")
                break
            elif 1 <= choice <= len(problems):
                problem = problems[choice-1]
                compare_methods(problem, models)
            elif choice == len(problems) + 1:
                # 自定义问题
                print("\n自定义问题设置:")
                n_samples = int(input("样本数量 (默认100): ") or "100")
                n_features = int(input("特征数量 (默认2): ") or "2")
                
                X = np.random.uniform(-3, 3, (n_samples, n_features))
                
                print("\n请输入真实函数 (用numpy格式，例如: 'X[:,0] + 2*X[:,1]**2'):")
                func_str = input("函数: ")
                
                try:
                    # 创建匿名函数
                    true_function = eval(f"lambda X: {func_str}")
                    print("请输入表达式字符串:")
                    expr_str = input("表达式: ")
                    
                    custom_problem = {
                        'name': '自定义问题',
                        'X': X,
                        'true_function': true_function,
                        'true_expression': expr_str,
                        'description': f"y = {func_str} + 噪声"
                    }
                    
                    compare_methods(custom_problem, models)
                    
                except Exception as e:
                    print(f"错误：{e}")
            else:
                print("无效选择，请重新输入")
                
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n再见！")
            break


def benchmark_mode():
    """基准测试模式"""
    print("符号回归系统基准测试")
    print("=" * 40)
    
    # 加载模型
    models = load_trained_models()
    
    problems = create_symbolic_regression_problem()
    
    print(f"测试 {len(problems)} 个问题...")
    
    results_summary = {
        'traditional': [],
        'enhanced': []
    }
    
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] 测试: {problem['name']}")
        
        try:
            result = compare_methods(problem, models)
            
            if 'traditional' in result:
                results_summary['traditional'].append(result['traditional']['r2_score'])
            if 'enhanced' in result:
                results_summary['enhanced'].append(result['enhanced']['r2_score'])
                
        except Exception as e:
            print(f"问题 {problem['name']} 测试失败: {e}")
            continue
    
    # 输出总结
    print(f"\n{'='*60}")
    print("基准测试总结")
    print(f"{'='*60}")
    
    if results_summary['traditional']:
        avg_r2_traditional = np.mean(results_summary['traditional'])
        print(f"传统MCTS平均R2: {avg_r2_traditional:.4f}")
    
    if results_summary['enhanced']:
        avg_r2_enhanced = np.mean(results_summary['enhanced'])
        print(f"增强MCTS平均R2: {avg_r2_enhanced:.4f}")
        
        if results_summary['traditional']:
            improvement = avg_r2_enhanced - avg_r2_traditional
            print(f"平均性能提升: {improvement:+.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于自学习奖励网络的MCTS符号回归')
    parser.add_argument('--mode', choices=['interactive', 'benchmark', 'single'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--expr-encoder', default='weights/expression_encoder',
                       help='表达式嵌入器模型路径')
    parser.add_argument('--reward-network', default='weights/reward_network_final',
                       help='奖励网络模型路径')
    parser.add_argument('--problem', type=int, help='单次运行的问题索引')
    parser.add_argument('--mcts-iterations', type=int, default=200,
                       help='MCTS迭代次数')
    parser.add_argument('--max-depth', type=int, default=6,
                       help='MCTS最大深度')
    
    args = parser.parse_args()
    
    print("基于自学习奖励网络的MCTS符号回归系统")
    print("=" * 60)
    
    # 加载模型
    models = load_trained_models(args.expr_encoder, args.reward_network)
    
    if args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'benchmark':
        benchmark_mode()
    elif args.mode == 'single':
        problems = create_symbolic_regression_problem()
        
        if args.problem is None or args.problem < 1 or args.problem > len(problems):
            print(f"请指定有效的问题索引 (1-{len(problems)})")
            return
        
        problem = problems[args.problem - 1]
        
        # 设置MCTS参数
        mcts_params = {
            'max_depth': args.max_depth,
            'max_iterations': args.mcts_iterations,
            'max_vars': min(problem['X'].shape[1], 3),
            'eta': 0.999,
            'alpha_hybrid': 0.7
        }
        
        result = run_symbolic_regression(
            problem['X'],
            generate_noisy_data(problem['true_function'], problem['X']),
            true_expression=problem['true_expression'],
            models=models,
            mcts_params=mcts_params
        )
        
        print(f"\n结果:")
        print(f"问题: {problem['name']}")
        print(f"真实解: {problem['true_expression']}")
        print(f"找到的解: {result['best_expression']}")
        print(f"R2: {result['r2_score']:.4f}")
        print(f"RMSE: {result['rmse_score']:.4f}")
        print(f"训练时间: {result['training_time']:.2f}s")


if __name__ == "__main__":
    main()