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
import datetime

def print_help():
    """打印帮助信息"""
    print("""
Meta-MCSR: 基于自学习奖励网络的符号回归系统

用法:
  python mcts_with_reward_network.py <模式> [选项]

模式:
  composite                 运行复合函数测试
  dataset-file             指定数据集文件测试

选项:
  --dataset-path PATH      dataset-file模式下的数据集文件路径
  --num-samples N          使用的数据集实例数量 (默认: 使用全部数据)
  --expr-encoder PATH      表达式编码器模型路径 (默认: weights/expression_encoder)
  --reward-network PATH    奖励网络模型路径 (默认: weights/reward_network_final)
  --max-iterations N       MCTS最大迭代次数 (默认: 500)
  --max-depth N            MCTS最大深度 (默认: 8)
  --train-test-split R     训练/测试集分割比例 (默认: 0.8)
  --use-reward-network VALUE  是否使用奖励网络 (true=使用, false=不使用, 默认: true)
  --no-use-reward-network    使用传统MCTS (不使用奖励网络)
  --help, -h               显示此帮助信息

示例:
  python mcts_with_reward_network.py composite
  python mcts_with_reward_network.py dataset-file --dataset-path dataset/Feynman_with_units/I.6.2
  python mcts_with_reward_network.py dataset-file --dataset-path dataset/Feynman_with_units/I.6.2 --num-samples 1000
  python mcts_with_reward_network.py composite --no-use-reward-network
  python mcts_with_reward_network.py dataset-file --dataset-path dataset/Feynman_with_units/I.6.2 --use-reward-network false

数据集文件格式:
  - 纯文本格式 (如费曼数据集):
    - 每行空格分隔的浮点数
    - 最后一列为标签y，前面为输入特征
    - 自动检测并处理，无需额外转换
""")

def save_experiment_results(args, problem_name, dataset_name, result, use_reward_network=True, extra_info=None):
    """保存实验结果到txt文件"""
    # 创建result文件夹（如果不存在）
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 生成文件名：基于时间和模式
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_base = os.path.splitext(os.path.basename(args.dataset_path))[0]
    filename = f"{dataset_base}_{timestamp}.txt"
    
    filepath = os.path.join(result_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 基本实验信息
            f.write(f"实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实验模式: {args.mode}\n")
            if args.mode == 'dataset-file':
                f.write(f"数据集路径: {args.dataset_path}\n")
            f.write(f"使用奖励网络: {'是' if use_reward_network else '否'}\n")
            f.write(f"数据集名称: {dataset_name}\n")
            f.write(f"问题名称: {problem_name}\n\n")
            
            # 超参数设置
            f.write("实验参数设置:\n")
            f.write("-" * 30 + "\n")
            f.write(f"最大迭代次数: {args.max_iterations}\n")
            f.write(f"最大深度: {args.max_depth}\n")
            f.write(f"训练/测试分割比例: {args.train_test_split}\n")
            f.write(f"样本数量: {args.num_samples if args.num_samples else '使用全部数据'}\n")
            f.write(f"使用奖励网络: {'是' if use_reward_network else '否'}\n")
            f.write("\n")
            
            # 实验结果
            f.write("实验结果:\n")
            f.write("-" * 30 + "\n")
            f.write(f"找到的解: {result['best_expression']}\n\n")
            
            f.write("性能指标:\n")
            f.write(f"训练集 R² 分数: {result['train_r2_score']:.6f}\n")
            f.write(f"训练集 RMSE: {result['train_rmse_score']:.6f}\n")
            f.write(f"测试集 R² 分数: {result['test_r2_score']:.6f}\n")
            f.write(f"测试集 RMSE: {result['test_rmse_score']:.6f}\n")
            f.write(f"训练时间: {result['training_time']:.2f} 秒\n\n")
            
        print(f"\n实验结果已保存到: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"\n警告：保存实验结果失败 - {e}")
        return None

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
        # 加载为纯文本格式（如费曼数据集）
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            # 单列数据，假设只有标签
            y = data
            X = np.arange(len(y)).reshape(-1, 1)  # 创建简单的索引特征
        else:
            # 多列数据，最后一列为标签，前面为特征
            X = data[:, :-1]  # 前面所有列作为特征
            y = data[:, -1]   # 最后一列作为标签
        
        return {
            'X': X,
            'y': y,
            'name': os.path.basename(file_path),
            'description': f"从文本文件加载的数据集: {file_path} (特征数: {X.shape[1]}, 样本数: {X.shape[0]})",
            'true_expression': None  # 用户数据没有真实表达式
        }
            
    except Exception as e:
        raise ValueError(f"加载数据集失败: {e}")


def main():
    """主函数"""
    # 检查命令行参数
    use_reward_network = True  # 默认使用奖励网络
    
    # 检查是否有--use-reward-network参数
    if '--use-reward-network' in sys.argv:
        # 找到该参数的索引并处理
        try:
            idx = sys.argv.index('--use-reward-network')
            if idx + 1 < len(sys.argv):
                value = sys.argv[idx + 1].lower()
                use_reward_network = value in ['true', '1', 'yes', 'on']
        except (ValueError, IndexError):
            use_reward_network = True
    elif '--no-use-reward-network' in sys.argv:
        use_reward_network = False
    
    # 创建args对象
    class Args:
        def __init__(self):
            self.mode = None
            self.dataset_path = None
            self.num_samples = None
            self.expr_encoder = 'weights/expression_encoder'
            self.reward_network = 'weights/reward_network_final'
            self.max_iterations = 500
            self.max_depth = 8
            self.train_test_split = 0.8
            self.use_reward_network = use_reward_network
    
    args = Args()
    
    # 解析位置参数
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        if sys.argv[1] in ['composite', 'dataset-file']:
            args.mode = sys.argv[1]
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print_help()
            return
    
    # 解析其他参数
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--dataset-path' and i + 1 < len(sys.argv):
            args.dataset_path = sys.argv[i + 1]
            i += 2
        elif arg == '--num-samples' and i + 1 < len(sys.argv):
            args.num_samples = int(sys.argv[i + 1])
            i += 2
        elif arg == '--expr-encoder' and i + 1 < len(sys.argv):
            args.expr_encoder = sys.argv[i + 1]
            i += 2
        elif arg == '--reward-network' and i + 1 < len(sys.argv):
            args.reward_network = sys.argv[i + 1]
            i += 2
        elif arg == '--max-iterations' and i + 1 < len(sys.argv):
            args.max_iterations = int(sys.argv[i + 1])
            i += 2
        elif arg == '--max-depth' and i + 1 < len(sys.argv):
            args.max_depth = int(sys.argv[i + 1])
            i += 2
        elif arg == '--train-test-split' and i + 1 < len(sys.argv):
            args.train_test_split = float(sys.argv[i + 1])
            i += 2
        elif arg == '--help' or arg == '-h':
            print_help()
            return
        else:
            i += 1
    
    # 如果没有提供参数，显示帮助
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == '--help'):
        print_help()
        return
    
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
        # 划分训练/测试集
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            print("错误：缺少sklearn库")
            print("请安装: pip install scikit-learn")
            return None
        
        train_ratio = args.train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-train_ratio, random_state=42
        )
        
        print(f"\n数据集分割:")
        print(f"  训练集: {X_train.shape[0]} 样本 ({train_ratio*100:.0f}%)")
        print(f"  测试集: {X_test.shape[0]} 样本 ({(1-train_ratio)*100:.0f}%)")
        
        # 设置MCTS参数
        mcts_params = {
            'max_depth': args.max_depth,
            'max_iterations': args.max_iterations,
            'max_vars': min(X.shape[1], 3),
            'eta': 0.999,
            'alpha_hybrid': 0.7
        }
        
        start_time = time.time()
        
        # 创建奖励网络（如果使用奖励网络模式）
        if args.use_reward_network:
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
                use_reward_network=True,
                **mcts_params
            )
        else:
            # 创建传统MCTS（不使用奖励网络）
            mcts = MCTSWithRewardNetwork(
                reward_network=None,
                use_reward_network=False,
                **mcts_params
            )
        
        # 设置神谕目标
        if true_expression:
            try:
                mcts.set_oracle_target(str(true_expression))
            except:
                print("警告：无法设置神谕目标")
        
        # 在训练集上训练模型
        if verbose:
            print("开始训练...")
        
        best_expr = mcts.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # 在训练集上评估
        train_r2, train_rmse = mcts.get_score(X_train, y_train)
        
        # 在测试集上评估
        test_r2, test_rmse = mcts.get_score(X_test, y_test)
        
        results = {
            'best_expression': str(best_expr),
            'train_r2_score': train_r2,
            'train_rmse_score': train_rmse,
            'test_r2_score': test_r2,
            'test_rmse_score': test_rmse,
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
        
        # 运行MCTS
        if args.use_reward_network:
            print("\n使用增强MCTS（奖励网络）")
            print("-" * 30)
        else:
            print("\n使用传统MCTS")
            print("-" * 30)
        
        result = run_symbolic_regression(
            X, y,
            true_expression=problem['true_expression'],
            models=models
        )
        
        # 显示结果
        print(f"\n结果:")
        print(f"找到的解: {result['best_expression']}")
        print(f"训练集 R2: {result['train_r2_score']:.4f}")
        print(f"训练集 RMSE: {result['train_rmse_score']:.4f}")
        print(f"测试集 R2: {result['test_r2_score']:.4f}")
        print(f"测试集 RMSE: {result['test_rmse_score']:.4f}")
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
        
        # 保存实验结果
        extra_info = f"真实解: {problem['true_expression']}\n真实解R2: {r2_true:.4f}\n真实解RMSE: {rmse_true:.4f}"
        save_experiment_results(
            args=args,
            problem_name=problem['name'],
            dataset_name="复合函数测试数据",
            result=result,
            use_reward_network=args.use_reward_network,
            extra_info=extra_info
        )
    
    def run_dataset_file_test():
        """运行指定数据集文件测试"""
        if not args.dataset_path:
            print("错误：使用dataset-file模式必须指定 --dataset-path 参数")
            print("示例: python mcts_with_reward_network.py dataset-file --dataset-path dataset/Feynman_with_units/I.6.2")
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
            
            # 如果指定了样本数量限制，随机选择样本
            if args.num_samples and args.num_samples > 0 and args.num_samples < len(X):
                import numpy as np
                indices = np.random.choice(len(X), args.num_samples, replace=False)
                X = X[indices]
                y = y[indices]
                print(f"\n数据集: {dataset['name']}")
                print(f"描述: {dataset['description']}")
                print(f"数据形状: X={X.shape}, y={y.shape} (从 {len(dataset['X'])} 条数据中随机选取 {args.num_samples} 条)")
            else:
                print(f"\n数据集: {dataset['name']}")
                print(f"描述: {dataset['description']}")
                print(f"数据形状: X={X.shape}, y={y.shape}")
            
            # 检查数据维度是否合适
            if X.shape[1] > 5:
                print(f"警告：特征维度较高 ({X.shape[1]})，可能影响符号回归效果")
                
            # 运行MCTS
            if args.use_reward_network:
                print("\n使用增强MCTS（奖励网络增强）...")
            else:
                print("\n使用传统MCTS...")
            
            result = run_symbolic_regression(
                X, y,
                true_expression=None,  # 用户数据没有真实表达式
                models=models
            )
            
            # 显示结果
            print(f"\n结果:")
            print(f"找到的解: {result['best_expression']}")
            print(f"训练集 R2: {result['train_r2_score']:.4f}")
            print(f"训练集 RMSE: {result['train_rmse_score']:.4f}")
            print(f"测试集 R2: {result['test_r2_score']:.4f}")
            print(f"测试集 RMSE: {result['test_rmse_score']:.4f}")
            print(f"训练时间: {result['training_time']:.2f}s")
            
            # 检查过拟合
            overfitting_gap = result['train_r2_score'] - result['test_r2_score']
            if overfitting_gap > 0.1:
                print(f"\n注意：可能存在过拟合 (训练-测试R2差值: {overfitting_gap:.4f})")
            elif overfitting_gap < -0.05:
                print(f"\n注意：可能存在欠拟合 (训练-测试R2差值: {overfitting_gap:.4f})")
            else:
                print(f"\n模型泛化性能: 良好 (训练-测试R2差值: {overfitting_gap:.4f})")
            
            # 保存实验结果
            save_experiment_results(
                args=args,
                problem_name="自定义数据集符号回归",
                dataset_name=dataset['name'],
                result=result,
                use_reward_network=args.use_reward_network,
                extra_info=f"数据集描述: {dataset['description']}\n原始数据形状: {dataset['X'].shape}"
            )
                
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