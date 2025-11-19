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
import subprocess
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


def load_pretrain_data(pretrain_data_path: str = "data/pysr_datasets") -> Optional[Dict[str, Any]]:
    """加载预训练数据"""
    if not os.path.exists(pretrain_data_path):
        print(f"警告：预训练数据路径 {pretrain_data_path} 不存在")
        print("尝试调用数据生成器...")
        return None
    
    try:
        # 如果是目录，读取所有txt文件
        if os.path.isdir(pretrain_data_path):
            pretrain_data = []
            txt_files = sorted([f for f in os.listdir(pretrain_data_path) if f.endswith('.txt')])
            
            print(f"在目录 {pretrain_data_path} 中找到 {len(txt_files)} 个数据文件")
            
            if len(txt_files) == 0:
                print("没有找到数据文件，调用数据生成器...")
                return None
            
            # 处理所有文件
            count = 0
            total_samples = 0
            
            for file_name in txt_files:
                file_path = os.path.join(pretrain_data_path, file_name)
                print(f"处理文件: {file_name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if len(lines) < 2:
                    print(f"  文件 {file_name} 内容不足，跳过")
                    continue
                
                # 解析第一行的表达式
                first_line = lines[0].strip()
                if first_line.startswith('表达式: '):
                    current_expression = first_line.replace('表达式: ', '').strip()
                    print(f"  找到表达式: {current_expression}")
                else:
                    print(f"  文件 {file_name} 第一行格式错误: {first_line[:50]}...")
                    continue
                
                # 解析数据行
                samples = []
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # 解析 x1,x2,y 格式
                        parts = line.split(',')
                        if len(parts) >= 3:
                            x1 = float(parts[0])
                            x2 = float(parts[1])
                            y = float(parts[2])
                            
                            samples.append({'x1': x1, 'x2': x2, 'y': y})
                        else:
                            print(f"    数据格式错误: {line}")
                    except (ValueError, IndexError) as e:
                        print(f"    解析数据失败: {line} (错误: {e})")
                        continue
                
                if current_expression and samples:
                    pretrain_data.append({
                        'expression': current_expression,
                        'samples': samples,
                        'variables': ['x1', 'x2']  # 双变量表达式
                    })
                    print(f"  加载样本数量: {len(samples)}")
                    count += 1
                    total_samples += len(samples)
                else:
                    print(f"  文件 {file_name} 无有效数据")
            
            print(f"成功加载 {len(pretrain_data)} 个表达式数据，总样本数: {total_samples}")
            
            if len(pretrain_data) == 0:
                print("没有加载到任何有效数据，调用数据生成器...")
                return None
            
            return pretrain_data
        
        # 如果是文件，保持原有逻辑（兼容性）
        else:
            return load_single_file_data(pretrain_data_path)
            
    except Exception as e:
        print(f"加载预训练数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_single_file_data(file_path: str) -> Optional[Dict[str, Any]]:
    """加载单个文件的预训练数据（原格式兼容性）"""
    # 这里保持原有逻辑的简化版本
    pretrain_data = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        current_expression = None
        samples = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Expression: '):
                if current_expression and samples:
                    pretrain_data.append({
                        'expression': current_expression,
                        'samples': samples
                    })
                current_expression = line.replace('Expression: ', '').strip()
                samples = []
            elif line.startswith('Sample ') and 'X=' in line:
                try:
                    import re
                    match = re.search(r'X=\[(.*?)\], y=(.*)', line)
                    if match:
                        x_value = float(match.group(1))
                        y_value = float(match.group(2))
                        samples.append({'X': x_value, 'y': y_value})
                except (ValueError, IndexError):
                    continue
        
        # 添加最后一个表达式
        if current_expression and samples:
            pretrain_data.append({
                'expression': current_expression,
                'samples': samples
            })
        
        return pretrain_data
    except Exception as e:
        print(f"加载单个文件数据失败: {e}")
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
    
    logger.info("开始在线微调流程...")
    logger.info("1. 加载配置和模型")
    logger.info("2. 自动加载预训练数据（若不存在则自动生成）")
    logger.info("3. 执行MCTS探索与微调")
    logger.info("4. 保存最终结果")
    
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
        pretrain_data_path = "data/pysr_datasets"
        
        pretrain_data = load_pretrain_data(pretrain_data_path)
        if pretrain_data:
            logger.info(f"从 {pretrain_data_path} 加载了预训练数据")
        else:
            logger.info("预训练数据不存在，自动调用数据生成器...")
            try:
                # 调用PySR数据生成器
                generate_script_path = os.path.join(project_root, 'scripts', 'generate_pretrain_data_PySR.py')
                if os.path.exists(generate_script_path):
                    logger.info(f"执行数据生成脚本: {generate_script_path}")
                    import subprocess
                    result = subprocess.run([sys.executable, generate_script_path], 
                                          capture_output=True, text=True, cwd=str(project_root))
                    if result.returncode == 0:
                        logger.info("数据生成完成，重新加载数据...")
                        pretrain_data = load_pretrain_data(pretrain_data_path)
                        if pretrain_data:
                            logger.info(f"重新加载预训练数据成功，表达式数量: {len(pretrain_data)}")
                        else:
                            logger.error("数据生成后仍然无法加载数据")
                    else:
                        logger.error(f"数据生成失败: {result.stderr}")
                else:
                    logger.error(f"数据生成脚本不存在: {generate_script_path}")
            except Exception as e:
                logger.error(f"调用数据生成器时出错: {e}")
                import traceback
                traceback.print_exc()
        
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
            # 选择第一个表达式进行测试
            selected_sample = pretrain_data[0]
            
            true_expression = selected_sample['expression']
            samples = selected_sample['samples']
            variables = selected_sample.get('variables', ['x1'])  # 获取变量列表
            logger.info(f"使用表达式进行测试: {true_expression}")
            
            # 根据变量数量构建X数组
            if variables == ['x1', 'x2'] or len(variables) == 2:
                # 双变量表达式
                X = np.array([[sample['x1'], sample['x2']] for sample in samples]).astype(np.float32)
            else:
                # 单变量表达式（兼容原格式）
                if 'X' in samples[0]:
                    X = np.array([[sample['X']] for sample in samples]).astype(np.float32)
                else:
                    # 如果是新格式但只有x1变量
                    X = np.array([[sample.get('x1', 0)] for sample in samples]).astype(np.float32)
            
            y = np.array([sample['y'] for sample in samples]).astype(np.float32)

            problem_data = {
                'true_expression': true_expression,
                'X': X,
                'y': y,
                'variables': variables
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
        
        # 根据X的形状构建变量字典
        allowed_names = {
            'sin': np.sin, 'cos': np.cos,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs
        }
        
        # 添加变量到允许的名称中
        if X.shape[1] >= 1:
            allowed_names['x1'] = X[:, 0]
        if X.shape[1] >= 2:
            allowed_names['x2'] = X[:, 1]
        if X.shape[1] >= 3:
            allowed_names['x3'] = X[:, 2]
        if X.shape[1] >= 4:
            allowed_names['x4'] = X[:, 3]
        
        y_pred = eval(expr_str, {"__builtins__": {}}, allowed_names)
        
        return R2_score(y, y_pred)
        
    except Exception as e:
        print(f"R2计算失败: {e}")
        return -np.inf


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
