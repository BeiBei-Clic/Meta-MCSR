#!/usr/bin/env python3
"""
阶段零：基于对比学习的预训练

此脚本实现第一阶段的预训练，使用对称性InfoNCE损失
对表达式编码器和数据编码器进行联合训练。
"""

import os
import sys
import torch
import numpy as np
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from symbolic_regression.models.expression_encoder import ExpressionEncoder
from symbolic_regression.models.data_encoder import DataEncoder
from symbolic_regression.training.pretrain_pipeline import PretrainPipeline
from symbolic_regression.utils.data_loader import DataLoader


def load_single_file_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """加载单个文件的预训练数据，支持任意维度"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        pretrain_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            print(f"文件内容不足: {file_path}")
            return None
        
        # 解析第一行的表达式
        first_line = lines[0].strip()
        if first_line.startswith('表达式: '):
            current_expression = first_line.replace('表达式: ', '').strip()
            print(f"找到表达式: {current_expression}")
        else:
            print(f"第一行格式错误: {first_line[:50]}...")
            return None
        
        # 解析数据行
        samples = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            try:
                # 解析任意维度数据格式: x1,x2,x3,...,y
                parts = line.split(',')
                if len(parts) >= 2:  # 至少要有x变量和y值
                    # 最后一个值是y，前面的都是x变量
                    x_values = [float(part) for part in parts[:-1]]
                    y_value = float(parts[-1])
                    
                    # 动态生成变量名 x1, x2, x3, ...
                    sample = {'y': y_value}
                    for i, x_val in enumerate(x_values, 1):
                        sample[f'x{i}'] = x_val
                    
                    samples.append(sample)
                else:
                    print(f"数据格式错误: {line}")
            except (ValueError, IndexError) as e:
                print(f"解析数据失败: {line} (错误: {e})")
                continue
        
        if current_expression and samples:
            # 动态获取变量名
            variables = []
            if samples:
                for key in sorted(samples[0].keys()):
                    if key.startswith('x'):
                        variables.append(key)
            
            pretrain_data.append({
                'expression': current_expression,
                'samples': samples,
                'variables': variables
            })
            print(f"加载样本数量: {len(samples)}, 变量维度: {len(variables)}")
        
        return pretrain_data if pretrain_data else None
        
    except Exception as e:
        print(f"加载文件数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def setup_logging():
    """设置日志"""
    # 确保日志目录存在
    os.makedirs('results/logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/logs/pretrain.log')
        ]
    )


def load_pretrain_data(pretrain_data_path: str = "data/pysr_datasets") -> Optional[Dict[str, Any]]:
    """加载PySR格式的预训练数据，如果不存在则自动生成"""
    if not os.path.exists(pretrain_data_path):
        print(f"预训练数据路径 {pretrain_data_path} 不存在，开始自动生成数据...")
        
        # 尝试生成PySR数据
        try:
            pysr_generate_script_path = os.path.join(project_root, 'scripts', 'generate_pretrain_data_PySR.py')
            print(f"执行PySR数据生成器: {pysr_generate_script_path}")
            
            result = subprocess.run([sys.executable, pysr_generate_script_path], 
                                   capture_output=True, text=True, cwd=str(project_root))
            if result.returncode != 0:
                print(f"PySR数据生成失败: {result.stderr}")
                return None
            else:
                print("PySR数据生成完成")
                
        except Exception as e:
            print(f"自动生成PySR数据失败: {e}")
            return None
    
    try:
        # 如果是目录，读取所有txt文件
        if os.path.isdir(pretrain_data_path):
            pretrain_data = []
            txt_files = sorted([f for f in os.listdir(pretrain_data_path) if f.endswith('.txt')])
            
            print(f"在目录 {pretrain_data_path} 中找到 {len(txt_files)} 个数据文件")
            
            if len(txt_files) == 0:
                print("没有找到数据文件")
                return None
            
            # 处理所有文件
            count = 0
            
            for file_name in txt_files:
                file_path = os.path.join(pretrain_data_path, file_name)
                # print(f"处理文件: {file_name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if len(lines) < 2:
                    print(f"  文件 {file_name} 内容不足，跳过")
                    continue
                
                # 解析第一行的表达式
                first_line = lines[0].strip()
                if first_line.startswith('表达式: '):
                    current_expression = first_line.replace('表达式: ', '').strip()
                    # print(f"  找到表达式: {current_expression}")
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
                        # 解析任意维度数据格式: x1,x2,x3,...,y
                        parts = line.split(',')
                        if len(parts) >= 2:  # 至少要有x变量和y值
                            # 最后一个值是y，前面的都是x变量
                            x_values = [float(part) for part in parts[:-1]]
                            y_value = float(parts[-1])
                            
                            # 动态生成变量名 x1, x2, x3, ...
                            sample = {'y': y_value}
                            for i, x_val in enumerate(x_values, 1):
                                sample[f'x{i}'] = x_val
                            
                            samples.append(sample)
                        else:
                            print(f"    数据格式错误: {line}")
                    except (ValueError, IndexError) as e:
                        print(f"    解析数据失败: {line} (错误: {e})")
                        continue
                
                if current_expression and samples:
                    # 动态获取变量名（从样本中推断维度）
                    variables = []
                    if samples:
                        # 从第一个样本中获取所有x变量名
                        for key in sorted(samples[0].keys()):
                            if key.startswith('x'):
                                variables.append(key)
                    
                    pretrain_data.append({
                        'expression': current_expression,
                        'samples': samples,
                        'variables': variables  # 动态变量列表
                    })
                    # print(f"  加载样本数量: {len(samples)}, 变量维度: {len(variables)}")
                    count += 1
                else:
                    print(f"  文件 {file_name} 无有效数据")
            
            print(f"成功加载 {len(pretrain_data)} 个表达式数据")
            
            if len(pretrain_data) == 0:
                print("没有加载到任何有效数据")
                return None
            
            return pretrain_data
        
        # 如果是单个文件，保持原有逻辑（兼容性）
        else:
            return load_single_file_data(pretrain_data_path)
            
    except Exception as e:
        print(f"加载预训练数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_pysr_data_to_pretrain_format(pysr_data: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """将PySR格式数据转换为预训练格式，支持任意维度"""
    expressions = []
    datasets = []
    
    for item in pysr_data:
        expression = item['expression']
        samples = item['samples']
        
        # 提取X和y数据，支持任意维度
        if len(samples) > 0:
            # 动态获取所有x变量名（按数字顺序排列）
            x_variables = []
            if 'variables' in item and item['variables']:
                # 如果有明确的变量列表，使用它
                x_variables = sorted([var for var in item['variables'] if var.startswith('x')])
            else:
                # 否则从样本中推断
                x_variables = sorted([key for key in samples[0].keys() if key.startswith('x')], 
                                   key=lambda x: int(x[1:]))  # 按数字部分排序
            
            # 构建X矩阵
            if x_variables:
                X = np.array([[sample[var] for var in x_variables] for sample in samples]).astype(np.float32)
            else:
                # 兼容性：如果没有x变量，创建一个虚拟维度
                X = np.array([[0.0] for sample in samples]).astype(np.float32)
            
            y = np.array([sample['y'] for sample in samples]).astype(np.float32)
            
            expressions.append(expression)
            datasets.append((X, y))
            
            # print(f"  转换表达式: {expression[:50]}..., 维度: {X.shape[1]}")
    
    return expressions, datasets


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
            },
            'contrastive': {
                'temperature': 0.07
            }
        },
        'training': {
            'pretrain': {
                'n_expressions': 10000,  # 减少数量以便快速测试
                'n_samples_per_expr': 100,
                'variables_range': [-5, 5],
                'noise_level': 0.01,
                'batch_size': 32,
                'learning_rate': 0.0001,  # 确保是数字类型
                'num_epochs': 10,  # 减少epochs以便快速测试
                'weight_decay': 0.0001,  # 确保是数字类型
                'warmup_steps': 1000,
                'output_dir': 'models_weights/pretrained/'
            }
        },
        'data': {
            'pretrain': {
                'output_path': 'data/pretrain/'
            }
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_seed': 42
    }


def main():
    """主函数"""
    print("=" * 60)
    print("基于对比学习的预训练 - 阶段零")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 加载配置
    try:
        config = load_config()
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
        
        # 创建预训练管道
        logger.info("创建预训练管道...")
        pretrain_pipeline = PretrainPipeline(
            expression_encoder=expression_encoder,
            data_encoder=data_encoder,
            config=config['training']['pretrain'],
            device=device
        )
        
        # 加载PySR格式的预训练数据（如果不存在会自动生成）
        pysr_data_path = "data/pysr_datasets"
        pysr_data = load_pretrain_data(pysr_data_path)
        
        if not pysr_data:
            logger.error("无法加载或生成预训练数据")
            raise FileNotFoundError("预训练数据加载失败")
            
        logger.info(f"使用PySR格式数据: {pysr_data_path}，共 {len(pysr_data)} 个表达式")
        # 转换数据格式
        expressions, datasets = convert_pysr_data_to_pretrain_format(pysr_data)
        logger.info(f"转换后数据: {len(expressions)} 个表达式")
        # 开始预训练
        training_history = pretrain_pipeline.fit(expressions=expressions, datasets=datasets)
            
        # 保存最终结果
        pretrain_pipeline.save_pretrained()

        # 打印结果
        print("\n" + "=" * 60)
        print("预训练完成！")
        print("=" * 60)
        print(f"最终训练损失: {training_history['train_loss']['loss'][-1]:.4f}")
        print(f"最终验证损失: {training_history['val_loss']['loss'][-1]:.4f}")
        print(f"模型保存路径: {config['training']['pretrain']['output_dir']}")
        print(f"日志保存路径: results/logs/pretrain.log")
        
        return 0
        
    except Exception as e:
        logger.error(f"预训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
