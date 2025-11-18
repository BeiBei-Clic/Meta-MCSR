#!/usr/bin/env python3
"""
数据生成脚本

生成阶段零预训练所需的(表达式, 数据集)对。
程序化地生成数百万个物理定律数学表达式，并为每个表达式生成对应的数值观测数据集。
"""

import os
import sys
import numpy as np
import pickle
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/logs/data_generation.log')
        ]
    )


def count_variables_in_expression(expression: str) -> int:
    """计算表达式中使用的变量数量"""
    max_var = 0
    for var in ['x1', 'x2', 'x3']:
        if var in expression:
            max_var = max(max_var, int(var[1]))
    return max_var


def generate_expression_templates() -> List[str]:
    """
    生成丰富的表达式模板集合
    
    Returns:
        表达式模板列表
    """
    expressions = []
    
    # 基础线性表达式
    base_linear = [
        "x1 + x2",
        "x1 - x2", 
        "2*x1 + 3*x2",
        "x1 + 0.5*x2 - 1",
        "3*x1 - x2 + 2",
    ]
    expressions.extend(base_linear)
    
    # 基础非线性表达式
    base_nonlinear = [
        "x1 * x2",
        "x1^2 + x2^2",
        "x1^3 + x2^3",
        "x1 * x2 + x1^2",
        "x1^2 * x2",
        "x1 / (x2 + 1)",
        "x1 / x2 + 1",
        "(x1 + x2)^2",
    ]
    expressions.extend(base_nonlinear)
    
    # 三角函数表达式
    trigonometric = [
        "sin(x1) + cos(x2)",
        "sin(x1 + x2)",
        "cos(x1 * x2)",
        "sin(x1) * cos(x2)",
        "tan(x1) + x2",
        "sin(x1)^2 + cos(x2)^2",
        "sin(x1 + x2) + cos(x1 - x2)",
        "tan(x1 / x2)",
    ]
    expressions.extend(trigonometric)
    
    # 指数和对数表达式
    exponential_log = [
        "exp(x1) + exp(x2)",
        "exp(-x1^2)",
        "log(x1^2 + x2^2)",
        "exp(x1) * sin(x2)",
        "log(x1) + x2",
        "exp(x1 + x2)",
        "x1 * exp(-x2)",
        "log(x1^2 + 1)",
    ]
    expressions.extend(exponential_log)
    
    # 根式表达式
    radical = [
        "sqrt(x1^2 + x2^2)",
        "sqrt(x1) + sqrt(x2)",
        "cbrt(x1) + cbrt(x2)",
        "sqrt(x1 * x2)",
        "sqrt(x1 + x2)",
    ]
    expressions.extend(radical)
    
    # 复合函数表达式
    composite = [
        "sin(x1) * exp(-x2)",
        "log(x1^2 + x2^2) + x1",
        "sqrt(x1 + x2) * sin(x1)",
        "exp(-x1^2) * cos(x2)",
        "log(x1) * sqrt(x2 + 1)",
        "sin(x1 + x2) + exp(-x1)",
        "sqrt(x1) + log(x2 + 1)",
        "cos(x1 * x2) + sin(x1 + x2)",
    ]
    expressions.extend(composite)
    
    # 复杂表达式
    complex_expressions = [
        "sin(x1 + x2) * cos(x1 - x2)",
        "exp(-x1^2) * sin(x2) + cos(x1)",
        "log(x1^2 + x2^2) + sqrt(x1 + x2)",
        "x1^3 * exp(-x2) + sin(x1 + x2)",
        "sqrt(x1^2 + x2^2) * cos(x1 * x2)",
        "sin(x1) * cos(x2) + exp(-x1^2 - x2^2)",
        "log(x1 + 1) * sqrt(x2 + 1) + sin(x1 * x2)",
        "exp(-x1) * sin(x2) + log(x1^2 + x2^2 + 1)",
    ]
    expressions.extend(complex_expressions)
    
    # 物理表达式
    physics_expressions = [
        "9.8 * sin(x1)",  # 重力
        "0.5 * x1^2",     # 动能
        "x1 * cos(x2)",   # 简谐运动
        "exp(-x1) * cos(2 * pi * x2)",  # 阻尼振荡
        "x1^2 + x2^2",    # 距离平方
        "x1 * sin(x2) + x2 * cos(x1)",  # 复杂运动
        "sqrt(x1^2 + x2^2)",  # 距离
        "exp(-x1^2 / (2 * 0.1^2))",  # 高斯函数
    ]
    expressions.extend(physics_expressions)
    
    # 三变量表达式
    three_var_expressions = [
        "x1 + x2 + x3",
        "x1 * x2 + x3^2",
        "sin(x1) + cos(x2) + x3",
        "exp(x1) * sin(x2) + x3",
        "sqrt(x1^2 + x2^2) + x3",
        "x1 * x2 * x3",
        "x1^2 + x2^2 + x3^2",
        "sin(x1 + x2) + cos(x2 + x3)",
    ]
    expressions.extend(three_var_expressions)
    
    # 生成变体表达式
    variants = []
    for base_expr in expressions:
        # 添加常数因子
        variants.extend([
            f"2 * ({base_expr})",
            f"0.5 * ({base_expr})",
            f"1.5 * ({base_expr}) - 1",
        ])
        
        # 添加偏移
        variants.extend([
            f"({base_expr}) + 3",
            f"({base_expr}) - 2",
        ])
        
        # 添加复合运算
        if "x1" in base_expr and "x2" in base_expr:
            variants.extend([
                f"sin({base_expr})",
                f"exp(-abs({base_expr}))",
                f"1 / (1 + exp(-{base_expr}))",  # sigmoid
            ])
    
    expressions.extend(variants)
    
    # 去重并返回
    unique_expressions = list(set(expressions))
    return unique_expressions


def generate_data_from_expression(
    expression: str,
    n_samples: int,
    variables_range: Tuple[float, float],
    noise_level: float,
    n_variables: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据表达式生成数据
    
    Args:
        expression: 数学表达式
        n_samples: 样本数量
        variables_range: 变量范围
        noise_level: 噪声水平
        n_variables: 变量数量
        
    Returns:
        (X, y) 元组
    """
    np.random.seed(np.abs(hash(expression)) % (2**32))
    
    # 生成输入数据，使用更安全的范围避免数学运算错误
    if 'log' in expression or 'ln' in expression:
        # 对于包含对数的表达式，确保输入值为正
        X = np.random.uniform(
            max(0.1, variables_range[0]), variables_range[1],
            (n_samples, n_variables)
        )
    else:
        # 标准范围
        X = np.random.uniform(
            variables_range[0], variables_range[1],
            (n_samples, n_variables)
        )
    
    # 安全计算表达式
    try:
        # 准备变量字典
        var_dict = {}
        for i in range(n_variables):
            var_dict[f'x{i+1}'] = X[:, i]
        
        # 添加安全的数学函数
        def safe_sqrt(x):
            return np.sqrt(np.maximum(0, x))
        
        def safe_log(x):
            return np.log(np.maximum(1e-10, x))
        
        def safe_log10(x):
            return np.log10(np.maximum(1e-10, x))
        
        def safe_exp(x):
            return np.exp(np.clip(x, -500, 500))  # 防止溢出
        
        # 添加允许的函数和常数
        var_dict.update({
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': safe_exp, 'log': safe_log, 'log10': safe_log10,
            'sqrt': safe_sqrt, 'abs': np.abs, 'cbrt': np.cbrt,
            'pi': np.pi, 'e': np.e, 'max': np.maximum, 'min': np.minimum
        })
        
        # 替换^为**
        safe_expr = expression.replace('^', '**')
        
        # 检查表达式是否与变量数量匹配
        for i in range(n_variables + 1, 4):  # 检查是否有未使用的变量
            var_name = f'x{i}'
            if var_name in safe_expr:
                raise ValueError(f"表达式包含未定义的变量 {var_name}，但只指定了 {n_variables} 个变量")
        
        # 计算目标值
        y = eval(safe_expr, {"__builtins__": {}}, var_dict)
        
        # 检查结果是否有效
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("计算结果包含 NaN 或无穷值")
        
        # 添加噪声
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(y))
            y += noise
            
    except Exception as e:
        # 如果表达式计算失败，使用简单的线性组合
        print(f"警告: 表达式计算失败 '{expression}', 使用默认表达式. 错误: {e}")
        y = np.sum(X, axis=1) + np.random.normal(0, noise_level, n_samples)
    
    return X, y


def generate_pretrain_dataset(
    n_expressions: int,
    n_samples_per_expr: int,
    output_path: str,
    variables_range: Tuple[float, float] = (-5, 5),
    noise_level: float = 0.01,
    save_interval: int = 1000
) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    生成预训练数据集
    
    Args:
        n_expressions: 表达式数量
        n_samples_per_expr: 每个表达式的样本数
        output_path: 输出路径
        variables_range: 变量范围
        noise_level: 噪声水平
        save_interval: 保存间隔
        
    Returns:
        (表达式列表, 数据集列表)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始生成 {n_expressions} 个表达式的数据集...")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 生成表达式模板
    expression_templates = generate_expression_templates()
    logger.info(f"生成了 {len(expression_templates)} 个基础表达式模板")
    
    expressions = []
    datasets = []
    
    # 生成表达式
    for i in range(n_expressions):
        # 决定变量数量（1-3个变量）
        n_variables = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
        
        # 选择适合变量数量的基础模板
        available_expressions = [expr for expr in expression_templates 
                               if count_variables_in_expression(expr) <= n_variables]
        
        if not available_expressions:
            # 如果没有合适的表达式，使用最简单的
            base_expr = "x1" if n_variables >= 1 else "1"
        else:
            base_expr = available_expressions[i % len(available_expressions)]
        
        # 随机添加变体
        if np.random.random() < 0.3:
            # 添加系数
            coeff = np.random.choice([0.5, 1, 1.5, 2, 3])
            expression = f"{coeff} * ({base_expr})"
        elif np.random.random() < 0.3:
            # 添加偏移
            offset = np.random.choice([-2, -1, 0, 1, 2])
            expression = f"({base_expr}) + {offset}"
        else:
            expression = base_expr
        
        try:
            # 生成数据
            X, y = generate_data_from_expression(
                expression=expression,
                n_samples=n_samples_per_expr,
                variables_range=variables_range,
                noise_level=noise_level,
                n_variables=n_variables
            )
            
            expressions.append(expression)
            datasets.append((X, y))
            
            # 定期保存
            if (i + 1) % save_interval == 0:
                save_progress(expressions, datasets, output_path, i + 1)
                logger.info(f"已生成 {i + 1}/{n_expressions} 个表达式")
                
        except Exception as e:
            logger.warning(f"生成表达式 {expression} 时出错: {e}")
            continue
    
    # 最终保存
    save_progress(expressions, datasets, output_path, len(expressions))
    
    logger.info(f"数据集生成完成，共 {len(expressions)} 个有效表达式")
    return expressions, datasets


def save_progress(
    expressions: List[str],
    datasets: List[Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    count: int
):
    """保存进度"""
    progress_path = os.path.join(output_path, f'progress_{count}')
    os.makedirs(progress_path, exist_ok=True)
    
    # 保存表达式
    with open(os.path.join(progress_path, 'expressions.pkl'), 'wb') as f:
        pickle.dump(expressions, f)
    
    # 保存数据集
    with open(os.path.join(progress_path, 'datasets.pkl'), 'wb') as f:
        pickle.dump(datasets, f)
    
    # 保存元数据
    metadata = {
        'count': count,
        'generated_at': str(np.datetime64('now')),
        'expression_count': len(expressions),
        'dataset_count': len(datasets)
    }
    
    with open(os.path.join(progress_path, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=2)


def main():
    """主函数"""
    print("=" * 60)
    print("预训练数据生成器")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 配置参数
    n_expressions = 100000  # 生成10万个表达式（可调整）
    n_samples_per_expr = 100  # 每个表达式100个数据点
    output_path = "data/pretrain/"
    variables_range = (-5, 5)
    noise_level = 0.01
    
    print(f"生成配置:")
    print(f"表达式数量: {n_expressions}")
    print(f"每个表达式的样本数: {n_samples_per_expr}")
    print(f"变量范围: {variables_range}")
    print(f"噪声水平: {noise_level}")
    print(f"输出路径: {output_path}")
    
    try:
        # 生成数据集
        expressions, datasets = generate_pretrain_dataset(
            n_expressions=n_expressions,
            n_samples_per_expr=n_samples_per_expr,
            output_path=output_path,
            variables_range=variables_range,
            noise_level=noise_level
        )
        
        # 打印统计信息
        print(f"\n生成完成统计:")
        print(f"总表达式数: {len(expressions)}")
        print(f"总数据集数: {len(datasets)}")
        
        if expressions:
            # 分析表达式
            var_counts = {'x1': 0, 'x2': 0, 'x3': 0}
            func_counts = {}
            
            for expr in expressions:
                for var in ['x1', 'x2', 'x3']:
                    if var in expr:
                        var_counts[var] += 1
                
                # 统计函数
                for func in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                    if func in expr:
                        func_counts[func] = func_counts.get(func, 0) + 1
            
            print(f"\n变量使用统计:")
            for var, count in var_counts.items():
                print(f"  {var}: {count} ({count/len(expressions)*100:.1f}%)")
            
            print(f"\n函数使用统计:")
            for func, count in sorted(func_counts.items()):
                print(f"  {func}: {count} ({count/len(expressions)*100:.1f}%)")
        
        print(f"\n数据已保存到: {output_path}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"数据生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
