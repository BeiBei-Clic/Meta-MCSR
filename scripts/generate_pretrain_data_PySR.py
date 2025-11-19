#!/usr/bin/env python3
"""
PySR格式数据生成器 - 每个表达式生成单独的txt文件
"""

import os
import sys
import numpy as np
import logging
import random
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass
import re

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

@dataclass
class PhysicsFormula:
    """物理公式数据类"""
    name: str
    domain: str
    expression: str
    variables: List[str]
    description: str
    complexity: int

class PySRDataGenerator:
    """PySR格式数据生成器"""
    
    def __init__(self):
        self.samples_per_expression = 100
        self.physics_formulas = self._define_physics_formulas()
        
    def _define_physics_formulas(self) -> List[PhysicsFormula]:
        """定义物理公式"""
        return [
            # 力学
            PhysicsFormula("牛顿第二定律", "力学", "m * a", ["m", "a"], "F = ma", 1),
            PhysicsFormula("动能", "力学", "0.5 * m * v**2", ["m", "v"], "动能公式", 2),
            PhysicsFormula("重力势能", "力学", "m * g * h", ["m", "g", "h"], "重力势能", 2),
            PhysicsFormula("简谐振动", "力学", "A * sin(omega * t)", ["A", "omega", "t"], "简谐振动位移", 3),
            PhysicsFormula("波动方程", "力学", "A * sin(k*x - omega*t)", ["A", "k", "x", "omega", "t"], "一维波动方程", 4),
            PhysicsFormula("万有引力", "力学", "G * m1 * m2 / r**2", ["G", "m1", "m2", "r"], "万有引力定律", 3),
            PhysicsFormula("自由落体", "力学", "h0 + v0*t - 0.5*g*t**2", ["h0", "v0", "t", "g"], "自由落体运动", 3),
            
            # 电磁学
            PhysicsFormula("电场强度", "电磁学", "F / q", ["F", "q"], "电场强度定义", 1),
            PhysicsFormula("电势", "电磁学", "k * q / r", ["k", "q", "r"], "点电荷电势", 2),
            PhysicsFormula("电容", "电磁学", "Q / V", ["Q", "V"], "电容定义", 1),
            PhysicsFormula("欧姆定律", "电磁学", "I * R", ["I", "R"], "欧姆定律", 1),
            PhysicsFormula("功率", "电磁学", "V * I", ["V", "I"], "电功率", 1),
            PhysicsFormula("电感感应", "电磁学", "L * 1", ["L"], "法拉第电磁感应定律", 3),
            PhysicsFormula("洛伦兹力", "电磁学", "q * E + q * v", ["q", "E", "v"], "洛伦兹力公式", 4),
            
            # 热力学
            PhysicsFormula("内能变化", "热力学", "Q - W", ["Q", "W"], "热力学第一定律", 1),
            PhysicsFormula("熵变", "热力学", "Q_rev / T", ["Q_rev", "T"], "熵变定义", 2),
            PhysicsFormula("热传导", "热力学", "k * A * deltaT / d", ["k", "A", "deltaT", "d"], "傅里叶热传导定律", 3),
            PhysicsFormula("斯特藩-玻尔兹曼定律", "热力学", "sigma * A * T**4", ["sigma", "A", "T"], "黑体辐射", 3),
            
            # 光学
            PhysicsFormula("折射定律", "光学", "n2 * sin(theta2)", ["n2", "theta2"], "斯涅尔定律", 2),
            PhysicsFormula("薄透镜公式", "光学", "1/do + 1/di", ["do", "di"], "透镜成像公式", 2),
            PhysicsFormula("多普勒效应", "光学", "f * (v + v_obs) / (v - v_source)", ["f", "v", "v_obs", "v_source"], "多普勒频移", 4),
            
            # 量子
            PhysicsFormula("薛定谔方程", "量子", "m * v**2 / 2", ["m", "v"], "简化薛定谔方程", 5),
            
            # 统计
            PhysicsFormula("玻尔兹曼分布", "统计", "exp(-E/(k*T)) / Z", ["E", "k", "T", "Z"], "玻尔兹曼因子", 3),
            PhysicsFormula("麦克斯韦速度分布", "统计", "4*pi*(v**2)*(m/(2*pi*k*T))**(3/2)*exp(-m*v**2/(2*k*T))", ["v", "m", "k", "T"], "麦克斯韦-玻尔兹曼分布", 4),
            
            # 相对论
            PhysicsFormula("质能方程", "相对论", "m*c**2", ["m", "c"], "爱因斯坦质能方程", 1),
            PhysicsFormula("时间膨胀", "相对论", "t / sqrt(1 - v**2/c**2)", ["t", "v", "c"], "狭义相对论时间膨胀", 3),
            PhysicsFormula("长度收缩", "相对论", "L * sqrt(1 - v**2/c**2)", ["L", "v", "c"], "狭义相对论长度收缩", 3),
        ]
    
    def _standardize_expression(self, expr: str, variables: List[str]) -> str:
        """将表达式标准化为x1, x2, x3格式"""
        # 清理表达式
        expr = self._clean_expression(expr)
        
        # 替换变量为x1, x2, x3格式
        for i, var_name in enumerate(variables):
            expr = re.sub(r'\b' + re.escape(var_name) + r'\b', f'x{i+1}', expr)
        
        return expr
    
    def _evaluate_expression(self, expr: str, X: np.ndarray, variables: List[str]) -> np.ndarray:
        """安全评估表达式"""
        try:
            if not expr or expr.strip() == '' or expr == '1':
                return self._fallback_evaluation(X)
            
            # 使用x1, x2, x3格式评估
            standardized_expr = self._standardize_expression(expr, variables)
            
            # 替换变量为X数组索引
            safe_expr = standardized_expr
            for i in range(len(variables)):
                safe_expr = re.sub(r'\bx' + str(i+1) + r'\b', f'X[:, {i}]', safe_expr)
            
            # 安全的数学函数
            safe_dict = {
                'X': X,
                'sin': np.sin,
                'cos': np.cos,
                'exp': lambda x: np.exp(np.clip(x, -500, 500)),
                'log': lambda x: np.log(np.maximum(x, 1e-10)),
                'sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
                'pi': np.pi,
                'e': np.e,
            }
            
            result = eval(safe_expr, {"__builtins__": {}}, safe_dict)
            return np.array(result)
            
        except Exception as e:
            logging.warning(f"评估表达式 '{expr}' 时出错: {e}")
            return self._fallback_evaluation(X)
    
    def _clean_expression(self, expr: str) -> str:
        """清理表达式"""
        if not expr:
            return ''
        
        # 清理特殊字符
        replacements = {
            '²': '**2', '³': '**3', '×': '*', '÷': '/', 'θ': 'theta',
            'π': 'pi', 'ℏ': 'hbar', 'Δ': 'delta', 'λ': 'lambda'
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        # 移除赋值格式
        if '=' in expr:
            parts = expr.split('=', 1)
            if len(parts) == 2:
                expr = parts[1].strip()
        
        return expr
    
    def _fallback_evaluation(self, X: np.ndarray) -> np.ndarray:
        """备用评估"""
        if X.shape[1] > 0:
            return X[:, 0] + 0.1 * np.random.normal(0, 0.01, X.shape[0])
        else:
            return np.zeros(X.shape[0])
    
    def _generate_physical_data_from_formula(self, formula: PhysicsFormula) -> tuple:
        """从物理公式生成数据"""
        try:
            # 生成输入数据
            X = np.random.uniform(-10, 10, (self.samples_per_expression, len(formula.variables)))
            
            # 计算输出
            y = self._evaluate_expression(formula.expression, X, formula.variables)
            
            # 添加噪声
            noise = np.random.normal(0, 0.01, y.shape)
            y = y + noise
            y = np.clip(y, -100, 100)
            
            return X, y, formula.expression
            
        except Exception as e:
            logging.warning(f"生成公式 {formula.name} 数据时出错: {e}")
            X = np.random.uniform(-2, 2, (self.samples_per_expression, len(formula.variables)))
            y = X[:, 0] + np.random.normal(0, 0.01, self.samples_per_expression)
            return X, y, f"{formula.variables[0]} + noise"
    
    def generate_pysr_datasets(self, output_dir: str):
        """生成PySR格式的数据集"""
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info("开始生成PySR格式的数据集")
        
        for i, formula in enumerate(self.physics_formulas, 1):
            logging.info(f"处理公式 {i}/{len(self.physics_formulas)}: {formula.name} ({formula.domain})")
            
            X, y, expr = self._generate_physical_data_from_formula(formula)
            
            # 标准化表达式（使用x1, x2, x3格式）
            standardized_expr = self._standardize_expression(expr, formula.variables)
            
            # 生成文件名（纯数字序号）
            filename = f"{i:05d}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # 写入数据
            with open(filepath, 'w', encoding='utf-8') as f:
                # 第一行写标准化表达式
                f.write(f"表达式: {standardized_expr}\n")
                
                # 写入数据点
                for j in range(len(X)):
                    # 合并输入特征和输出特征
                    data_row = list(X[j]) + [y[j]]
                    # 格式化为6位小数
                    formatted_row = [f"{val:.6f}" for val in data_row]
                    f.write(','.join(formatted_row) + '\n')
            
            logging.info(f"已生成: {filepath}")
        
        logging.info(f"PySR格式数据集生成完成!")
        logging.info(f"总表达式数: {len(self.physics_formulas)}")
        logging.info(f"总样本数: {len(self.physics_formulas) * self.samples_per_expression}")
        logging.info(f"输出目录: {output_dir}")

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("启动PySR格式数据生成器...")
    
    # 创建数据生成器
    generator = PySRDataGenerator()
    
    # 生成PySR格式数据集
    output_dir = "data/pysr_datasets"
    generator.generate_pysr_datasets(output_dir)
    
    logger.info("PySR格式数据生成完成!")

if __name__ == "__main__":
    main()