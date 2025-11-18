"""
表达式解析器

提供表达式的解析、验证和操作功能。
"""

import re
import ast
import sys
import os
from typing import List, Dict, Set, Optional, Tuple, Any, Union
import numpy as np


class ExpressionParser:
    """表达式解析器"""
    
    def __init__(self):
        # 预定义的数学函数
        self.supported_functions = {
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
            'sinh', 'cosh', 'tanh',
            'exp', 'log', 'ln', 'log10', 'log2',
            'sqrt', 'cbrt',
            'abs', 'sign',
            'min', 'max',
            'floor', 'ceil', 'round'
        }
        
        # 预定义的数学常数
        self.supported_constants = {
            'pi', 'e', 'π', 'τ'
        }
        
        # 支持的运算符
        self.supported_operators = {
            '+', '-', '*', '/', '^', '**', '%', '//',
            '>', '<', '>=', '<=', '==', '!=',
            '&', '|', '~', 'and', 'or', 'not'
        }
        
        # 预定义的变量模式
        self.variable_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
        
    def tokenize(self, expression: str) -> List[str]:
        """
        将表达式分解为token
        
        Args:
            expression: 数学表达式字符串
            
        Returns:
            token列表
        """
        # 预处理表达式
        expression = self._preprocess(expression)
        
        # 分词模式
        patterns = [
            (r'\d+\.?\d*', 'NUMBER'),  # 数字
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),  # 标识符
            (r'[+\-*/^(),]', 'OPERATOR'),  # 运算符和括号
            (r'\s+', 'WHITESPACE'),  # 空白
        ]
        
        tokens = []
        pos = 0
        
        while pos < len(expression):
            matched = False
            
            for pattern, token_type in patterns:
                regex = re.compile(pattern)
                match = regex.match(expression, pos)
                
                if match:
                    value = match.group()
                    
                    # 跳过空白
                    if token_type != 'WHITESPACE':
                        tokens.append((value, token_type))
                    
                    pos = match.end()
                    matched = True
                    break
            
            if not matched:
                raise ValueError(f"无法解析表达式在位置 {pos}: {expression[pos:]}")
        
        return tokens
    
    def parse(self, expression: str) -> Dict[str, Any]:
        """
        解析表达式并返回语法树
        
        Args:
            expression: 数学表达式字符串
            
        Returns:
            语法树字典
        """
        try:
            # 预处理表达式
            processed_expr = self._preprocess(expression)
            
            # 使用Python的ast模块解析
            tree = ast.parse(processed_expr, mode='eval')
            
            # 转换为自定义格式
            parsed = self._ast_to_dict(tree.body)
            
            # 验证表达式
            validation_result = self.validate_expression(expression)
            parsed['validation'] = validation_result
            
            return parsed
            
        except SyntaxError as e:
            return {
                'error': f'语法错误: {str(e)}',
                'expression': expression,
                'position': e.offset if hasattr(e, 'offset') else None
            }
        except Exception as e:
            return {
                'error': f'解析错误: {str(e)}',
                'expression': expression
            }
    
    def validate_expression(self, expression: str) -> Dict[str, Any]:
        """
        验证表达式
        
        Args:
            expression: 数学表达式字符串
            
        Returns:
            验证结果
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'tokens': [],
            'variables': set(),
            'functions': set(),
            'complexity': 0
        }
        
        try:
            # 分词
            tokens = self.tokenize(expression)
            validation_result['tokens'] = tokens
            
            # 检查token
            for token_value, token_type in tokens:
                if token_type == 'IDENTIFIER':
                    if token_value in self.supported_functions:
                        validation_result['functions'].add(token_value)
                    elif token_value in self.supported_constants:
                        pass  # 常数，不需要特殊处理
                    elif token_value.lower() in ['x', 'y', 'z'] or self.variable_pattern.match(token_value):
                        validation_result['variables'].add(token_value)
                    else:
                        validation_result['warnings'].append(f"未知的标识符: {token_value}")
                
                elif token_type == 'NUMBER':
                    try:
                        float(token_value)
                    except ValueError:
                        validation_result['errors'].append(f"无效的数字: {token_value}")
            
            # 检查语法
            try:
                self.parse(expression)
            except Exception as e:
                validation_result['errors'].append(str(e))
            
            # 计算复杂度
            validation_result['complexity'] = self._calculate_complexity(tokens)
            
            # 如果有错误，标记为无效
            if validation_result['errors']:
                validation_result['is_valid'] = False
                
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"验证失败: {str(e)}")
        
        return validation_result
    
    def simplify_expression(self, expression: str) -> str:
        """
        简化表达式
        
        Args:
            expression: 原始表达式
            
        Returns:
            简化后的表达式
        """
        # 基本的简化规则
        simplifications = [
            # 移除多余的括号
            (r'\(([^()]+)\)', r'\1'),
            # 简化 +0 和 -0
            (r'\+0\b', ''),
            (r'-0\b', ''),
            (r'\*0\b', '0'),
            (r'0\b(?![0-9])', '0'),  # 单独的0
            # 简化 *1 和 /1
            (r'\*1\b', ''),
            (r'/1\b', ''),
            # 简化指数运算
            (r'\*\*2\b', '^2'),
            (r'\*\*3\b', '^3'),
        ]
        
        simplified = expression
        for pattern, replacement in simplifications:
            simplified = re.sub(pattern, replacement, simplified)
        
        # 清理多余的运算符
        simplified = re.sub(r'\s+', '', simplified)  # 移除空白
        simplified = re.sub(r'^\+', '', simplified)  # 移除开头的+
        simplified = re.sub(r'\+\-', '-', simplified)  # +- 变成 -
        simplified = re.sub(r'-\+', '-', simplified)  # -+ 变成 -
        
        return simplified
    
    def extract_variables(self, expression: str) -> Set[str]:
        """
        提取表达式中的变量
        
        Args:
            expression: 数学表达式
            
        Returns:
            变量集合
        """
        tokens = self.tokenize(expression)
        variables = set()
        
        for token_value, token_type in tokens:
            if token_type == 'IDENTIFIER':
                if (token_value not in self.supported_functions and 
                    token_value not in self.supported_constants and
                    token_value.lower() not in ['x', 'y', 'z'] and
                    not token_value.isdigit()):
                    variables.add(token_value)
        
        return variables
    
    def extract_functions(self, expression: str) -> Set[str]:
        """
        提取表达式中的函数
        
        Args:
            expression: 数学表达式
            
        Returns:
            函数集合
        """
        tokens = self.tokenize(expression)
        functions = set()
        
        for token_value, token_type in tokens:
            if token_type == 'IDENTIFIER' and token_value in self.supported_functions:
                functions.add(token_value)
        
        return functions
    
    def evaluate_expression(self, expression: str, variables: Dict[str, float]) -> float:
        """
        安全地计算表达式值
        
        Args:
            expression: 数学表达式
            variables: 变量字典
            
        Returns:
            表达式值
        """
        # 预处理表达式
        processed_expr = self._preprocess(expression)
        
        # 准备安全的数学函数
        def safe_sin(x):
            return np.sin(np.asarray(x))
        
        def safe_cos(x):
            return np.cos(np.asarray(x))
        
        def safe_tan(x):
            return np.tan(np.asarray(x))
        
        def safe_exp(x):
            # 限制指数函数的上限以避免溢出
            if np.isscalar(x):
                return np.exp(min(x, 500))
            else:
                x = np.asarray(x)
                x = np.clip(x, None, 500)  # 限制最大值为500
                return np.exp(x)
        
        def safe_log(x):
            # 对数函数只允许正值，并设置最小值避免log(0)
            if np.isscalar(x):
                if x <= 0:
                    x = 1e-10
                return np.log(max(x, 1e-10))
            else:
                x = np.asarray(x)
                x = np.maximum(x, 1e-10)  # 最小值设为1e-10
                return np.log(x)
        
        def safe_sqrt(x):
            # 平方根只允许非负数
            if np.isscalar(x):
                return np.sqrt(max(x, 0))
            else:
                x = np.asarray(x)
                return np.sqrt(np.maximum(x, 0))
        
        def safe_cbrt(x):
            # 立方根允许负数
            if np.isscalar(x):
                return np.sign(x) * np.power(abs(x), 1/3)
            else:
                x = np.asarray(x)
                return np.sign(x) * np.power(np.abs(x), 1/3)
        
        def safe_pow(base, exponent):
            # 安全的幂运算，避免0^0和负数的非整数次幂
            base = np.asarray(base)
            exponent = np.asarray(exponent)
            
            # 避免0^0
            if np.isscalar(base) and np.isscalar(exponent):
                if base == 0 and exponent == 0:
                    return 1.0  # 定义0^0 = 1
                if base < 0 and exponent != np.round(exponent):
                    # 负数的非整数次幂，使用复数，但我们返回实数部分
                    return abs(base) ** exponent  # 使用绝对值避免复数
            else:
                # 向量情况
                result = np.power(np.maximum(base, 1e-10), exponent)  # 避免负底数
                # 处理0^0情况
                zero_mask = (base == 0) & (exponent == 0)
                result[zero_mask] = 1.0
                return result
            
            return np.power(base, exponent)
        
        def safe_divide(a, b):
            # 安全的除法，避免除零
            if np.isscalar(a) and np.isscalar(b):
                if abs(b) < 1e-10:
                    # 如果分母接近0，返回一个小的值或根据情况调整
                    if abs(a) < 1e-10:
                        return 0.0
                    else:
                        return a * 1e10  # 避免除零，设为大数
            else:
                a = np.asarray(a)
                b = np.asarray(b)
                b = np.where(np.abs(b) < 1e-10, 1e-10, b)  # 替换接近0的分母
                return a / b
            
            return a / b
        
        # 准备执行环境
        env = variables.copy()
        env.update({
            # 安全的数学函数
            'sin': safe_sin, 'cos': safe_cos, 'tan': safe_tan,
            'exp': safe_exp, 'log': safe_log, 'ln': safe_log,
            'sqrt': safe_sqrt, 'cbrt': safe_cbrt, 'abs': np.abs,
            'pi': np.pi, 'e': np.e,
            # 安全函数
            'min': min, 'max': max, 'round': round, 'abs': abs,
            'pow': safe_pow
        })
        
        try:
            # 处理幂运算符号替换
            processed_expr = processed_expr.replace('^', '**')
            
            # 安全求值
            result = eval(processed_expr, {"__builtins__": {}}, env)
            
            # 最终的数值检查
            if np.isscalar(result):
                result = float(result)
                if np.isnan(result) or np.isinf(result):
                    raise ValueError("计算结果包含 NaN 或无穷值")
            else:
                result = np.asarray(result)
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    raise ValueError("计算结果包含 NaN 或无穷值")
                result = result.tolist() if hasattr(result, 'tolist') else float(result)
            
            return result
        except Exception as e:
            raise ValueError(f"表达式计算失败: {str(e)}")
    
    def convert_to_scientific_notation(self, expression: str, precision: int = 6) -> str:
        """
        将表达式中的数字转换为科学计数法
        
        Args:
            expression: 原始表达式
            precision: 精度
            
        Returns:
            转换后的表达式
        """
        def replace_number(match):
            number_str = match.group()
            try:
                number = float(number_str)
                if number != 0 and (abs(number) < 1e-3 or abs(number) > 1e6):
                    # 使用科学计数法
                    return f"{number:.{precision}e}"
                else:
                    # 保留原格式，但控制精度
                    return f"{number:.{precision}g}"
            except ValueError:
                return number_str
        
        # 匹配数字的正则表达式
        number_pattern = r'\b\d+\.?\d*(?:[eE][+-]?\d+)?\b'
        converted_expr = re.sub(number_pattern, replace_number, expression)
        
        return converted_expr
    
    def _preprocess(self, expression: str) -> str:
        """
        预处理表达式
        
        Args:
            expression: 原始表达式
            
        Returns:
            预处理后的表达式
        """
        # 移除多余的空白
        expression = re.sub(r'\s+', ' ', expression.strip())
        
        # 标准化运算符
        replacements = {
            '^': '**',  # 幂运算
            'π': 'pi',  # π符号
            'τ': '2*pi',  # τ符号
            'ln': 'log',  # ln函数
        }
        
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        
        # 处理隐式乘法 (如 2x -> 2*x)
        expression = self._handle_implicit_multiplication(expression)
        
        return expression
    
    def _handle_implicit_multiplication(self, expression: str) -> str:
        """
        处理隐式乘法 (如 2x -> 2*x, x( -> x*( )
        
        Args:
            expression: 原始表达式
            
        Returns:
            处理后的表达式
        """
        # 数字后跟变量或函数
        expression = re.sub(r'(\d)([a-zA-Z_])', r'\1*\2', expression)
        
        # 变量或函数后跟变量或左括号
        expression = re.sub(r'([a-zA-Z_])(\d)', r'\1*\2', expression)
        expression = re.sub(r'([a-zA-Z_])(\()', r'\1*(', expression)
        
        # 右括号后跟变量或左括号
        expression = re.sub(r'(\))([a-zA-Z_])', r'\1*\2', expression)
        expression = re.sub(r'(\))(\()', r'\1*(', expression)
        
        return expression
    
    def _ast_to_dict(self, node) -> Dict[str, Any]:
        """将AST节点转换为字典"""
        if isinstance(node, ast.BinOp):
            return {
                'type': 'binary_op',
                'op': self._get_operator_name(node.op),
                'left': self._ast_to_dict(node.left),
                'right': self._ast_to_dict(node.right)
            }
        elif isinstance(node, ast.UnaryOp):
            return {
                'type': 'unary_op',
                'op': self._get_operator_name(node.op),
                'operand': self._ast_to_dict(node.operand)
            }
        elif isinstance(node, ast.Call):
            return {
                'type': 'function_call',
                'func': node.func.id if hasattr(node.func, 'id') else str(node.func),
                'args': [self._ast_to_dict(arg) for arg in node.args]
            }
        elif isinstance(node, ast.Name):
            return {
                'type': 'variable',
                'name': node.id
            }
        elif isinstance(node, ast.Constant):
            return {
                'type': 'constant',
                'value': node.value
            }
        elif isinstance(node, ast.Num):  # Python < 3.8
            return {
                'type': 'constant',
                'value': node.n
            }
        else:
            return {
                'type': str(type(node).__name__),
                'content': ast.dump(node)
            }
    
    def _get_operator_name(self, op) -> str:
        """获取运算符名称"""
        operator_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Pow: '**',
            ast.Mod: '%',
            ast.USub: '-',
            ast.UAdd: '+',
        }
        return operator_map.get(type(op), str(type(op).__name__))
    
    def _calculate_complexity(self, tokens: List[Tuple[str, str]]) -> int:
        """计算表达式复杂度"""
        complexity = 0
        
        for token_value, token_type in tokens:
            if token_type == 'OPERATOR':
                if token_value in '+-':
                    complexity += 1
                elif token_value in '*/':
                    complexity += 2
                elif token_value in '^**':
                    complexity += 3
                elif token_value in '(),':
                    complexity += 0.5
            elif token_type == 'IDENTIFIER':
                if token_value in self.supported_functions:
                    complexity += 2
                else:
                    complexity += 1
        
        return int(complexity)


# 便捷函数
def parse_expression(expression: str) -> Dict[str, Any]:
    """
    快速解析表达式
    
    Args:
        expression: 数学表达式字符串
        
    Returns:
        解析结果
    """
    parser = ExpressionParser()
    return parser.parse(expression)


def validate_expression(expression: str) -> Dict[str, Any]:
    """
    快速验证表达式
    
    Args:
        expression: 数学表达式字符串
        
    Returns:
        验证结果
    """
    parser = ExpressionParser()
    return parser.validate_expression(expression)


def evaluate_expression(expression: str, variables: Dict[str, float]) -> float:
    """
    快速计算表达式值
    
    Args:
        expression: 数学表达式字符串
        variables: 变量字典
        
    Returns:
        表达式值
    """
    parser = ExpressionParser()
    return parser.evaluate_expression(expression, variables)