"""
数据加载器

专门处理txt格式的数据集，支持多种数据格式的加载和预处理。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import subprocess
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class SymbolicRegressionDataset:
    """符号回归数据集类"""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.X = X
        self.y = y
        self.variables = variables or [f'x{i+1}' for i in range(X.shape[1])]
        self.metadata = metadata or {}
        
        # 数据统计信息
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.X[idx], self.y[idx]
        elif isinstance(idx, list):
            return self.X[idx], self.y[idx]
        else:
            raise IndexError("不支持的索引类型")


class DataLoader:
    """数据加载器，支持多种格式"""
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        random_state: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.1
    ):
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        
        self.scaler = None
        
    def _create_scaler(self):
        """创建数据缩放器"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            return None
    
    def load_from_csv(
        self,
        file_path: str,
        target_column: str = 'y',
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> SymbolicRegressionDataset:
        """
        从CSV文件加载数据
        
        Args:
            file_path: CSV文件路径
            target_column: 目标变量列名
            feature_columns: 特征列名列表，如果为None则使用除目标列外的所有列
            **kwargs: 传递给pandas.read_csv的额外参数
            
        Returns:
            SymbolicRegressionDataset对象
        """
        df = pd.read_csv(file_path, **kwargs)
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        metadata = {
            'source': 'csv',
            'file_path': file_path,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'original_shape': df.shape
        }
        
        return SymbolicRegressionDataset(X, y, feature_columns, metadata)
    
    def load_from_npy(
        self,
        X_path: str,
        y_path: str,
        variables: Optional[List[str]] = None
    ) -> SymbolicRegressionDataset:
        """
        从numpy文件加载数据
        
        Args:
            X_path: 特征矩阵文件路径
            y_path: 目标向量文件路径
            variables: 变量名列表
            
        Returns:
            SymbolicRegressionDataset对象
        """
        X = np.load(X_path)
        y = np.load(y_path)
        
        metadata = {
            'source': 'npy',
            'X_path': X_path,
            'y_path': y_path
        }
        
        return SymbolicRegressionDataset(X, y, variables, metadata)
    
    def load_from_pretrain_txt(
        self,
        file_path: str,
        max_expressions: Optional[int] = None
    ) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        从预训练的 datasets.txt 文件加载数据
        
        文件格式如下：
        === Expression 1 ===
        Expression: exp(x1)
        Sample input data:
          Sample 1: X=[2.08269263], y=8.024869995498069
          Sample 2: X=[3.29458438], y=26.846634573080276
          ...
        
        Args:
            file_path: 文件路径
            max_expressions: 最大表达式数量，None表示加载所有
            
        Returns:
            (表达式列表, 数据集列表)
        """
        import re
        
        expressions = []
        datasets = []
        
        current_expression = None
        current_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # 检查是否是新的表达式开始
            if line.startswith('=== Expression'):
                # 保存前一个表达式（如果有）
                if current_expression is not None and current_data:
                    X_data = np.array([sample[0] for sample in current_data])
                    y_data = np.array([sample[1] for sample in current_data])
                    datasets.append((X_data, y_data))
                
                # 检查是否达到最大数量限制
                if max_expressions and len(expressions) >= max_expressions:
                    break
                
                current_expression = None
                current_data = []
                
            # 解析表达式
            elif line.startswith('Expression:'):
                current_expression = line.replace('Expression:', '').strip()
                expressions.append(current_expression)
                
            # 解析数据样本
            elif line.startswith('Sample') and 'X=' in line and 'y=' in line:
                # 使用正则表达式提取X和y值
                # Sample 1: X=[2.08269263], y=8.024869995498069
                pattern = r'X=\[([^\]]+)\], y=([-\d.]+)'
                match = re.search(pattern, line)
                
                if match:
                    x_str = match.group(1)
                    y_str = match.group(2)
                    
                    try:
                        # 解析X值（可能是逗号分隔的多个值）
                        if ',' in x_str:
                            x_values = [float(x.strip()) for x in x_str.split(',')]
                        else:
                            x_values = [float(x_str)]
                        
                        y_value = float(y_str)
                        
                        current_data.append((x_values, y_value))
                        
                    except ValueError:
                        # 如果解析失败，跳过这一行
                        continue
        
        # 保存最后一个表达式（如果有）
        if current_expression is not None and current_data:
            X_data = np.array([sample[0] for sample in current_data])
            y_data = np.array([sample[1] for sample in current_data])
            datasets.append((X_data, y_data))
        
        # 验证表达式和数据集数量一致性
        if len(expressions) != len(datasets):
            print(f"警告：表达式数量 ({len(expressions)}) 与数据集数量 ({len(datasets)}) 不一致")
            # 如果数量不一致，取较小的数量以保证配对
            min_count = min(len(expressions), len(datasets))
            expressions = expressions[:min_count]
            datasets = datasets[:min_count]
            print(f"已截取到 {min_count} 个有效的表达式-数据集对")
        
        print(f"成功解析 {len(expressions)} 个表达式，共 {len(datasets)} 个数据集")
        
        return expressions, datasets

    def load_from_txt(
        self,
        file_path: str,
        delimiter: str = ' ',
        skip_rows: int = 0,
        target_column: int = -1,
        feature_columns: Optional[List[int]] = None,
        variable_names: Optional[List[str]] = None
    ) -> SymbolicRegressionDataset:
        """
        从TXT文件加载数据
        
        Args:
            file_path: TXT文件路径
            delimiter: 分隔符
            skip_rows: 跳过的行数
            target_column: 目标列索引（负数表示倒数第几列）
            feature_columns: 特征列索引列表
            variable_names: 变量名列表
            
        Returns:
            SymbolicRegressionDataset对象
        """
        data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip_rows)
        
        # 处理列索引
        if feature_columns is None:
            if target_column < 0:
                feature_columns = list(range(data.shape[1] + target_column))
            else:
                feature_columns = [i for i in range(data.shape[1]) if i != target_column]
        
        X = data[:, feature_columns]
        y = data[:, target_column] if target_column >= 0 else data[:, target_column]
        
        if variable_names is None:
            variable_names = [f'x{i+1}' for i in range(len(feature_columns))]
        
        metadata = {
            'source': 'txt',
            'file_path': file_path,
            'delimiter': delimiter,
            'skip_rows': skip_rows,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'original_shape': data.shape
        }
        
        return SymbolicRegressionDataset(X, y, variable_names, metadata)
    
    def load_from_dict(
        self,
        data_dict: Dict[str, np.ndarray],
        target_key: str = 'y'
    ) -> SymbolicRegressionDataset:
        """
        从字典加载数据
        
        Args:
            data_dict: 包含特征和目标值的字典
            target_key: 目标值的键
            
        Returns:
            SymbolicRegressionDataset对象
        """
        X_dict = {k: v for k, v in data_dict.items() if k != target_key}
        y = data_dict[target_key]
        
        variables = list(X_dict.keys())
        X = np.column_stack([X_dict[var] for var in variables])
        
        metadata = {
            'source': 'dict',
            'target_key': target_key,
            'variables': variables
        }
        
        return SymbolicRegressionDataset(X, y, variables, metadata)
    
    def preprocess_dataset(
        self,
        dataset: SymbolicRegressionDataset,
        fit_scaler: bool = True,
        return_scaler: bool = False
    ) -> Union[SymbolicRegressionDataset, Tuple[SymbolicRegressionDataset, Any]]:
        """
        预处理数据集
        
        Args:
            dataset: 输入数据集
            fit_scaler: 是否拟合缩放器
            return_scaler: 是否返回缩放器
            
        Returns:
            预处理后的数据集，如果return_scaler为True则返回(数据集, 缩放器)
        """
        X_processed = dataset.X.copy()
        
        # 创建并应用缩放器
        if self.scaler_type != 'none':
            if fit_scaler:
                self.scaler = self._create_scaler()
                X_processed = self.scaler.fit_transform(X_processed)
            elif self.scaler is not None:
                X_processed = self.scaler.transform(X_processed)
        
        # 创建新的数据集对象
        processed_dataset = SymbolicRegressionDataset(
            X_processed,
            dataset.y.copy(),
            dataset.variables.copy(),
            dataset.metadata.copy()
        )
        
        if return_scaler and fit_scaler:
            return processed_dataset, self.scaler
        else:
            return processed_dataset
    
    def split_dataset(
        self,
        dataset: SymbolicRegressionDataset,
        stratify: Optional[np.ndarray] = None,
        return_splits: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
               Tuple[SymbolicRegressionDataset, SymbolicRegressionDataset, 
                     SymbolicRegressionDataset, SymbolicRegressionDataset]]:
        """
        分割数据集为训练、验证和测试集
        
        Args:
            dataset: 输入数据集
            stratify: 分层标签
            return_splits: 是否返回SymbolicRegressionDataset对象
            
        Returns:
            分割后的数据，如果return_splits为True则返回Dataset对象
        """
        X, y = dataset.X, dataset.y
        
        # 首先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state,
            stratify=stratify
        )
        
        # 从剩余数据中分割出验证集
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify if stratify is not None else None
        )
        
        if return_splits:
            train_dataset = SymbolicRegressionDataset(
                X_train, y_train, dataset.variables, 
                {**dataset.metadata, 'split': 'train'}
            )
            val_dataset = SymbolicRegressionDataset(
                X_val, y_val, dataset.variables,
                {**dataset.metadata, 'split': 'val'}
            )
            test_dataset = SymbolicRegressionDataset(
                X_test, y_test, dataset.variables,
                {**dataset.metadata, 'split': 'test'}
            )
            
            return train_dataset, val_dataset, test_dataset, test_dataset
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_and_prepare_pretrain_data(
        self,
        data_source: str,
        max_expressions: Optional[int] = None
    ) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        加载并准备预训练数据
        
        Args:
            data_source: 数据源路径
            max_expressions: 最大表达式数量
            
        Returns:
            (表达式列表, 数据集列表)
        """
        return self.load_from_pretrain_txt(data_source, max_expressions)

    def load_and_prepare(
        self,
        data_source: Union[str, Dict, np.ndarray],
        data_type: str = 'auto',
        **kwargs
    ) -> SymbolicRegressionDataset:
        """
        加载并准备数据
        
        Args:
            data_source: 数据源（文件路径、字典或数组）
            data_type: 数据类型 ('csv', 'npy', 'txt', 'dict', 'array')
            **kwargs: 加载参数
            
        Returns:
            准备好的数据集
        """
        # 自动检测数据类型
        if data_type == 'auto':
            if isinstance(data_source, str):
                if data_source.endswith('.csv'):
                    data_type = 'csv'
                elif data_source.endswith('.npy'):
                    data_type = 'npy'
                elif data_source.endswith('.txt'):
                    # 检查是否是预训练数据格式
                    with open(data_source, 'r') as f:
                        first_lines = [f.readline().strip() for _ in range(10)]
                        if any(line.startswith('=== Expression') for line in first_lines):
                            data_type = 'pretrain_txt'
                        else:
                            data_type = 'txt'
                else:
                    raise ValueError(f"无法自动检测文件类型: {data_source}")
            elif isinstance(data_source, dict):
                data_type = 'dict'
            elif isinstance(data_source, tuple) and len(data_source) == 2:
                data_type = 'array'
            else:
                raise ValueError(f"不支持的数据源类型: {type(data_source)}")
        
        # 加载数据
        if data_type == 'csv':
            dataset = self.load_from_csv(data_source, **kwargs)
        elif data_type == 'npy':
            if len(kwargs) == 0 and isinstance(data_source, tuple):
                X_path, y_path = data_source
                dataset = self.load_from_npy(X_path, y_path)
            else:
                dataset = self.load_from_npy(data_source, **kwargs)
        elif data_type == 'txt':
            dataset = self.load_from_txt(data_source, **kwargs)
        elif data_type == 'pretrain_txt':
            # 对于预训练数据格式，返回原始的表达式和数据集列表
            return self.load_from_pretrain_txt(data_source, **kwargs)
        elif data_type == 'dict':
            dataset = self.load_from_dict(data_source, **kwargs)
        elif data_type == 'array':
            X, y = data_source
            dataset = SymbolicRegressionDataset(X, y)
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
        
        # 预处理
        dataset = self.preprocess_dataset(dataset)
        
        return dataset
    
    def load_pysr_data(
        self, 
        data_source: str,
        auto_generate: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        统一加载PySR格式数据的函数
        
        支持文件或目录路径，解析包含表达式和数据样本的txt文件
        
        Args:
            data_source: 数据路径（文件或目录）
            auto_generate: 如果数据不存在是否自动生成
            
        Returns:
            数据字典列表，每个包含'expression'和'samples'键
        """
        if not os.path.exists(data_source):
            if auto_generate:
                print(f"数据路径不存在，开始生成数据...")
                # 获取项目根目录
                project_root = Path(__file__).parent.parent.parent
                generate_script = os.path.join(project_root, 'scripts', 'generate_pretrain_data_PySR.py')
                subprocess.run([sys.executable, generate_script], cwd=str(project_root))
            else:
                return None
        
        datasets = []
        
        # 获取所有txt文件
        if os.path.isdir(data_source):
            txt_files = [f for f in os.listdir(data_source) if f.endswith('.txt')]
            file_paths = [os.path.join(data_source, f) for f in txt_files]
        else:
            file_paths = [data_source]
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                continue
            
            # 解析表达式
            expression = lines[0].replace('表达式: ', '').strip()
            
            # 解析数据行
            samples = []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 2:
                    x_values = [float(part) for part in parts[:-1]]
                    y_value = float(parts[-1])
                    samples.append((x_values, y_value))
            
            if expression and samples:
                datasets.append({
                    'expression': expression,
                    'samples': samples
                })
        
        return datasets

    def generate_synthetic_data(
        self,
        expression: str,
        n_samples: int = 1000,
        n_features: int = 2,
        variables_range: Tuple[float, float] = (-5, 5),
        noise_level: float = 0.01,
        variable_names: Optional[List[str]] = None
    ) -> SymbolicRegressionDataset:
        """
        生成合成数据
        
        Args:
            expression: 数学表达式字符串
            n_samples: 样本数量
            n_features: 特征数量
            variables_range: 变量取值范围
            noise_level: 噪声水平
            variable_names: 变量名列表
            
        Returns:
            合成的数据集
        """
        # 生成输入数据
        np.random.seed(self.random_state)
        
        X = np.random.uniform(
            variables_range[0], variables_range[1], 
            (n_samples, n_features)
        )
        
        # 安全的表达式求值
        try:
            # 准备变量字典
            var_dict = {}
            if variable_names is None:
                variable_names = [f'x{i+1}' for i in range(n_features)]
            
            for i, var_name in enumerate(variable_names):
                var_dict[var_name] = X[:, i]
            
            # 添加允许的函数
            allowed_functions = {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'abs': np.abs, 'pi': np.pi, 'e': np.e
            }
            
            var_dict.update(allowed_functions)
            
            # 安全求值
            y = eval(expression, {"__builtins__": {}}, var_dict)
            
        except Exception as e:
            # 如果表达式求值失败，使用简单的线性组合
            y = np.sum(X, axis=1) + np.random.normal(0, noise_level, n_samples)
        
        # 添加噪声
        if noise_level > 0:
            y += np.random.normal(0, noise_level, n_samples)
        
        metadata = {
            'source': 'synthetic',
            'expression': expression,
            'n_samples': n_samples,
            'n_features': n_features,
            'variables_range': variables_range,
            'noise_level': noise_level
        }
        
        return SymbolicRegressionDataset(X, y, variable_names, metadata)


# 便捷函数
def load_data(
    data_path: str,
    data_format: str = 'auto',
    **kwargs
) -> SymbolicRegressionDataset:
    """
    快速加载数据
    
    Args:
        data_path: 数据文件路径
        data_format: 数据格式
        **kwargs: 额外参数
        
    Returns:
        加载的数据集
    """
    loader = DataLoader()
    return loader.load_and_prepare(data_path, data_format, **kwargs)


def generate_data(
    expression: str,
    n_samples: int = 1000,
    **kwargs
) -> SymbolicRegressionDataset:
    """
    快速生成合成数据
    
    Args:
        expression: 表达式字符串
        n_samples: 样本数量
        **kwargs: 额外参数
        
    Returns:
        生成的数据集
    """
    loader = DataLoader()
    return loader.generate_synthetic_data(expression, n_samples, **kwargs)


def load_pysr_data(
    data_source: str,
    auto_generate: bool = False
) -> Optional[List[Dict[str, Any]]]:
    """
    快速加载PySR格式数据
    
    Args:
        data_source: 数据路径（文件或目录）
        auto_generate: 如果数据不存在是否自动生成
        
    Returns:
        数据字典列表
    """
    loader = DataLoader()
    return loader.load_pysr_data(data_source, auto_generate)