"""
数据加载器

专门处理txt格式的数据集，支持多种数据格式的加载和预处理。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import os
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