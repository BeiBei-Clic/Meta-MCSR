# 多元函数符号回归

这个项目扩展了原本只支持一元函数的MCTS符号回归算法，使其能够支持多元函数的符号回归。

## 主要特点

1. **统一实现**：不再区分一元和多元函数，使用相同的数据结构和处理流程
2. **灵活变量支持**：支持任意数量的输入变量，只需调整`max_vars`参数
3. **数值稳定性**：添加了数值稳定性处理，避免了线性代数计算中的警告和错误
4. **模块化设计**：核心搜索算法已分离到独立模块，便于集成和复用
5. **兼容性**：保持了原有API的兼容性，一元函数的使用方式不变

## 使用方法

### 基本用法

```python
import numpy as np
from mcts_core import MCTSSymbolicRegression

# 生成示例数据
np.random.seed(42)
n_samples = 100
X = np.random.uniform(-5, 5, (n_samples, 3))  # 3个特征
y = X[:, 0] + 2 * X[:, 1] * np.sin(X[:, 0]) + 0.5 * X[:, 2]**2 + np.random.normal(0, 0.1, n_samples)

# 创建并训练模型
model = MCTSSymbolicRegression(max_depth=10, max_iterations=1000, max_vars=5)

print("开始训练...")
best_expr = model.fit(X, y)
print(f"最佳表达式: {best_expr}")

# 评估模型
r2, rmse = model.get_score(X, y)
print(f"R2 分数: {r2:.4f}")
print(f"RMSE 分数: {rmse:.4f}")
```

### 参数说明

- `max_depth`: 表达式树的最大深度，默认为10
- `max_iterations`: MCTS算法的最大迭代次数，默认为1000
- `max_vars`: 最大变量数，默认为5
- `eta`: 复杂度惩罚系数，默认为0.999

## 示例

### 一元函数回归

```python
# 生成一元函数数据
X = np.random.uniform(-3, 3, (n_samples, 1))
y = X[:, 0]**2 + np.sin(X[:, 0]) + np.random.normal(0, 0.1, n_samples)

# 创建并训练模型
model = MCTSSymbolicRegression(max_depth=8, max_iterations=500, max_vars=3)
best_expr = model.fit(X, y)
```

### 多元函数回归

```python
# 生成多元函数数据
X = np.random.uniform(-3, 3, (n_samples, 5))
y = X[:, 0] + 2 * X[:, 1] * np.sin(X[:, 0]) + 0.5 * X[:, 2]**2 + X[:, 3] * X[:, 4] + np.random.normal(0, 0.1, n_samples)

# 创建并训练模型
model = MCTSSymbolicRegression(max_depth=8, max_iterations=500, max_vars=8)
best_expr = model.fit(X, y)
```

### 自定义变量名

```python
# 使用字典格式输入数据，支持自定义变量名
X_dict = {
    'temperature': np.random.uniform(0, 100, n_samples),
    'pressure': np.random.uniform(1, 10, n_samples),
    'humidity': np.random.uniform(0, 1, n_samples)
}
y = X_dict['temperature'] * X_dict['pressure'] + 5 * X_dict['humidity'] + np.random.normal(0, 1, n_samples)

# 创建并训练模型
model = MCTSSymbolicRegression(max_depth=5, max_iterations=500, max_vars=3)
best_expr = model.fit(X_dict, y)
```

## 测试

运行测试文件以验证代码的正确性：

```bash
python test_multivariate.py
```

## 实现细节

1. **Node类修改**：支持多个表达式树(eqtrees)和组合表达式(phi)
2. **数据预处理**：添加preprocess函数，支持numpy数组到变量字典的转换
3. **表达式评估**：支持多个表达式树的组合评估，使用线性回归方法组合
4. **变异操作**：支持对多个表达式树的变异操作，包括添加、替换和修改
5. **数值稳定性**：添加正则化和伪逆回退机制，提高数值稳定性
6. **模块化设计**：核心搜索算法分离到mcts_core.py，便于集成和复用

## 注意事项

1. 对于高维数据，建议增加`max_vars`参数以获得更好的性能
2. 如果遇到数值不稳定问题，可以调整`eta`参数以改变复杂度惩罚
3. 训练时间随`max_iterations`和`max_vars`增加而增加，需要根据实际情况调整
4. mcts_core.py是核心模块，专注于搜索算法，可直接导入使用