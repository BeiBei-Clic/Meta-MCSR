# Meta-MCSR: 基于自学习奖励网络的符号回归系统

Meta-MCSR是一个基于蒙特卡洛树搜索(MCTS)和自学习奖励网络的符号回归系统。通过表达式编码器和奖励网络的协同工作，实现更准确的奖励信号和更高效的符号回归搜索。

## 核心特点

- **算法与训练分离**: 核心算法在 `src/` 文件夹中，可独立导入使用
- **PyTorch最佳实践**: 遵循模块化设计原则
- **数据生成**: 训练文件负责生成数据集和训练流程
- **通过导入调用**: 外部文件通过 `from src.xxx import xxx` 调用核心算法

## 快速开始

### 使用流程

1. **训练表达式编码器**
   ```bash
   python expression_encoder_training.py
   ```

2. **训练奖励网络**
   ```bash
   python reward_network_training.py
   ```

3. **运行增强MCTS**
   ```bash
   # 复合函数测试（自动训练/测试分割）
   python mcts_with_reward_network.py composite
   
   # 指定数据集文件测试（自动训练/测试分割）
   python mcts_with_reward_network.py dataset-file --dataset-path your_data.npz
   ```

**重要改进**: 两种测试模式现在都自动进行训练/测试集分割，提供更可靠的泛化性能评估！

## 测试模式说明

### 1. 复合函数测试 (composite)
运行预定义的复合函数符号回归测试，适用于验证系统性能。

```bash
python mcts_with_reward_network.py composite
```

**复合函数详情：**
- **函数形式**: `y = x1 + 2*x2*sin(x1) + 0.5*x3^2 + 噪声`
- **特征变量**: x1, x2, x3
- **样本数量**: 100个
- **数据范围**: 特征在[-3, 3]范围内均匀分布
- **函数特点**: 
  - 线性项：x1
  - 三角函数：sin(x1) 
  - 多项式：x3^2
  - 交互项：x2*sin(x1)
- **噪声**: 加性高斯噪声（标准差0.1）

### 2. 指定数据集文件测试 (dataset-file)
使用用户提供的自定义数据集进行符号回归测试。

```bash
python mcts_with_reward_network.py dataset-file --dataset-path your_data.npz
```

### 3. Feynman数据集测试
使用费曼物理方程数据集进行符号回归测试。这些数据集的格式规范：
- **数据格式**: 每行包含多列数据，最后一列为标签y，前面为输入特征
- **文件格式**: 纯文本文件，每行空格分隔
- **数据类型**: 费曼物理方程的数值数据

**在费曼数据集 I.6.2 上运行示例：**
```bash
# 使用默认参数在 I.6.2 数据集上运行
python mcts_with_reward_network.py dataset-file --dataset-path dataset/Feynman_with_units/I.6.2

# 自定义参数运行
python mcts_with_reward_network.py dataset-file \
    --dataset-path dataset/Feynman_with_units/I.6.2 \
    --max-iterations 1000 \
    --max-depth 8 \
    --train-test-split 0.7 \
    --expr-encoder weights/expression_encoder \
    --reward-network weights/reward_network_final

# 查看帮助信息
python mcts_with_reward_network.py --help
```

**I.6.2 数据集说明：**
- **数据规模**: 100万行数据，2个特征
- **特征含义**: 最后一列为输出y，前两列为输入特征x1, x2
- **数据范围**: 特征值为浮点数，包含物理意义
- **运行示例**: 
  ```bash
  # 快速测试（10次迭代）
  python mcts_with_reward_network.py dataset-file \
      --dataset-path dataset/Feynman_with_units/I.6.2 \
      --max-iterations 10
  ```

**实际运行效果**：
- 数据加载: 从文本文件正确解析100万行数据
- 特征数: 2个输入特征 + 1个输出标签
- 算法性能: 可获得R2 > 0.8的优秀拟合效果
- 训练/测试分割: 自动进行80%训练，20%测试分割，评估泛化性能

**输出说明**：
- `训练集 R2/RMSE`: 在训练集上的性能指标
- `测试集 R2/RMSE`: 在测试集上的性能指标（更重要的泛化指标）
- 过拟合检测: 自动检测训练-测试性能差值，警告潜在的过拟合问题

### 命令行参数说明

```bash
# 查看完整帮助信息
python mcts_with_reward_network.py --help

# 自定义模型路径和参数
python mcts_with_reward_network.py composite \
    --expr-encoder weights/expression_encoder \
    --reward-network weights/reward_network_final \
    --max-iterations 1000 \
    --max-depth 10 \
    --train-test-split 0.8
```

**主要参数：**
- `--expr-encoder`: 指定表达式编码器模型路径
- `--reward-network`: 指定奖励网络模型路径  
- `--max-iterations`: MCTS最大迭代次数（默认500）
- `--max-depth`: MCTS最大深度（默认8）
- `--train-test-split`: 训练/测试集分割比例（默认0.8，即80%训练，20%测试）
- `--dataset-path`: dataset-file模式下的数据文件路径

**数据集文件格式要求：**

**1. NPZ格式 (推荐):**
- 支持numpy .npz格式
- 必须包含'X'（输入数据）和'y'（输出数据）数组
- X的形状应该是 (n_samples, n_features)
- y的形状应该是 (n_samples,)

**2. 纯文本格式 (如费曼数据集):**
- **文件格式**: 纯文本文件，每行空格分隔
- **列格式**: 最后一列为标签y，前面为输入特征
- **数据类型**: 浮点数数据
- **示例**: 三列数据表示为 `x1 x2 y`，其中y为标签，x1,x2为特征
- **自动检测**: 代码会自动检测并处理这种格式，无需手动转换

**费曼数据集格式转换示例：**
```python
import numpy as np

# 读取费曼数据集I.6.2 (纯文本格式)
data = np.loadtxt('dataset/Feynman_with_units/I.6.2')

# 数据格式: 前两列为特征，最后一列为标签
X = data[:, :-1]  # 前两列作为输入特征
y = data[:, -1]   # 最后一列作为标签

print(f"X shape: {X.shape}, y shape: {y.shape}")

# 转换为NPZ格式保存
np.savez('feynman_I6_2_processed.npz', X=X, y=y)

# 验证转换结果
loaded_data = np.load('feynman_I6_2_processed.npz')
print(f"转换后 - X shape: {loaded_data['X'].shape}, y shape: {loaded_data['y'].shape}")
```

## 环境设置

项目使用 `uv` 进行包管理，推荐使用以下方式：

```bash
# 使用uv安装依赖（推荐）
uv sync

# 或使用pip安装
pip install torch numpy matplotlib scikit-learn pyyaml seaborn tqdm python-dotenv requests

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 验证安装
python mcts_with_reward_network.py --help
```

### 依赖要求

- **Python**: >= 3.11
- **PyTorch**: >= 2.9.1  
- **NumPy**: 最新版本
- **其他依赖**: matplotlib, scikit-learn, pyyaml, seaborn, tqdm, python-dotenv, requests

### 依赖列表（来自pyproject.toml）
- torch>=2.9.1
- numpy
- matplotlib  
- scikit-learn
- pyyaml
- seaborn
- tqdm
- python-dotenv
- requests

## 工具脚本

### 权重清理工具

清理训练产生的临时权重文件：

```bash
python tools/clean_weights.py
```

这将清理weights/目录中非最终的模型文件，保留最终训练完成的模型。

## 输出和模型

训练过程会生成以下文件：
- `weights/expression_encoder` - 预训练的表达 encoder模型
- `weights/reward_network_final` - 最终的奖励网络模型
- `checkpoints/` - 训练过程中的检查点
- `training_logs/` - 训练日志和历史记录

## 常见问题

### 1. 模型加载失败
- 确保weights/目录下存在对应的模型文件
- 检查模型文件路径是否正确
- 验证PyTorch版本兼容性

### 2. 内存不足
- 减少 `--max-iterations` 参数
- 降低 `--max-depth` 参数
- 使用更小的数据集

### 3. 训练时间过长
- 调整MCTS参数：`--max-iterations`, `--max-depth`
- 使用更快的硬件（GPU推荐）
- 简化问题复杂度

### 4. 结果不理想
- 增加迭代次数
- 调整搜索深度
- 检查数据质量和范围