# Meta-MCSR: 基于自学习奖励网络的符号回归系统

Meta-MCSR是一个基于蒙特卡洛树搜索(MCTS)和自学习奖励网络的符号回归系统。通过表达式编码器和奖励网络的协同工作，实现更准确的奖励信号和更高效的符号回归搜索。

## 核心特点

- **算法与训练分离**: 核心算法在 `src/` 文件夹中，可独立导入使用
- **PyTorch最佳实践**: 遵循模块化设计原则
- **数据生成**: 训练文件负责生成数据集和训练流程
- **通过导入调用**: 外部文件通过 `from src.xxx import xxx` 调用核心算法

## 项目结构

```
Meta-MCSR/
├── src/                          # 核心算法实现
│   ├── expression_encoder.py     # 表达式编码器
│   ├── reward_network.py         # 奖励网络
│   └── mcts.py                   # 蒙特卡洛树搜索
│
├── expression_encoder_training.py # 表达式编码器训练 + 数据生成
├── reward_network_training.py     # 奖励网络训练 + 数据生成
├── mcts_with_reward_network.py    # 增强MCTS应用示例
│
├── weights/                       # 训练后的模型权重
│   ├── expression_encoder_*.pth   # 表达式编码器模型
│   └── reward_network_final_*.pth # 奖励网络模型
│
├── checkpoints/                   # 训练检查点
│   ├── expression_encoder/        # 编码器检查点
│   └── reward_network/            # 奖励网络检查点
│
├── dataset/                       # 数据集目录
│   └── Feynman_with_units/        # 费曼数据集
│
├── training_logs/                 # 训练日志
├── tools/                         # 工具脚本
│   └── clean_weights.py           # 权重清理工具
└── nd2py_package/                 # 符号计算包
```

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
   # 复合函数测试
   python mcts_with_reward_network.py composite
   
   # 指定数据集文件测试
   python mcts_with_reward_network.py dataset-file --dataset-path your_data.npz
   ```

### 直接使用核心算法

```python
from src.expression_encoder import ExpressionEmbedding
from src.reward_network import RewardNetwork
from src.mcts import MCTS

## API 使用示例
embedding = ExpressionEmbedding()
reward_network = RewardNetwork()
mcts = MCTS(use_reward_network=True)
```

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

### 命令行参数说明

```bash
# 查看完整帮助信息
python mcts_with_reward_network.py --help

# 自定义模型路径和参数
python mcts_with_reward_network.py composite \
    --expr-encoder weights/expression_encoder \
    --reward-network weights/reward_network_final \
    --max-iterations 1000 \
    --max-depth 10
```

**主要参数：**
- `--expr-encoder`: 指定表达式编码器模型路径
- `--reward-network`: 指定奖励网络模型路径  
- `--max-iterations`: MCTS最大迭代次数（默认500）
- `--max-depth`: MCTS最大深度（默认8）
- `--dataset-path`: dataset-file模式下的数据文件路径

**数据集文件格式要求：**
- 支持numpy .npz格式
- 必须包含'X'（输入数据）和'y'（输出数据）数组
- X的形状应该是 (n_samples, n_features)
- y的形状应该是 (n_samples,)

**示例数据集生成代码：**
```python
import numpy as np

# 生成示例数据
X = np.random.uniform(-3, 3, (100, 3))
y = X[:, 0] + 2 * X[:, 1] * np.sin(X[:, 0]) + 0.5 * X[:, 2]**2 + np.random.normal(0, 0.1, 100)

# 保存为npz格式
np.savez('my_data.npz', X=X, y=y)

# 加载和验证数据
data = np.load('my_data.npz')
print(f"X shape: {data['X'].shape}, y shape: {data['y'].shape}")
```

## 系统架构

本系统基于自学习的奖励网络增强的蒙特卡洛树搜索方法，主要包含三个核心组件：

1. **表达式编码器 (Expression Encoder)**
   - 使用Transformer架构将数学表达式编码为向量表示
   - 支持多种数学操作符：+、-、*、/、sin、cos、sqrt、log、exp等
   - 基于子词编码的字符级分词器

2. **奖励网络 (Reward Network)**
   - 结合表达式嵌入和数据特征预测奖励
   - 支持注意力机制和残差连接
   - 支持经验回放和在线学习

3. **增强蒙特卡洛树搜索 (Enhanced MCTS)**
   - 支持传统UCT搜索和基于网络的奖励预测
   - 可配置搜索深度、迭代次数等参数
   - 支持多种表达式生成策略

## 训练流程

### 1. 表达式编码器预训练
- 生成大规模随机数学表达式数据集
- 使用三元组损失进行对比学习预训练
- 训练后模型保存在 `weights/expression_encoder`

### 2. 奖励网络训练
- 使用预训练的表达式编码器初始化
- 通过MCTS搜索生成训练数据
- 联合训练奖励网络和微调编码器
- 最终模型保存在 `weights/reward_network_final`

### 3. 推理阶段
- 加载预训练的编码器和奖励网络
- 使用增强MCTS进行符号回归搜索
- 输出最优数学表达式

## 输出和模型

训练过程会生成以下文件：
- `weights/expression_encoder` - 预训练的表达 encoder模型
- `weights/reward_network_final` - 最终的奖励网络模型
- `checkpoints/` - 训练过程中的检查点
- `training_logs/` - 训练日志和历史记录

## 依赖要求

- **Python**: >= 3.11
- **PyTorch**: >= 2.9.1  
- **NumPy**: 最新版本
- **其他依赖**: matplotlib, scikit-learn, pyyaml, seaborn, tqdm, python-dotenv, requests
- **包管理**: 项目使用 `uv` 进行依赖管理

### 环境设置

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

## 开发信息

- **设计理念**: 算法实现与训练逻辑分离
- **项目结构**: 遵循PyTorch最佳实践
- **模块化**: 核心算法可独立使用和测试
- **扩展性**: 易于添加新的算法或训练方法

## 工具脚本

### 权重清理工具

清理训练产生的临时权重文件：

```bash
python tools/clean_weights.py
```

这将清理weights/目录中非最终的模型文件，保留最终训练完成的模型。

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