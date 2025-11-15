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
└── mcts_with_reward_network.py    # 增强MCTS应用示例
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
   python mcts_with_reward_network.py
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

- Python 3.11+
- PyTorch >= 2.9.1
- NumPy, Matplotlib, Scikit-learn
- 其他依赖见 `pyproject.toml`

## 开发信息

- **设计理念**: 算法实现与训练逻辑分离
- **项目结构**: 遵循PyTorch最佳实践
- **模块化**: 核心算法可独立使用和测试
- **扩展性**: 易于添加新的算法或训练方法