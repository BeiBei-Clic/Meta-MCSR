# 基于自学习"裁判"网络的MCTS符号回归系统

这是一个基于蒙特卡洛树搜索(MCTS)和自学习奖励网络的符号回归系统。该系统通过一个"裁判"网络来改进传统MCTS的奖励信号，实现更好的符号回归性能。

## 系统架构

### 核心组件

1. **表达式嵌入器 (Expression Encoder)**: 基于Transformer的神经网络，将数学表达式编码为向量表示
2. **奖励网络 (Reward Network)**: 评估表达式-数据对的"潜力"，结合性能分量和结构分量
3. **增强的MCTS引擎**: 集成奖励网络的蒙特卡洛树搜索
4. **经验回放池**: 存储历史经验供网络训练

### 文件结构

- `expression_encoder.py`: 表达式嵌入器核心实现
- `reward_network.py`: 奖励网络实现
- `mcts_enhanced.py`: 增强的MCTS实现
- `expression_encoder_training.py`: 表达式嵌入器预训练程序
- `expression_encoder_inference.py`: 表达式嵌入器预测程序
- `reward_network_training.py`: 奖励网络训练程序
- `mcts_with_reward_network.py`: 最终融合系统程序

## 使用说明

### 1. 环境准备

```bash
# 激活uv虚拟环境
uv venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
uv sync
```

### 2. 训练流程

#### 步骤1: 预训练表达式嵌入器

```bash
python expression_encoder_training.py
```

这个程序会：
- 生成50,000个随机数学表达式
- 使用掩码语言模型(MLM)进行预训练
- 保存模型到 `weights/expression_encoder`

#### 步骤2: 训练奖励网络

```bash
python reward_network_training.py
```

这个程序会：
- 生成多个符号回归问题数据集
- 使用MCTS生成训练经验
- 训练奖励网络并微调表达式嵌入器
- 保存模型到 `weights/reward_network_final`

### 3. 运行系统

#### 交互模式
```bash
python mcts_with_reward_network.py --mode interactive
```

#### 基准测试模式
```bash
python mcts_with_reward_network.py --mode benchmark
```

#### 单次运行
```bash
python mcts_with_reward_network.py --mode single --problem 1
```

### 4. 表达式嵌入器单独使用

#### 演示模式
```bash
python expression_encoder_inference.py demo
```

#### 交互模式
```bash
python expression_encoder_inference.py interactive
```

#### 命令行模式
```bash
# 编码单个表达式
python expression_encoder_inference.py encode "sin(x1) + cos(x2)"

# 计算相似度
python expression_encoder_inference.py similarity "sin(x1)" "cos(x1)"
```

## 核心特性

### 1. 自学习机制
- **神谕目标**: 使用真实表达式作为"靶心"，指导网络学习
- **混合奖励**: 结合性能奖励(R²)和结构奖励(相似度)
- **经验回放**: 存储历史经验，训练更稳定

### 2. 端到端训练
- 表达式嵌入器和奖励网络同时训练
- 通过MCTS探索生成高质量训练数据
- 自适应的网络更新机制

### 3. 多样化支持
- 支持多种数学函数：三角函数、指数、对数、多项式
- 支持复杂表达式组合
- 自适应复杂度控制

## 性能提升

相比传统基于R²分数的MCTS，该系统的优势：

1. **更准确的奖励信号**: 奖励网络能够更好地评估表达式的"潜力"
2. **结构感知**: 不仅考虑拟合性能，还考虑结构相似度
3. **自学习能力**: 通过经验累积不断改进评估能力
4. **探索效率**: 更智能的探索策略，减少无效搜索

## 技术细节

### 表达式嵌入器
- **架构**: 6层Transformer编码器
- **嵌入维度**: 256维
- **注意力头数**: 8
- **预训练任务**: 掩码语言模型(MLM)

### 奖励网络
- **表达式编码器**: 复用预训练的嵌入器
- **数据编码器**: 3层MLP
- **融合模块**: 交叉注意力机制
- **输出层**: 5层MLP，输出[0,1]范围的奖励值

### MCTS增强
- **最大深度**: 8层
- **每轮迭代**: 500次
- **扩展策略**: 10个子节点
- **混合权重**: α=0.7 (性能权重)

## 注意事项

1. **模型依赖**: 训练好的模型文件需要完整的权重目录结构
2. **计算资源**: 训练过程需要一定的GPU/CPU资源
3. **数据质量**: 生成数据的质量直接影响系统性能
4. **超参数调优**: 根据具体任务可能需要调整超参数

## 扩展建议

1. **更多函数支持**: 扩展支持的数学函数库
2. **多目标优化**: 支持多输出符号回归
3. **约束处理**: 添加数学表达式约束
4. **可解释性**: 增加模型解释和可视化功能

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！