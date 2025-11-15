# 自适应裁判网络与MCTS集成框架

这是一个基于蒙特卡洛树搜索(MCTS)和自适应裁判网络的符号回归系统。系统通过表达式编码器和奖励网络的协同工作，实现更准确的奖励信号和更高效的搜索。

## 系统架构

### 核心组件

1. **表达式编码器 (Expression Encoder)** - "资深结构学家"
   - 基于Transformer的神经网络，将数学表达式编码为向量表示
   - 预训练：使用对比学习（三元组损失）在大规模表达式数据集上学习结构特征
   - 在线微调：在MCTS运行过程中以低学习率持续优化

2. **奖励网络 (Reward Network)** - "潜力分析师"
   - 接收表达式嵌入向量和数据特征，预测表达式的最终性能潜力
   - 完全在线学习，使用MCTS生成的经验数据
   - 包含数据编码器、融合模块和输出层

3. **增强的MCTS引擎** - "探索者"
   - 集成奖励网络指导搜索过程
   - 双网络协作奖励计算：结构相似度 + 性能潜力预测

4. **经验回放池**
   - 存储MCTS探索的表达式和性能数据
   - 用于训练奖励网络和微调表达式编码器

## 文件结构

- `expression_encoder.py` - 表达式编码器核心实现
- `expression_encoder_training.py` - 表达式编码器预训练（对比学习）
- `expression_encoder_inference.py` - 表达式编码器预测和测试
- `reward_network.py` - 奖励网络核心实现
- `reward_network_training.py` - 奖励网络训练（含表达式编码器微调）
- `mcts_enhanced.py` - 增强的MCTS实现
- `mcts_with_reward_network.py` - 融合系统主程序

## 安装和配置

### 环境准备

```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
uv sync
```

### 目录结构

```
weights/
├── expression_encoder/          # 预训练的表达式编码器
└── reward_network_final/        # 训练好的奖励网络
```

## 使用流程

### 步骤1：预训练表达式编码器

使用对比学习（三元组损失）预训练表达式编码器：

```bash
python expression_encoder_training.py
```

**功能说明：**
- 生成50,000个随机数学表达式
- 构建三元组（anchor, positive, negative）
- 使用三元组损失训练Transformer模型
- 保存模型到 `weights/expression_encoder/`

**训练参数：**
- 嵌入维度：256
- Transformer层数：6
- 注意力头数：8
- 批次大小：64
- 学习率：1e-4
- 训练轮数：30

### 步骤2：训练奖励网络（含表达式编码器微调）

训练奖励网络，同时微调表达式编码器：

```bash
python reward_network_training.py
```

**功能说明：**
- 加载预训练的表达式编码器
- 生成多个符号回归数据集
- 使用MCTS生成训练经验
- **分离式训练：**
  - 阶段1：训练奖励网络（正常学习率 1e-4）
  - 阶段2：微调表达式编码器（低学习率 1e-5）
- 保存模型到 `weights/reward_network_final/`

**训练参数：**
- 奖励网络学习率：1e-4
- 表达式编码器学习率：1e-5
- 批次大小：16
- 经验池大小：500
- MCTS迭代次数：10

### 步骤3：运行融合MCTS符号回归

使用训练好的模型进行符号回归：

```bash
python mcts_with_reward_network.py
```

**功能说明：**
- 加载表达式编码器和奖励网络
- 运行增强的MCTS搜索
- 双网络协作计算奖励：
  - 结构奖励：表达式与真实解的嵌入相似度
  - 性能奖励：奖励网络预测的潜力
  - 混合奖励：`β * R_pot + (1-β) * S_struct`

### 步骤4：表达式编码器单独使用

测试表达式编码器的功能：

```bash
# 运行演示
python expression_encoder_inference.py demo

# 交互模式
python expression_encoder_inference.py interactive

# 编码单个表达式
python expression_encoder_inference.py encode "sin(x1) + cos(x2)"

# 计算相似度
python expression_encoder_inference.py similarity "sin(x1)" "cos(x1)"
```

**可用命令：**
- `encode <expression>` - 编码表达式为向量
- `similarity <expr1> <expr2>` - 计算两个表达式的余弦相似度
- `similar <expression>` - 在候选表达式中找最相似的
- `analyze <expression>` - 分析表达式的复杂度特征

## 核心特性

### 1. 对比学习预训练
- 使用三元组损失学习表达式结构特征
- Anchor：原始表达式
- Positive：相似/简化表达式
- Negative：不同结构的表达式
- 学习目标：`max(0, d(a,p) - d(a,n) + margin)`

### 2. 分离式在线微调
- **奖励网络**：正常学习率（1e-4），快速适应新数据
- **表达式编码器**：低学习率（1e-5），稳定微调避免灾难性遗忘
- 经验回放池确保训练稳定性

### 3. 双网络协作奖励
在MCTS模拟阶段，同时使用两个网络：

```python
# 结构评估
expr_embedding = Encoder_expr(current_expression)
struct_reward = cosine_similarity(expr_embedding, true_solution_embedding)

# 潜力预测
potential_reward = R_phi(expr_embedding, data_features)

# 最终奖励
final_reward = beta * potential_reward + (1-beta) * struct_reward
```

### 4. 自学习闭环
```
预训练 → MCTS探索 → 经验生成 → 网络更新 → 更强指导 → 更高质量经验 → ...
```

## 技术参数

### 表达式编码器
- **架构**：6层Transformer编码器
- **嵌入维度**：256
- **注意力头数**：8
- **词汇表大小**：动态构建
- **最大序列长度**：128

### 奖励网络
- **表达式编码器**：复用预训练模型
- **数据编码器**：3层MLP（128→64→256）
- **融合模块**：交叉注意力机制
- **输出层**：5层MLP，输出[0,1]范围的奖励值

### MCTS超参数
- **最大深度**：8层
- **每轮迭代**：500次
- **扩展策略**：10个子节点
- **混合权重**：β=0.7（性能权重）
- **UCT常数**：c=1.4

## 示例

### 基本用法

```python
import numpy as np
from mcts_with_reward_network import run_symbolic_regression

# 生成数据
np.random.seed(42)
X = np.random.uniform(-5, 5, (100, 3))
y = X[:, 0] + 2 * X[:, 1] * np.sin(X[:, 0]) + 0.5 * X[:, 2]**2

# 运行符号回归
result = run_symbolic_regression(
    X, y,
    true_expression="x1 + 2*x2*sin(x1) + 0.5*x3^2",
    models={
        'expr_encoder': 'weights/expression_encoder',
        'reward_network': 'weights/reward_network_final'
    }
)

print(f"找到的解: {result['best_expression']}")
print(f"R2分数: {result['r2_score']:.4f}")
```

### 自定义MCTS参数

```python
from mcts_enhanced import MCTSWithRewardNetwork
from reward_network import RewardNetwork

# 创建奖励网络
reward_network = RewardNetwork(
    expr_encoder_path='weights/expression_encoder',
    fusion_type='attention'
)
reward_network.set_data_encoder_dim(X.shape[1])

# 创建增强MCTS
mcts = MCTSWithRewardNetwork(
    max_depth=10,
    max_iterations=1000,
    max_vars=5,
    reward_network=reward_network,
    alpha_hybrid=0.7  # 性能权重
)

# 设置神谕目标（可选）
mcts.set_oracle_target("x1 + 2*x2")

# 训练
best_expr = mcts.fit(X, y)
```

## 性能对比

相比传统基于R²分数的MCTS，本系统的优势：

1. **更准确的奖励信号**
   - 奖励网络学习评估表达式潜力
   - 结构相似度提供几何指导
   - 混合奖励平衡性能和结构

2. **结构感知**
   - 不仅考虑拟合性能
   - 学习表达式结构特征
   - 向真实解结构靠拢

3. **自学习能力**
   - 经验累积持续改进
   - 在线微调适应任务
   - 探索效率不断提升

4. **启动稳定**
   - 预训练编码器提供良好初始
   - 低学习率微调避免灾难性遗忘
   - 分离式训练保证稳定性

## 注意事项

1. **模型依赖**：必须先运行步骤1预训练表达式编码器
2. **计算资源**：训练过程需要GPU加速（可选）
3. **数据质量**：生成数据的质量影响系统性能
4. **超参数调优**：根据具体任务可能需要调整学习率、批次大小等

## 故障排查

### 问题：找不到模型文件
```
错误：未找到预训练的表达式嵌入器模型
```
**解决**：先运行 `python expression_encoder_training.py`

### 问题：CUDA内存不足
```
错误：CUDA out of memory
```
**解决**：减小批次大小（batch_size）或模型维度（d_model）

### 问题：训练损失不下降
```
训练损失保持不变或波动很大
```
**解决**：
- 检查学习率是否合适
- 增加经验池大小
- 减少MCTS迭代次数

## 扩展建议

1. **更多函数支持**：扩展支持的数学函数库
2. **约束处理**：添加数学表达式约束（如单调性、对称性）
3. **多目标优化**：支持多输出符号回归
4. **可视化**：增加训练过程可视化和模型解释
5. **分布式训练**：支持多GPU并行训练

## 许可证

本项目采用MIT许可证

## 贡献

欢迎提交Issue和Pull Request！
