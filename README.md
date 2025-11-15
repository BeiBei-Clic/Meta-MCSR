# Meta-MCSR: 基于自学习奖励网络的符号回归系统

这是一个基于蒙特卡洛树搜索(MCTS)和自学习奖励网络的符号回归系统，通过表达式编码器和奖励网络的协同工作，实现更准确的奖励信号和更高效的符号回归搜索。

## 快速开始

Meta-MCSR系统包含4个主要阶段：

1. **表达式编码器预训练** - 学习数学表达式结构特征（一次性）
2. **奖励网络训练** - 训练奖励网络并微调编码器（一次性）  
3. **直接运行符号回归** - 使用预训练模型进行推理（可重复使用）
4. **测试编码器功能** - 验证和演示编码器功能

## 系统流程

```
步骤1: 预训练表达式编码器           ← 一次性
    ↓
步骤2: 训练奖励网络                 ← 一次性
    ↓  
步骤3: 推理阶段                    ← 可重复使用
    ├── 运行完整系统进行符号回归
    └── 测试编码器功能
```

---

## 1. 表达式编码器预训练

### 1.1 训练表达式编码器

**命令：**
```bash
python expression_encoder_training.py
```

**功能：**
- 生成50,000个随机数学表达式
- 使用对比学习训练Transformer模型
- 保存预训练模型到 `weights/expression_encoder/`

**输出文件：**
```
weights/expression_encoder/
├── model.pth              # 预训练模型权重
├── tokenizer.pkl          # 分词器配置
└── config.json            # 模型配置
```

**训练时间：** 约30分钟（GPU）或2小时（CPU）

**参数说明：**
此脚本使用固定参数，无需手动设置。核心参数包括：
- 嵌入维度：256维
- Transformer层数：6层
- 批次大小：64
- 训练轮数：30

**重要提示：** 这是预训练阶段，只执行一次，后续推理阶段会直接使用生成的模型文件。

---

## 2. 奖励网络训练

### 2.1 训练奖励网络

**命令：**
```bash
python reward_network_training.py
```

**功能：**
- 生成多个符号回归数据集
- 使用MCTS生成训练经验
- 训练奖励网络并微调表达式编码器
- 保存最终模型到 `weights/reward_network_final/`

**训练时间：** 约1小时（GPU）或3小时（CPU）

**输出文件：**
```
weights/reward_network_final/
├── expr_encoder_model.pth           # 微调后的表达式编码器权重
├── reward_network.pth               # 训练好的奖励网络权重
├── tokenizer.pkl                    # 分词器配置
└── training_history.pkl             # 训练历史记录
```

**参数说明：**
此脚本也使用固定参数，包括：
- 奖励网络学习率：1e-4
- 表达式编码器微调学习率：1e-5
- 批次大小：16
- 经验池大小：500
- 混合权重：70%性能奖励 + 30%结构奖励

**重要提示：**
- 必须先完成步骤1（表达式编码器预训练）
- 这是预训练阶段，只执行一次
- 训练过程会自动微调已预训练的表达式编码器

---

## 3. 直接运行符号回归（推理阶段）

### 3.1 运行完整系统

**命令：**
```bash
python mcts_with_reward_network.py
```

**功能：**
- 自动检查并加载预训练模型
- 在预定义的符号回归问题上运行
- 显示搜索结果和性能对比
- 使用混合奖励（性能奖励 + 结构奖励）

**推理流程：**
1. **检查模型文件** - 如果存在预训练模型就加载，否则警告
2. **生成测试数据** - 使用预定义的线性函数问题
3. **运行MCTS搜索** - 使用训练好的奖励网络指导搜索
4. **显示结果** - 找到的表达式、R²分数、运行时间等

**输出示例：**
```
基于自学习奖励网络的MCTS符号回归系统
============================================================
加载表达式嵌入器: weights/expression_encoder
加载奖励网络: weights/reward_network_final

问题: 简单线性函数
描述: y = x1 + 2*x2 + 噪声
真实解: x1 + 2*x2
============================================================

使用增强MCTS（奖励网络）
------------------------------
找到的解: x1 + 2*x2
R2: 0.9985
RMSE: 0.0324
训练时间: 12.45s
```

**重要特点：**
- **推理模式** - 不进行任何训练，直接使用预训练模型
- **可重复使用** - 每次运行都使用相同的预训练模型
- **自动降级** - 如果没有预训练模型，会自动使用基础MCTS（基于R²分数）

### 3.2 高级使用

如果您想运行不同的符号回归问题，可以修改 `mcts_with_reward_network.py` 中的问题定义部分，或者创建一个自定义脚本加载预训练模型：

```python
from mcts_with_reward_network import load_trained_models, run_symbolic_regression
import numpy as np

# 加载预训练模型
models = load_trained_models()

# 自定义数据
X = np.random.uniform(-5, 5, (100, 3))
y = X[:, 0] + 2 * X[:, 1] * np.sin(X[:, 0]) + 0.5 * X[:, 2]**2

# 运行符号回归
result = run_symbolic_regression(
    X, y, 
    true_expression="x1 + 2*x2*sin(x1) + 0.5*x3^2",
    models=models
)

print(f"找到的解: {result['best_expression']}")
print(f"R²分数: {result['r2_score']:.4f}")
```

---

## 4. 测试编码器功能

### 4.1 编码器推理工具

**命令：**
```bash
python expression_encoder_inference.py [模式] [参数...]
```

### 4.2 各种使用模式

#### 演示模式
```bash
python expression_encoder_inference.py demo
```
**功能：** 展示编码器的基本功能，包括编码、相似度计算等

#### 交互模式  
```bash
python expression_encoder_inference.py interactive
```
**功能：** 进入交互式对话，可以输入表达式进行分析

#### 编码单个表达式
```bash
python expression_encoder_inference.py encode "sin(x1) + cos(x2)"
python expression_encoder_inference.py encode "x1^2 + 2*x2*sin(x1)"
```
**功能：** 将指定表达式编码为向量表示

#### 计算相似度
```bash
python expression_encoder_inference.py similarity "sin(x1)" "cos(x1)"
python expression_encoder_inference.py similarity "x1 + x2" "x1 - x2"
```
**功能：** 计算两个表达式的余弦相似度（0-1之间，1表示完全相似）

#### 查找相似表达式
```bash
python expression_encoder_inference.py similar "x1 + x2"
python expression_encoder_inference.py similar "sin(x1)"
```
**功能：** 在内置的表达式库中查找与输入表达式最相似的表达式

#### 分析表达式特征
```bash
python expression_encoder_inference.py analyze "x1^2 + 2*x2*sin(x1)"
python expression_encoder_inference.py analyze "log(x1) + exp(x2)"
```
**功能：** 分析表达式的复杂度、函数类型、变量数量等特征

### 4.3 使用示例

```bash
# 演示基本功能
python expression_encoder_inference.py demo

# 编码一个复杂表达式
python expression_encoder_inference.py encode "sin(x1) + cos(x2) + sqrt(x3)"

# 计算相似度
python expression_encoder_inference.py similarity "x1 + x2" "2*x1 + 2*x2"

# 在内置库中找相似表达式
python expression_encoder_inference.py similar "x1^2 + x2"

# 分析表达式
python expression_encoder_inference.py analyze "log(x1) * sin(x2) + exp(x3)"
```

---

## 完整使用流程

### 第一次使用（完整训练）

```bash
# 1. 预训练表达式编码器（必须先做）
python expression_encoder_training.py

# 2. 训练奖励网络（含编码器微调）
python reward_network_training.py

# 3. 验证编码器功能
python expression_encoder_inference.py demo

# 4. 运行完整系统进行符号回归
python mcts_with_reward_network.py
```

### 日常使用（直接推理）

```bash
# 每次直接运行符号回归（使用已训练的模型）
python mcts_with_reward_network.py

# 或者测试编码器功能
python expression_encoder_inference.py demo
python expression_encoder_inference.py encode "x1 + x2^2"
```

### 重要说明

- **步骤1和2只需要执行一次** - 生成预训练模型
- **步骤3和4可以重复使用** - 直接使用预训练模型进行推理
- **模型持久化** - 训练好的模型保存在 `weights/` 目录，重启后仍然可用

---

## 常见问题

### Q: 运行时提示"找不到预训练模型"
**A:** 请先运行：
```bash
python expression_encoder_training.py
python reward_network_training.py
```

### Q: 训练速度太慢
**A:** 
- 确保使用GPU：检查 `nvidia-smi` 是否可用
- 减少参数：修改脚本中的批次大小或模型维度
- 使用较少的数据：降低生成的数据量

### Q: 搜索结果不理想
**A:** 尝试以下方法：
- 调整MCTS参数（修改 `mcts_with_reward_network.py` 中的参数）
- 增加搜索迭代次数
- 调整混合权重参数

### Q: 内存不足
**A:**
- 减小批次大小
- 使用CPU模式运行（取消CUDA设置）
- 关闭其他程序释放内存

### Q: 想自定义符号回归问题
**A:** 修改 `mcts_with_reward_network.py` 中的问题定义，或使用Python API：

```python
from mcts_with_reward_network import load_trained_models, run_symbolic_regression
import numpy as np

# 加载模型
models = load_trained_models()

# 自定义问题
X = np.random.uniform(-5, 5, (100, 2))
y = X[:, 0]**2 + np.sin(X[:, 1])

# 运行
result = run_symbolic_regression(X, y, models=models)
```

---

## 性能说明

- **预训练时间**：总计约1.5小时（GPU）或5小时（CPU）
- **推理速度**：单个表达式编码 < 0.1秒
- **搜索速度**：通常10-100秒完成一个符号回归问题
- **内存需求**：训练时2-4GB，运行时512MB-1GB

## 系统要求

- **Python**: 3.8+  
- **内存**: 最少4GB，推荐8GB
- **存储**: 2GB可用空间
- **GPU**: 可选，推荐用于加速训练

## 架构说明

这个系统采用**分离式训练架构**：

1. **预训练阶段**：
   - 表达式编码器学习通用的表达式结构表示
   - 奖励网络学习预测表达式性能的映射
   - 训练完成后模型保存到磁盘

2. **推理阶段**：
   - 直接加载预训练模型，不进行训练
   - 使用训练好的模型进行符号回归
   - 支持重复使用和批量处理

这种设计的优势：
- 一次训练，多次使用
- 推理速度快
- 模型可移植和复用
- 便于模型升级和替换

## 许可证

本项目采用MIT许可证