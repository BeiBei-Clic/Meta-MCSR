# Meta-MCSR: 基于自学习奖励网络的符号回归系统

这是一个基于蒙特卡洛树搜索(MCTS)和自学习奖励网络的符号回归系统，通过表达式编码器和奖励网络的协同工作，实现更准确的奖励信号和更高效的符号回归搜索。

## 快速开始

Meta-MCSR系统包含4个主要训练和使用阶段：

1. **表达式编码器预训练** - 学习数学表达式结构特征
2. **表达式编码器推理** - 使用训练好的编码器
3. **奖励网络训练** - 训练奖励网络和微调编码器  
4. **完整系统运行** - 运行融合的MCTS符号回归

## 系统流程

```
步骤1: 预训练表达式编码器
    ↓
步骤2: 训练奖励网络（含编码器微调）
    ↓  
步骤3: 使用完整系统进行符号回归
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

---

## 2. 表达式编码器推理（预测）

### 2.1 基础用法

**命令：**
```bash
python expression_encoder_inference.py [模式] [参数...]
```

### 2.2 各种使用模式

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

### 2.3 使用示例

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

## 3. 奖励网络训练

### 3.1 训练奖励网络

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
- 训练过程会自动微调已预训练的表达式编码器

---

## 4. 完整系统运行符号回归

### 4.1 运行完整系统

**命令：**
```bash
python mcts_with_reward_network.py [模式选项]
```

### 4.2 各种运行模式

#### 交互模式
```bash
python mcts_with_reward_network.py --mode interactive
```
**功能：** 进入交互模式，可以自定义数据集和参数

#### 基准测试模式
```bash
python mcts_with_reward_network.py --mode benchmark
```
**功能：** 运行预定义的基准测试集，评估系统性能

#### 单次运行模式
```bash
python mcts_with_reward_network.py --mode single --problem 1
python mcts_with_reward_network.py --mode single --problem 2
```
**功能：** 运行单个预定义问题

#### 自定义数据模式
```bash
python mcts_with_reward_network.py --mode custom --data custom_data.csv --output result.txt
```
**功能：** 使用自定义数据文件进行符号回归

### 4.3 高级参数选项

#### 搜索深度设置
```bash
python mcts_with_reward_network.py --mode single --problem 1 --depth 10
```
**说明：** 设置最大搜索深度为10层（默认8层）

#### 迭代次数设置
```bash
python mcts_with_reward_network.py --mode single --problem 1 --iterations 1000
```
**说明：** 设置MCTS迭代次数为1000次（默认500次）

#### 混合权重调整
```bash
python mcts_with_reward_network.py --mode single --problem 1 --alpha 0.8
```
**说明：** 设置性能奖励权重为0.8（默认0.7，范围0-1）

#### 变量数量限制
```bash
python mcts_with_reward_network.py --mode single --problem 1 --max-vars 5
```
**说明：** 限制表达式最多使用5个变量（默认3个）

#### 真实表达式设置（可选）
```bash
python mcts_with_reward_network.py --mode single --problem 1 --oracle "x1 + 2*x2*sin(x1)"
```
**说明：** 提供真实表达式作为参考，提升搜索精度

### 4.4 组合使用示例

```bash
# 使用默认参数运行基准测试
python mcts_with_reward_network.py --mode benchmark

# 自定义参数运行单个问题
python mcts_with_reward_network.py --mode single --problem 2 --depth 12 --iterations 1000 --alpha 0.8

# 使用真实表达式提升精度
python mcts_with_reward_network.py --mode single --problem 3 --oracle "x1^2 + sin(x2)"

# 交互模式运行
python mcts_with_reward_network.py --mode interactive
```

### 4.5 输出说明

系统运行后会输出：
- **找到的最佳表达式**
- **R²分数**（拟合质量，1.0为完美拟合）
- **搜索迭代次数**
- **搜索过程**（可选）
- **运行时间**

### 4.6 交互模式使用方法

运行 `python mcts_with_reward_network.py --mode interactive` 后：

1. **输入数据维度**：例如输入 `3` 表示有3个变量
2. **输入数据范围**：例如输入 `-5,5` 表示变量范围在-5到5之间
3. **生成数据点数量**：例如输入 `100` 表示生成100个数据点
4. **真实表达式**：（可选）输入真实表达式，或按回车跳过
5. **搜索参数**：按提示设置深度、迭代次数等
6. **开始搜索**：系统开始MCTS搜索并实时显示进度

---

## 完整使用流程

### 第一次使用

```bash
# 1. 预训练表达式编码器（必须先做）
python expression_encoder_training.py

# 2. 训练奖励网络（含编码器微调）
python reward_network_training.py

# 3. 运行完整系统
python mcts_with_reward_network.py --mode benchmark
```

### 日常使用

```bash
# 使用训练好的模型进行符号回归
python mcts_with_reward_network.py --mode single --problem 1

# 或者使用交互模式
python mcts_with_reward_network.py --mode interactive
```

### 测试编码器功能

```bash
# 演示编码器功能
python expression_encoder_inference.py demo

# 编码特定表达式
python expression_encoder_inference.py encode "sin(x1) + cos(x2)"

# 计算相似度
python expression_encoder_inference.py similarity "x1 + x2" "2*x1 + 2*x2"
```

---

## 常见问题

### Q: 运行时提示"找不到预训练模型"
**A:** 请先运行 `python expression_encoder_training.py` 训练编码器

### Q: 训练速度太慢
**A:** 
- 确保使用GPU：检查 `nvidia-smi` 是否可用
- 减少参数：修改脚本中的批次大小或模型维度
- 使用较少的数据：降低生成的数据量

### Q: 搜索结果不理想
**A:**
- 增加迭代次数：`--iterations 1000`
- 增加搜索深度：`--depth 10`
- 调整混合权重：`--alpha 0.8`
- 提供真实表达式：`--oracle "真实表达式"`

### Q: 内存不足
**A:**
- 减小批次大小
- 使用CPU模式运行（取消CUDA设置）
- 关闭其他程序释放内存

### Q: 想测试不同参数组合
**A:** 使用交互模式 `python mcts_with_reward_network.py --mode interactive`，可以实时调整参数

---

## 性能说明

- **训练时间**：总计约1.5小时（GPU）或5小时（CPU）
- **推理速度**：单个表达式编码 < 0.1秒
- **搜索速度**：通常10-100秒完成一个符号回归问题
- **内存需求**：训练时2-4GB，运行时512MB-1GB

## 系统要求

- **Python**: 3.8+  
- **内存**: 最少4GB，推荐8GB
- **存储**: 2GB可用空间
- **GPU**: 可选，推荐用于加速训练

## 许可证

本项目采用MIT许可证