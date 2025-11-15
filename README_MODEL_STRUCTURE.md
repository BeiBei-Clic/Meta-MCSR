# 模型文件结构说明

## 文件夹结构

### `weights/` 文件夹
**用途**: 保存最终训练好的权重参数，用于推理和部署

**内容**:
- `expression_encoder_model.pth` - 表达式编码器最终权重
- `expression_encoder_tokenizer.pkl` - 表达式分词器配置
- `reward_network_final_expr_encoder_model.pth` - 奖励网络表达式编码器最终权重
- `reward_network_final_expr_encoder_tokenizer.pkl` - 奖励网络分词器配置
- `reward_network_final_reward_network.pth` - 奖励网络核心组件最终权重
- `reward_network_final_training_history.pkl` - 最终训练历史记录

### `checkpoints/` 文件夹
**用途**: 保存训练过程中的检查点文件，用于断点续训和模型分析

**子文件夹结构**:
- `checkpoints/expression_encoder/` - 表达式编码器检查点
  - `best_epoch_X` - 验证损失最佳时的模型
  - `epoch_X` - 定期保存的检查点（每10个epoch）
- `checkpoints/reward_network/` - 奖励网络检查点
  - `best_epoch_X` - 训练损失最佳时的模型

## 修改历史

- **2024-11-15**: 重新设计文件结构，将检查点和最终权重分离保存
  - 检查点文件保存在 `checkpoints/` 文件夹
  - `weights/` 文件夹只保留最终训练好的权重参数
  - 更新了 `expression_encoder_training.py` 和 `reward_network_training.py`
  - 更新了 `.gitignore` 文件，忽略 `checkpoints/` 文件夹

## 训练流程

1. **表达式编码器训练** (`expression_encoder_training.py`)
   - 检查点保存到: `checkpoints/expression_encoder/`
   - 最终权重保存到: `weights/expression_encoder`

2. **奖励网络训练** (`reward_network_training.py`)
   - 检查点保存到: `checkpoints/reward_network/`
   - 最终权重保存到: `weights/reward_network_final`

## 注意事项

- `weights/` 文件夹中的文件用于生产环境部署
- `checkpoints/` 文件夹用于研究和调试，可以安全删除
- 两个文件夹都被 `.gitignore` 忽略，不会提交到版本控制