#!/bin/bash

# 基于自学习"裁判"网络的MCTS符号回归系统快速启动脚本

echo "基于自学习奖励网络的MCTS符号回归系统"
echo "======================================"
echo

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "错误：未找到uv包管理器"
    echo "请先安装uv: https://docs.astral.sh/uv/"
    exit 1
fi

echo "1. 检查环境..."
# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    uv venv
fi

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
echo "安装依赖..."
uv sync

echo
echo "2. 训练系统组件..."
echo

# 检查是否已有训练好的模型
if [ ! -f "weights/expression_encoder_tokenizer.pkl" ]; then
    echo "步骤1: 预训练表达式嵌入器"
    echo "这可能需要一些时间，请耐心等待..."
    uv run python expression_encoder_training.py
    echo
else
    echo "跳过表达式嵌入器训练（模型已存在）"
fi

if [ ! -f "weights/reward_network_final_reward_network.pth" ]; then
    echo "步骤2: 训练奖励网络"
    echo "这可能需要一些时间，请耐心等待..."
    uv run python reward_network_training.py
    echo
else
    echo "跳过奖励网络训练（模型已存在）"
fi

echo "3. 系统准备就绪！"
echo

echo "可用命令："
echo "- 交互模式:           uv run python mcts_with_reward_network.py --mode interactive"
echo "- 基准测试:           uv run python mcts_with_reward_network.py --mode benchmark"
echo "- 单次运行 (问题1):    uv run python mcts_with_reward_network.py --mode single --problem 1"
echo
echo "- 表达式嵌入器演示:    uv run python expression_encoder_inference.py demo"
echo "- 表达式嵌入器交互:    uv run python expression_encoder_inference.py interactive"
echo
echo "- 查看帮助:           uv run python mcts_with_reward_network.py --help"
echo

read -p "是否现在运行演示？(y/n): " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo
    echo "运行表达式嵌入器演示..."
    uv run python expression_encoder_inference.py demo
    echo
    echo "现在可以运行完整的系统了！"
    echo "建议先尝试: uv run python mcts_with_reward_network.py --mode interactive"
else
    echo "系统已准备就绪，随时可以使用！"
fi