#!/usr/bin/env python3
"""
SNIP相似度计算器
基于SNIP项目计算数学表达式和数据点对的潜向量相似度
"""

import sys
import os

# 添加源代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入SNIP相似度计算器
from snip_similarity_calculator import SNIPSimmilarityCalculator

def main():
    """主函数"""
    base_dir = "/home/xyh/Meta-MCSR/similarity_project"

    # 初始化计算器
    model_path = "/home/xyh/Meta-MCSR/weights/snip-1d-normalized.pth"
    calculator = SNIPSimmilarityCalculator(model_path)

    # 运行分析
    calculator.run_analysis(base_dir)


if __name__ == "__main__":
    main()