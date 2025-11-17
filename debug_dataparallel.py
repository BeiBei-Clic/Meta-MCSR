#!/usr/bin/env python3
"""
诊断DataParallel是否正常工作
"""

import torch
import torch.nn as nn
import sys
import os

def test_dataparallel():
    """测试DataParallel是否正常工作"""
    
    print("🔍 DataParallel诊断工具")
    print("=" * 50)
    
    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"检测到GPU数量: {gpu_count}")
    
    if gpu_count < 2:
        print("⚠️ 警告：检测到GPU数量 < 2，无法测试DataParallel")
        return
    
    # 创建简单测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    # 测试单GPU vs 多GPU
    print("\n1. 测试单GPU性能...")
    model_single = SimpleModel().cuda(0)
    dummy_input = torch.randn(32, 100).cuda(0)
    
    # 单GPU测试
    with torch.no_grad():
        output_single = model_single(dummy_input)
        print(f"单GPU输出形状: {output_single.shape}")
    
    print("\n2. 测试DataParallel性能...")
    
    # 多GPU测试
    model_multi = SimpleModel().cuda(0)
    model_multi = nn.DataParallel(model_multi, device_ids=list(range(gpu_count)))
    
    print(f"DataParallel设备ID: {model_multi.device_ids}")
    print(f"主设备: {next(model_multi.parameters()).device}")
    
    # 测试GPU内存分布
    print("\n3. 检查GPU内存分布...")
    for i in range(gpu_count):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: 分配 {memory_allocated:.2f}GB, 保留 {memory_reserved:.2f}GB")
    
    # 测试前向传播
    print("\n4. 测试前向传播...")
    
    # 清空内存
    for i in range(gpu_count):
        torch.cuda.empty_cache()
    
    # 重新分配
    dummy_input = torch.randn(32, 100).cuda(0)
    
    with torch.no_grad():
        output_multi = model_multi(dummy_input)
        print(f"多GPU输出形状: {output_multi.shape}")
    
    # 检查GPU内存使用变化
    print("\n5. 检查GPU内存使用变化...")
    for i in range(gpu_count):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i} 使用: {memory_allocated:.2f}GB")
    
    # 检查是否有负载不均衡
    memories = [torch.cuda.memory_allocated(i) / 1024**3 for i in range(gpu_count)]
    max_memory = max(memories)
    min_memory = min(memories)
    
    if max_memory > min_memory * 2:
        print(f"⚠️ 检测到负载不均衡: 最高 {max_memory:.2f}GB vs 最低 {min_memory:.2f}GB")
    else:
        print("✅ GPU负载相对均衡")
    
    print("\n6. 测试批次大小影响...")
    
    # 测试不同批次大小
    for batch_size in [16, 32, 64, 128]:
        try:
            dummy_input = torch.randn(batch_size, 100).cuda(0)
            with torch.no_grad():
                output = model_multi(dummy_input)
            
            print(f"  批次大小 {batch_size}: ✅ 成功")
            
            # 检查GPU内存
            memories = [torch.cuda.memory_allocated(i) / 1024**3 for i in range(gpu_count)]
            avg_memory = sum(memories) / len(memories)
            print(f"    平均GPU内存: {avg_memory:.2f}GB")
            
        except Exception as e:
            print(f"  批次大小 {batch_size}: ❌ 失败 ({e})")
    
    print("\n🎯 诊断完成!")
    print("\n建议:")
    if max_memory < 1.0:
        print("- DataParallel可能工作正常，但批次太小")
    elif max_memory > 8.0:
        print("- GPU内存使用过高，建议降低批次大小")
    else:
        print("- 整体看起来正常")

if __name__ == "__main__":
    test_dataparallel()
