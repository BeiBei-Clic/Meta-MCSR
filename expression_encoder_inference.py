import sys
import os
import torch
import numpy as np

# 添加nd2py包路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'nd2py_package'))
import nd2py as nd
from expression_encoder import ExpressionEmbedding


def encode_expression(expression_str, model_path='weights/expression_encoder'):
    """
    编码单个表达式为嵌入向量
    
    Args:
        expression_str: 表达式字符串
        model_path: 模型路径
    
    Returns:
        numpy.ndarray: 嵌入向量
    """
    try:
        # 创建嵌入器
        embedding = ExpressionEmbedding(model_path)
        
        # 编码表达式
        embedding_vector = embedding.encode_single_expression(expression_str)
        
        return embedding_vector
        
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}")
        return None
    except Exception as e:
        print(f"编码表达式时出错: {e}")
        return None


def encode_expressions_batch(expressions, model_path='weights/expression_encoder', batch_size=32):
    """
    批量编码表达式
    
    Args:
        expressions: 表达式字符串列表
        model_path: 模型路径
        batch_size: 批处理大小
    
    Returns:
        numpy.ndarray: 嵌入向量矩阵 (n_expressions, embedding_dim)
    """
    try:
        # 创建嵌入器
        embedding = ExpressionEmbedding(model_path)
        
        # 编码所有表达式
        embeddings = embedding.encode_expressions(expressions, batch_size)
        
        return embeddings
        
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}")
        return None
    except Exception as e:
        print(f"批量编码表达式时出错: {e}")
        return None


def compute_similarity(expr1, expr2, model_path='weights/expression_encoder'):
    """
    计算两个表达式的相似度
    
    Args:
        expr1: 第一个表达式字符串
        expr2: 第二个表达式字符串
        model_path: 模型路径
    
    Returns:
        float: 相似度分数（余弦相似度）
    """
    try:
        # 编码两个表达式
        emb1 = encode_expression(expr1, model_path)
        emb2 = encode_expression(expr2, model_path)
        
        if emb1 is None or emb2 is None:
            return None
        
        # 计算余弦相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return similarity
        
    except Exception as e:
        print(f"计算相似度时出错: {e}")
        return None


def find_similar_expressions(target_expr, candidate_expressions, 
                           model_path='weights/expression_encoder', top_k=5):
    """
    找到与目标表达式最相似的候选表达式
    
    Args:
        target_expr: 目标表达式字符串
        candidate_expressions: 候选表达式列表
        model_path: 模型路径
        top_k: 返回前k个最相似的表达式
    
    Returns:
        list: (表达式, 相似度) 元组列表
    """
    try:
        # 编码目标表达式
        target_embedding = encode_expression(target_expr, model_path)
        if target_embedding is None:
            return None
        
        # 批量编码候选表达式
        candidate_embeddings = encode_expressions_batch(
            candidate_expressions, model_path, batch_size=len(candidate_expressions)
        )
        if candidate_embeddings is None:
            return None
        
        # 计算相似度
        similarities = []
        for i, candidate_emb in enumerate(candidate_embeddings):
            similarity = np.dot(target_embedding, candidate_emb) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(candidate_emb)
            )
            similarities.append((candidate_expressions[i], similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
        
    except Exception as e:
        print(f"寻找相似表达式时出错: {e}")
        return None


def analyze_expression_complexity(expression_str, model_path='weights/expression_encoder'):
    """
    分析表达式的复杂度特征
    
    Args:
        expression_str: 表达式字符串
        model_path: 模型路径
    
    Returns:
        dict: 复杂度分析结果
    """
    try:
        embedding_vector = encode_expression(expression_str, model_path)
        if embedding_vector is None:
            return None
        
        # 基本的复杂度指标
        complexity_metrics = {
            'expression_length': len(expression_str),
            'variable_count': expression_str.count('x'),
            'function_count': sum(1 for func in ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt'] 
                                if func in expression_str),
            'operator_count': sum(1 for op in ['+', '-', '*', '/'] 
                                if expression_str.count(op) > 0),
            'embedding_norm': np.linalg.norm(embedding_vector),
            'embedding_variance': np.var(embedding_vector)
        }
        
        return complexity_metrics
        
    except Exception as e:
        print(f"分析表达式复杂度时出错: {e}")
        return None


def demonstrate_embedding_capabilities():
    """演示嵌入器的功能"""
    print("表达式嵌入器预测演示")
    print("=" * 50)
    
    # 检查模型是否存在
    model_path = 'weights/expression_encoder'
    if not os.path.exists(model_path + '_tokenizer.pkl'):
        print("错误：未找到预训练的表达式嵌入器模型")
        print("请先运行 expression_encoder_training.py 进行预训练")
        return
    
    # 测试表达式
    test_expressions = [
        "x1 + x2",
        "x1 - x2", 
        "2*x1 + 3*x2",
        "sin(x1)",
        "cos(x1) + sin(x2)",
        "sqrt(x1)",
        "log(x1 + x2)",
        "exp(x1)",
        "x1*x2 + x1*x2",  # 重复表达式
        "sin(x1)*cos(x1)"  # 三角函数组合
    ]
    
    print("测试表达式列表:")
    for i, expr in enumerate(test_expressions):
        print(f"  {i+1:2d}. {expr}")
    
    print("\n" + "-" * 50)
    
    # 1. 编码单个表达式
    print("\n1. 编码单个表达式示例:")
    target_expr = "sin(x1) + cos(x1)"
    embedding = encode_expression(target_expr, model_path)
    if embedding is not None:
        print(f"表达式: {target_expr}")
        print(f"嵌入向量形状: {embedding.shape}")
        print(f"嵌入向量 (前10维): {embedding[:10]}")
    else:
        print("编码失败")
    
    # 2. 批量编码
    print("\n2. 批量编码示例:")
    batch_embeddings = encode_expressions_batch(test_expressions[:5], model_path)
    if batch_embeddings is not None:
        print(f"编码了 {len(test_expressions[:5])} 个表达式")
        print(f"嵌入矩阵形状: {batch_embeddings.shape}")
    else:
        print("批量编码失败")
    
    # 3. 相似度计算
    print("\n3. 表达式相似度计算:")
    expr1 = "sin(x1)"
    expr2 = "cos(x1)"
    similarity = compute_similarity(expr1, expr2, model_path)
    if similarity is not None:
        print(f"'{expr1}' 与 '{expr2}' 的相似度: {similarity:.4f}")
    else:
        print("相似度计算失败")
    
    # 4. 寻找相似表达式
    print("\n4. 寻找相似表达式:")
    target_expr = "sin(x1)"
    similar_exprs = find_similar_expressions(target_expr, test_expressions, model_path, top_k=3)
    if similar_exprs is not None:
        print(f"与 '{target_expr}' 最相似的表达式:")
        for expr, sim in similar_exprs:
            print(f"  {expr:<20} (相似度: {sim:.4f})")
    else:
        print("寻找相似表达式失败")
    
    # 5. 复杂度分析
    print("\n5. 表达式复杂度分析:")
    test_expr = "sin(x1) * cos(x1) + log(x2) * sqrt(x3)"
    complexity = analyze_expression_complexity(test_expr, model_path)
    if complexity is not None:
        print(f"表达式: {test_expr}")
        for metric, value in complexity.items():
            print(f"  {metric:<20}: {value}")
    else:
        print("复杂度分析失败")
    
    print("\n" + "=" * 50)
    print("演示完成！")


def interactive_mode():
    """交互模式"""
    print("表达式嵌入器交互模式")
    print("=" * 50)
    
    model_path = 'weights/expression_encoder'
    if not os.path.exists(model_path + '_tokenizer.pkl'):
        print("错误：未找到预训练的表达式嵌入器模型")
        print("请先运行 expression_encoder_training.py 进行预训练")
        return
    
    embedding = ExpressionEmbedding(model_path)
    
    while True:
        print("\n可用命令:")
        print("1. encode <expression> - 编码表达式")
        print("2. similarity <expr1> <expr2> - 计算相似度")
        print("3. similar <expression> - 找相似表达式")
        print("4. analyze <expression> - 分析复杂度")
        print("5. help - 显示帮助")
        print("6. quit - 退出")
        
        command = input("\n请输入命令: ").strip()
        
        if not command:
            continue
            
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        
        if cmd == 'quit':
            break
        elif cmd == 'help':
            continue
        elif cmd == 'encode':
            if len(parts) < 2:
                print("用法: encode <expression>")
                continue
            expr = parts[1]
            result = encode_expression(expr, model_path)
            if result is not None:
                print(f"表达式: {expr}")
                print(f"嵌入向量形状: {result.shape}")
                print(f"嵌入向量 (前10维): {result[:10]}")
            else:
                print("编码失败")
                
        elif cmd == 'similarity':
            if len(parts) < 2:
                print("用法: similarity <expr1> <expr2>")
                continue
            exprs = parts[1].split(maxsplit=1)
            if len(exprs) < 2:
                print("请提供两个表达式")
                continue
            result = compute_similarity(exprs[0], exprs[1], model_path)
            if result is not None:
                print(f"'{exprs[0]}' 与 '{exprs[1]}' 的相似度: {result:.4f}")
            else:
                print("相似度计算失败")
                
        elif cmd == 'similar':
            if len(parts) < 2:
                print("用法: similar <expression>")
                continue
            expr = parts[1]
            print("输入候选表达式（每行一个，空行结束）:")
            candidates = []
            while True:
                candidate = input().strip()
                if not candidate:
                    break
                candidates.append(candidate)
            
            if candidates:
                result = find_similar_expressions(expr, candidates, model_path, top_k=len(candidates))
                if result is not None:
                    print(f"与 '{expr}' 的相似度排序:")
                    for similar_expr, sim in result:
                        print(f"  {similar_expr:<30} (相似度: {sim:.4f})")
                else:
                    print("寻找相似表达式失败")
            else:
                print("未提供候选表达式")
                
        elif cmd == 'analyze':
            if len(parts) < 2:
                print("用法: analyze <expression>")
                continue
            expr = parts[1]
            result = analyze_expression_complexity(expr, model_path)
            if result is not None:
                print(f"表达式: {expr}")
                for metric, value in result.items():
                    print(f"  {metric:<20}: {value}")
            else:
                print("复杂度分析失败")
        else:
            print("未知命令，请输入 'help' 查看可用命令")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'demo':
            demonstrate_embedding_capabilities()
        elif command == 'interactive':
            interactive_mode()
        elif command == 'encode':
            if len(sys.argv) < 3:
                print("用法: python expression_encoder_inference.py encode <expression>")
                sys.exit(1)
            expression = sys.argv[2]
            result = encode_expression(expression)
            if result is not None:
                print(f"嵌入向量: {result}")
            else:
                print("编码失败")
        elif command == 'similarity':
            if len(sys.argv) < 4:
                print("用法: python expression_encoder_inference.py similarity <expr1> <expr2>")
                sys.exit(1)
            expr1, expr2 = sys.argv[2], sys.argv[3]
            result = compute_similarity(expr1, expr2)
            if result is not None:
                print(f"相似度: {result:.4f}")
            else:
                print("计算失败")
        else:
            print("可用命令: demo, interactive, encode, similarity")
    else:
        # 默认运行演示
        demonstrate_embedding_capabilities()


if __name__ == "__main__":
    main()