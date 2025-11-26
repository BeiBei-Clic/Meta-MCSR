#!/usr/bin/env python3
"""
SNIP相似度计算器
基于SNIP项目计算数学表达式和数据点对的潜向量相似度
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from collections import defaultdict

# 添加SNIP模块路径
sys.path.append('/home/xyh/Meta-MCSR')

from snip.model import build_modules, check_model_params
from snip.envs import build_env
# 不使用get_parser，手动创建参数


class SNIPSimmilarityCalculator:
    """基于SNIP的相似度计算器"""

    def __init__(self, model_path="weights/snip-1d-normalized.pth"):
        """
        初始化SNIP计算器
        Args:
            model_path: 预训练模型路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化SNIP组件
        self._init_snip(model_path)

    def _init_snip(self, model_path):
        """初始化SNIP组件"""
        try:
            # 创建基础参数
            self.params = self._create_params(model_path)

            # 构建环境和模块
            self.env = build_env(self.params)
            self.modules = build_modules(self.env, self.params)

            # 提取核心组件
            self.embedder = self.modules["embedder"]
            self.encoder_f = self.modules["encoder_f"]  # 符号编码器
            self.encoder_y = self.modules["encoder_y"]  # 数值编码器

            # 设置为评估模式
            self.embedder.eval()
            self.encoder_f.eval()
            self.encoder_y.eval()

            # 移动到设备
            if not self.params.cpu:
                self.embedder.cuda()
                self.encoder_f.cuda()
                self.encoder_y.cuda()

            print("SNIP模型初始化成功")
            self.use_snip = True

        except Exception as e:
            print(f"SNIP初始化失败，使用简化方法: {e}")
            self.use_snip = False

    def _create_params(self, model_path):
        """创建SNIP参数对象"""
        # 手动创建参数对象，避免命令行解析
        class Params:
            def __init__(self):
                # 模型结构参数
                self.max_input_dimension = 10
                self.enc_emb_dim = 512
                self.dec_emb_dim = 512
                self.enc_n_layers = 6
                self.dec_n_layers = 6
                self.enc_n_heads = 8
                self.dec_n_heads = 8
                self.latent_dim = 512

                # 训练参数
                self.batch_size = 32
                self.max_output_length = 128
                self.max_src_len = 128
                self.max_target_len = 128

                # CLIP参数
                self.loss_type = 'CLIP'
                self.clip_temperature = 0.07

                # 位置编码
                self.enc_positional_embeddings = 'sinusoidal'
                self.dec_positional_embeddings = 'sinusoidal'

                # 嵌入层参数
                self.emb_emb_dim = 512
                self.n_emb_layers = 1
                self.emb_expansion_factor = 4

                # 网络参数
                self.n_enc_hidden_layers = 1
                self.n_dec_hidden_layers = 1
                self.norm_attention = True
                self.dropout = 0.1
                self.attention_dropout = 0.1
                self.share_inout_emb = False

                # 其他参数
                self.env_base_seed = 1
                self.tokens_per_batch = 512
                self.float_precision = 32
                self.mantissa_len = 23
                self.max_exponent = 127
                self.max_token_len = 8
                self.max_int = 100000
                self.min_len_per_dim = 1
                self.max_len = 512

                # 设备相关
                self.cpu = self.device.type == 'cpu'
                self.device = self.device
                self.reload_model = model_path if os.path.exists(model_path) else ""
                self.is_proppred = False  # 预训练模式

                # 环境参数
                self.env_name = "symbolic"
                self.use_sympy = True
                self.simplify = True
                self.use_abs = True

                # 其他必需参数
                self.ablation_to_keep = None
                self.max_input_points = 10000
                self.required_operators = []
                self.extra_unary_operators = []
                self.extra_binary_operators = []
                self.extra_constants = []

                # 验证参数
                check_model_params(self)

        return Params()

    def embed_expression(self, expression):
        """
        将数学表达式嵌入为潜向量
        Args:
            expression: 数学表达式字符串
        Returns:
            torch.Tensor: 表达式潜向量
        """
        if self.use_snip:
            with torch.no_grad():
                # 将表达式转换为token序列
                tokens = self.env.word_to_idx([expression], float_input=False)
                x, lengths = self.env.batch_equations(tokens)

                # 移动到设备
                if not self.params.cpu:
                    x = x.cuda()
                    lengths = lengths.cuda()

                # 通过符号编码器
                encoded = self.encoder_f("fwd", x=x, lengths=lengths, causal=False)

                # 取平均作为表达式嵌入 (格式: seq_len x batch x hidden)
                return encoded.mean(dim=0).squeeze(0)
        else:
            # 简化嵌入方法
            return self._embed_expression_simple(expression)

    def embed_data_points(self, data_points):
        """
        将数据点对嵌入为潜向量
        Args:
            data_points: [(x, y), ...] 格式的数据点列表
        Returns:
            torch.Tensor: 数据点潜向量
        """
        if self.use_snip:
            with torch.no_grad():
                # 通过嵌入器处理数据点
                embedded_data, lengths = self.embedder([data_points])

                # 移动到设备
                if not self.params.cpu:
                    embedded_data = embedded_data.cuda()
                    lengths = lengths.cuda()

                # 通过数值编码器
                encoded = self.encoder_y("fwd", x=embedded_data, lengths=lengths, causal=False)

                # 取平均作为数据嵌入 (格式: seq_len x batch x hidden)
                return encoded.mean(dim=0).squeeze(0)
        else:
            # 简化嵌入方法
            return self._embed_data_points_simple(data_points)

    def _embed_expression_simple(self, expression):
        """简化的表达式嵌入方法"""
        # 基于数学函数的特征
        features = [
            float('sin' in expression),
            float('cos' in expression),
            float('exp' in expression),
            float('log' in expression),
            float('sqrt' in expression),
            float('tan' in expression),
            float('abs' in expression),
            float('**' in expression),
            float('/' in expression),
            float('+' in expression),
            float(len(expression)),
            float(expression.count('('))
        ]
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _embed_data_points_simple(self, data_points):
        """简化的数据点嵌入方法"""
        if not data_points:
            return torch.zeros(7, dtype=torch.float32, device=self.device)

        # 提取x和y值
        x_vals = np.array([p[0] for p in data_points])
        y_vals = np.array([p[1] for p in data_points])

        # 计算统计特征
        features = [
            np.mean(y_vals),      # y均值
            np.std(y_vals),       # y标准差
            np.max(y_vals),       # y最大值
            np.min(y_vals),       # y最小值
        ]

        # 计算变化率特征
        if len(y_vals) > 1:
            y_diff = np.diff(y_vals)
            features.extend([
                np.mean(y_diff),    # 平均变化率
                np.std(y_diff)       # 变化率标准差
            ])
        else:
            features.extend([0.0, 0.0])

        features.append(len(data_points))  # 数据点数量

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def cosine_similarity(self, embedding1, embedding2):
        """计算余弦相似度"""
        # 确保是1D张量
        if embedding1.dim() > 1:
            embedding1 = embedding1.squeeze()
        if embedding2.dim() > 1:
            embedding2 = embedding2.squeeze()

        # 处理NaN值
        embedding1 = torch.nan_to_num(embedding1, nan=0.0)
        embedding2 = torch.nan_to_num(embedding2, nan=0.0)

        # 计算余弦相似度
        norm1 = torch.norm(embedding1)
        norm2 = torch.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(torch.dot(embedding1, embedding2) / (norm1 * norm2))

    def load_expressions(self, file_path):
        """从文件加载表达式列表"""
        expressions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    expressions.append(line)
        return expressions

    def load_data_points(self, file_path):
        """从CSV文件加载数据点对"""
        df = pd.read_csv(file_path)
        data_dict = defaultdict(list)

        for _, row in df.iterrows():
            expr_id = int(row['expression_id'])
            x = float(row['x'])
            y = float(row['y'])
            data_dict[expr_id].append((x, y))

        # 转换为列表，按expression_id排序
        if data_dict:
            max_id = max(data_dict.keys())
            return [data_dict[i] if i in data_dict else [] for i in range(max_id + 1)]
        return []

    def load_expressions_with_data(self, file_path):
        """
        从新格式文件加载表达式和对应的数据点
        格式: EXPRESSION: <expression>
              x1,x2,x3,...,y (data rows)
        Returns:
            tuple: (expressions_list, data_points_list)
        """
        expressions = []
        data_points = []
        current_data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # 跳过注释行和空行
                if not line or line.startswith('#'):
                    continue

                # 检查是否是表达式行
                if line.startswith('EXPRESSION:'):
                    # 保存之前的数据点
                    if current_data:
                        data_points.append(current_data)
                        current_data = []

                    # 提取表达式
                    expression = line.replace('EXPRESSION:', '').strip()
                    expressions.append(expression)

                # 检查是否是数据行
                elif ',' in line and not line.startswith('EXPRESSION:'):
                    # 解析数据点: x1,x2,x3,...,y
                    parts = line.split(',')
                    try:
                        # 最后一列是y，前面都是x
                        x_values = [float(x) for x in parts[:-1]]
                        y_value = float(parts[-1])

                        # 存储为 (x1, x2, x3, ..., y) 格式
                        data_point = tuple(x_values + [y_value])
                        current_data.append(data_point)
                    except ValueError:
                        # 跳过无法解析的行
                        continue

            # 保存最后一个表达式的数据
            if current_data:
                data_points.append(current_data)

        return expressions, data_points

    def load_expressions_with_data_multidim(self, file_path):
        """
        从新格式文件加载表达式和对应的多维数据点
        返回格式适合多维处理
        Returns:
            tuple: (expressions_list, data_points_list)
            其中data_points是每个表达式的数据点列表，每个数据点是(features, y)格式
        """
        expressions = []
        data_points = []
        current_data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # 跳过注释行和空行
                if not line or line.startswith('#'):
                    continue

                # 检查是否是表达式行
                if line.startswith('EXPRESSION:'):
                    # 保存之前的数据点
                    if current_data:
                        data_points.append(current_data)
                        current_data = []

                    # 提取表达式
                    expression = line.replace('EXPRESSION:', '').strip()
                    expressions.append(expression)

                # 检查是否是数据行
                elif ',' in line and not line.startswith('EXPRESSION:'):
                    # 解析数据点: x1,x2,x3,...,y
                    parts = line.split(',')
                    try:
                        # 最后一列是y，前面都是features
                        features = [float(x) for x in parts[:-1]]
                        y_value = float(parts[-1])

                        # 存储为 (features, y) 格式，其中features是列表
                        data_point = (features, y_value)
                        current_data.append(data_point)
                    except ValueError:
                        # 跳过无法解析的行
                        continue

            # 保存最后一个表达式的数据
            if current_data:
                data_points.append(current_data)

        return expressions, data_points

    def embed_multidimensional_data_points(self, data_points):
        """
        将多维数据点嵌入为潜向量
        Args:
            data_points: [(features, y), ...] 格式的数据点列表
        Returns:
            torch.Tensor: 数据点潜向量
        """
        if self.use_snip:
            # 对于SNIP模型，需要将多维数据转换为其格式
            # 这里简化处理，取第一个特征作为x，y作为标签
            snip_data = []
            for features, y in data_points:
                if len(features) > 0:
                    # 使用第一个特征作为主要变量，其他特征作为辅助信息
                    x = features[0]
                    snip_data.append((x, y))

            return self.embed_data_points(snip_data)
        else:
            # 简化嵌入方法，处理多维数据
            return self._embed_multidimensional_data_points_simple(data_points)

    def _embed_multidimensional_data_points_simple(self, data_points):
        """简化的多维数据点嵌入方法"""
        if not data_points:
            return torch.zeros(15, dtype=torch.float32, device=self.device)  # 增加维度

        # 提取所有特征和y值
        all_features = []
        y_values = []

        for features, y in data_points:
            all_features.extend(features)
            y_values.append(y)

        # 计算统计特征
        features_array = np.array(all_features)
        y_array = np.array(y_values)

        # 特征统计
        features_stats = [
            np.mean(features_array),     # 特征均值
            np.std(features_array),      # 特征标准差
            np.max(features_array),      # 特征最大值
            np.min(features_array),      # 特征最小值
        ]

        # y值统计
        y_stats = [
            np.mean(y_array),           # y均值
            np.std(y_array),            # y标准差
            np.max(y_array),            # y最大值
            np.min(y_array),            # y最小值
        ]

        # 计算特征间的关系
        if len(data_points) > 1 and len(all_features) >= len(data_points) * 2:
            # 假设每个数据点至少有2个特征
            n_features_per_point = len(all_features) // len(data_points)
            if n_features_per_point >= 2:
                # 计算第一个和第二个特征的相关性
                feat1 = [all_features[i*n_features_per_point] for i in range(len(data_points))]
                feat2 = [all_features[i*n_features_per_point + 1] for i in range(len(data_points))]

                if len(feat1) > 1 and len(feat2) > 1:
                    corr = np.corrcoef(feat1, feat2)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0

                relation_features = [
                    corr,                     # 特征相关性
                    np.mean(feat1),          # 第一个特征均值
                    np.std(feat1),           # 第一个特征标准差
                    np.mean(feat2),          # 第二个特征均值
                    np.std(feat2),           # 第二个特征标准差
                ]
            else:
                relation_features = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            relation_features = [0.0, 0.0, 0.0, 0.0, 0.0]

        # 其他特征
        other_features = [
            len(data_points),             # 数据点数量
            len(all_features) // len(data_points),  # 平均特征数
        ]

        # 合并所有特征
        all_feature_stats = features_stats + y_stats + relation_features + other_features

        return torch.tensor(all_feature_stats, dtype=torch.float32, device=self.device)

    def calculate_similarity_matrix(self, expressions1, data_points1, expressions2, data_points2):
        """
        计算两组表达式和数据点的相似度矩阵
        Args:
            expressions1, expressions2: 表达式列表
            data_points1, data_points2: 数据点列表
        Returns:
            dict: 包含各种相似度矩阵
        """
        print("计算嵌入向量...")

        # 计算第一组嵌入
        embeddings1_expr = [self.embed_expression(expr) for expr in expressions1]
        embeddings1_data = [self.embed_data_points(data) for data in data_points1]

        # 计算第二组嵌入
        embeddings2_expr = [self.embed_expression(expr) for expr in expressions2]
        embeddings2_data = [self.embed_data_points(data) for data in data_points2]

        print("计算相似度矩阵...")

        n1, n2 = len(expressions1), len(expressions2)

        # 计算表达式相似度
        expr_expr_similarities = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                expr_expr_similarities[i, j] = self.cosine_similarity(
                    embeddings1_expr[i], embeddings2_expr[j]
                )

        # 计算数据相似度
        data_data_similarities = np.zeros((n1, n2))
        for i in range(min(n1, len(embeddings1_data))):
            for j in range(min(n2, len(embeddings2_data))):
                data_data_similarities[i, j] = self.cosine_similarity(
                    embeddings1_data[i], embeddings2_data[j]
                )

        # 组合相似度
        combined_similarities = (expr_expr_similarities + data_data_similarities) / 2

        return {
            'combined': combined_similarities,
            'expr_expr': expr_expr_similarities,
            'data_data': data_data_similarities
        }

    def visualize_similarity_matrix(self, similarity_matrix, expressions1, expressions2, output_path):
        """可视化相似度矩阵"""
        plt.figure(figsize=(15, 12))
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 组合相似度
        sns.heatmap(similarity_matrix['combined'],
                   xticklabels=[f"E{i+1}" for i in range(len(expressions2))],
                   yticklabels=[f"E{i+1}" for i in range(len(expressions1))],
                   annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('组合相似度 (表达式+数据)')
        axes[0, 0].set_xlabel('表达式组2')
        axes[0, 0].set_ylabel('表达式组1')

        # 表达式相似度
        sns.heatmap(similarity_matrix['expr_expr'],
                   xticklabels=[f"E{i+1}" for i in range(len(expressions2))],
                   yticklabels=[f"E{i+1}" for i in range(len(expressions1))],
                   annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 1])
        axes[0, 1].set_title('表达式相似度')
        axes[0, 1].set_xlabel('表达式组2')
        axes[0, 1].set_ylabel('表达式组1')

        # 数据相似度
        sns.heatmap(similarity_matrix['data_data'],
                   xticklabels=[f"E{i+1}" for i in range(len(expressions2))],
                   yticklabels=[f"E{i+1}" for i in range(len(expressions1))],
                   annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('数据相似度')
        axes[1, 0].set_xlabel('表达式组2')
        axes[1, 0].set_ylabel('表达式组1')

        # 相似度分布
        all_similarities = similarity_matrix['combined'].flatten()
        axes[1, 1].hist(all_similarities, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('相似度分布')
        axes[1, 1].set_xlabel('相似度值')
        axes[1, 1].set_ylabel('频次')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"相似度矩阵可视化已保存到: {output_path}")

    def run_analysis(self, base_dir):
        """运行完整的相似度分析"""
        # 文件路径
        expressions1_path = f"{base_dir}/data/expressions/expressions1.txt"
        expressions2_path = f"{base_dir}/data/expressions/expressions2.txt"
        data_points1_path = f"{base_dir}/data/data_points/data_points1.csv"
        data_points2_path = f"{base_dir}/data/data_points/data_points2.csv"

        # 检查文件存在
        for path in [expressions1_path, expressions2_path, data_points1_path, data_points2_path]:
            if not os.path.exists(path):
                print(f"文件不存在: {path}")
                return

        print("加载数据...")

        # 加载数据
        expressions1 = self.load_expressions(expressions1_path)
        expressions2 = self.load_expressions(expressions2_path)
        data_points1 = self.load_data_points(data_points1_path)
        data_points2 = self.load_data_points(data_points2_path)

        print(f"加载了 {len(expressions1)} 个表达式和 {len(data_points1)} 组数据点 (组1)")
        print(f"加载了 {len(expressions2)} 个表达式和 {len(data_points2)} 组数据点 (组2)")

        # 显示示例
        print("\n表达式示例:")
        for i in range(min(3, len(expressions1))):
            print(f"  组1 E{i+1}: {expressions1[i]}")
        for i in range(min(3, len(expressions2))):
            print(f"  组2 E{i+1}: {expressions2[i]}")

        # 计算相似度矩阵
        similarity_matrix = self.calculate_similarity_matrix(
            expressions1, data_points1, expressions2, data_points2
        )

        # 保存结果
        output_dir = f"{base_dir}/results"
        os.makedirs(output_dir, exist_ok=True)

        # 保存CSV矩阵
        df = pd.DataFrame(
            similarity_matrix['combined'],
            index=[f"E{i+1}" for i in range(len(expressions1))],
            columns=[f"E{i+1}" for i in range(len(expressions2))]
        )
        df.to_csv(f"{output_dir}/snip_similarity_matrix.csv")

        # 保存分析报告
        self._save_analysis_report(
            similarity_matrix, expressions1, expressions2,
            output_dir, self.use_snip
        )

        # 生成可视化
        self.visualize_similarity_matrix(
            similarity_matrix, expressions1, expressions2,
            f"{output_dir}/snip_similarity_matrix.png"
        )

        print(f"\n结果已保存到 {output_dir}")
        print(f"  - snip_similarity_matrix.csv: 相似度矩阵")
        print(f"  - snip_similarity_analysis.txt: 详细分析")
        print(f"  - snip_similarity_matrix.png: 可视化图表")

    def _save_analysis_report(self, similarity_matrix, expressions1, expressions2, output_dir, use_snip):
        """保存分析报告"""
        with open(f"{output_dir}/snip_similarity_analysis.txt", "w") as f:
            f.write("SNIP相似度计算结果\n")
            f.write("=" * 50 + "\n\n")

            model_type = "SNIP预训练模型" if use_snip else "简化计算方法"
            f.write(f"使用模型: {model_type}\n\n")

            f.write("表达式组1:\n")
            for i, expr in enumerate(expressions1):
                f.write(f"  E{i+1}: {expr}\n")

            f.write("\n表达式组2:\n")
            for i, expr in enumerate(expressions2):
                f.write(f"  E{i+1}: {expr}\n")

            f.write(f"\n相似度统计:\n")
            sim_matrix = similarity_matrix['combined']
            f.write(f"  平均相似度: {np.mean(sim_matrix):.4f}\n")
            f.write(f"  最大相似度: {np.max(sim_matrix):.4f}\n")
            f.write(f"  最小相似度: {np.min(sim_matrix):.4f}\n")
            f.write(f"  相似度标准差: {np.std(sim_matrix):.4f}\n")

            # 找出最相似的配对
            max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            f.write(f"\n最相似配对: 组1 E{max_idx[0]+1} <-> 组2 E{max_idx[1]+1}\n")
            f.write(f"  组1表达式: {expressions1[max_idx[0]]}\n")
            f.write(f"  组2表达式: {expressions2[max_idx[1]]}\n")
            f.write(f"  相似度: {sim_matrix[max_idx]:.4f}\n")

    def run_analysis_new_format(self, base_dir):
        """运行新格式数据的完整相似度分析"""
        # 新格式文件路径
        expressions1_path = f"{base_dir}/data/new_format/expressions1.csv"
        expressions2_path = f"{base_dir}/data/new_format/expressions2.csv"

        # 检查文件存在
        for path in [expressions1_path, expressions2_path]:
            if not os.path.exists(path):
                print(f"文件不存在: {path}")
                return

        print("加载新格式数据...")

        # 加载新格式数据
        expressions1, data_points1 = self.load_expressions_with_data_multidim(expressions1_path)
        expressions2, data_points2 = self.load_expressions_with_data_multidim(expressions2_path)

        print(f"加载了 {len(expressions1)} 个表达式和对应的多维数据点 (组1)")
        print(f"加载了 {len(expressions2)} 个表达式和对应的多维数据点 (组2)")

        # 显示示例
        print("\n表达式和数据示例:")
        for i in range(min(3, len(expressions1))):
            expr = expressions1[i]
            data_count = len(data_points1[i])
            feature_dim = len(data_points1[i][0][0]) if data_points1[i] else 0
            print(f"  组1 E{i+1}: {expr} (数据点: {data_count}, 特征维度: {feature_dim})")

        for i in range(min(3, len(expressions2))):
            expr = expressions2[i]
            data_count = len(data_points2[i])
            feature_dim = len(data_points2[i][0][0]) if data_points2[i] else 0
            print(f"  组2 E{i+1}: {expr} (数据点: {data_count}, 特征维度: {feature_dim})")

        print("\n计算嵌入向量...")

        # 计算第一组嵌入
        embeddings1_expr = [self.embed_expression(expr) for expr in expressions1]
        embeddings1_data = [self.embed_multidimensional_data_points(data) for data in data_points1]

        # 计算第二组嵌入
        embeddings2_expr = [self.embed_expression(expr) for expr in expressions2]
        embeddings2_data = [self.embed_multidimensional_data_points(data) for data in data_points2]

        print("计算相似度矩阵...")

        n1, n2 = len(expressions1), len(expressions2)

        # 计算表达式相似度
        expr_expr_similarities = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                expr_expr_similarities[i, j] = self.cosine_similarity(
                    embeddings1_expr[i], embeddings2_expr[j]
                )

        # 计算数据相似度
        data_data_similarities = np.zeros((n1, n2))
        for i in range(min(n1, len(embeddings1_data))):
            for j in range(min(n2, len(embeddings2_data))):
                data_data_similarities[i, j] = self.cosine_similarity(
                    embeddings1_data[i], embeddings2_data[j]
                )

        # 组合相似度
        combined_similarities = (expr_expr_similarities + data_data_similarities) / 2

        similarity_matrix = {
            'combined': combined_similarities,
            'expr_expr': expr_expr_similarities,
            'data_data': data_data_similarities
        }

        # 保存结果
        output_dir = f"{base_dir}/results/new_format"
        os.makedirs(output_dir, exist_ok=True)

        # 保存CSV矩阵
        df = pd.DataFrame(
            similarity_matrix['combined'],
            index=[f"E{i+1}" for i in range(len(expressions1))],
            columns=[f"E{i+1}" for i in range(len(expressions2))]
        )
        df.to_csv(f"{output_dir}/multidim_similarity_matrix.csv")

        # 保存分析报告
        self._save_analysis_report_new_format(
            similarity_matrix, expressions1, expressions2, data_points1, data_points2,
            output_dir, self.use_snip
        )

        # 生成可视化
        self.visualize_similarity_matrix(
            similarity_matrix, expressions1, expressions2,
            f"{output_dir}/multidim_similarity_matrix.png"
        )

        print(f"\n结果已保存到 {output_dir}")
        print(f"  - multidim_similarity_matrix.csv: 多维相似度矩阵")
        print(f"  - multidim_analysis.txt: 详细分析")
        print(f"  - multidim_similarity_matrix.png: 可视化图表")

    def _save_analysis_report_new_format(self, similarity_matrix, expressions1, expressions2,
                                         data_points1, data_points2, output_dir, use_snip):
        """保存新格式分析报告"""
        with open(f"{output_dir}/multidim_analysis.txt", "w") as f:
            f.write("多维SNIP相似度计算结果\n")
            f.write("=" * 60 + "\n\n")

            model_type = "SNIP预训练模型" if use_snip else "简化计算方法"
            f.write(f"使用模型: {model_type}\n\n")

            f.write("表达式组1:\n")
            for i, expr in enumerate(expressions1):
                data_count = len(data_points1[i])
                feature_dim = len(data_points1[i][0][0]) if data_points1[i] else 0
                f.write(f"  E{i+1}: {expr}\n")
                f.write(f"      数据点数: {data_count}, 特征维度: {feature_dim}\n")

            f.write("\n表达式组2:\n")
            for i, expr in enumerate(expressions2):
                data_count = len(data_points2[i])
                feature_dim = len(data_points2[i][0][0]) if data_points2[i] else 0
                f.write(f"  E{i+1}: {expr}\n")
                f.write(f"      数据点数: {data_count}, 特征维度: {feature_dim}\n")

            f.write(f"\n相似度统计:\n")
            sim_matrix = similarity_matrix['combined']
            f.write(f"  平均相似度: {np.mean(sim_matrix):.4f}\n")
            f.write(f"  最大相似度: {np.max(sim_matrix):.4f}\n")
            f.write(f"  最小相似度: {np.min(sim_matrix):.4f}\n")
            f.write(f"  相似度标准差: {np.std(sim_matrix):.4f}\n")

            # 找出最相似的配对
            max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            f.write(f"\n最相似配对: 组1 E{max_idx[0]+1} <-> 组2 E{max_idx[1]+1}\n")
            f.write(f"  组1表达式: {expressions1[max_idx[0]]}\n")
            f.write(f"  组2表达式: {expressions2[max_idx[1]]}\n")
            f.write(f"  相似度: {sim_matrix[max_idx]:.4f}\n")

            # 数据维度统计
            f.write(f"\n数据维度统计:\n")
            all_feature_dims = []
            all_data_counts = []

            for data_list in [data_points1, data_points2]:
                for data in data_list:
                    if data:
                        all_data_counts.append(len(data))
                        all_feature_dims.append(len(data[0][0]))

            if all_feature_dims:
                f.write(f"  平均特征维度: {np.mean(all_feature_dims):.1f}\n")
                f.write(f"  平均数据点数: {np.mean(all_data_counts):.1f}\n")


def main():
    """主函数"""
    base_dir = "/home/xyh/Meta-MCSR/similarity_project"

    # 初始化计算器
    model_path = "/home/xyh/Meta-MCSR/weights/snip-1d-normalized.pth"
    calculator = SNIPSimmilarityCalculator(model_path)

    # 运行新格式分析
    print("=" * 60)
    print("运行新格式多维数据分析")
    print("=" * 60)
    calculator.run_analysis_new_format(base_dir)

    print("\n" + "=" * 60)
    print("运行原始格式分析 (对比)")
    print("=" * 60)
    # 运行原始分析作为对比
    calculator.run_analysis(base_dir)


if __name__ == "__main__":
    main()