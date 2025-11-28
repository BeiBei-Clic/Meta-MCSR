import numpy as np
import pandas as pd
import operator
import random
from deap import base, creator, tools, gp, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import os
import json
import functools
import sys
import torch

# 导入相似度计算器
from src.snip_similarity_calculator import SimilarityCalculator
import snip
from snip.model import check_model_params, build_modules
from snip.envs import build_env

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 全局变量，避免重复创建类
_fitness_created = False
_individual_created = False
_similarity_calculator = None

def tree_to_prefix_string(tree):
    """将DEAP的树结构转换为前序遍历字符串"""
    def traverse(node):
        if isinstance(node, gp.Primitive):
            # 操作符节点
            args = [traverse(child) for child in node.args]
            if len(args) == 1:
                return f"{node.name},{args[0]}"
            else:
                return f"{node.name}," + ",".join(args)
        elif isinstance(node, gp.Terminal):
            # 终端节点（变量或常数）
            if isinstance(node.value, str):
                # 变量
                return node.value
            else:
                # 常数
                return str(node.value)
        else:
            return str(node)

    return traverse(tree)

def tree_to_prefix_tokens(tree):
    """将DEAP的树结构转换为前序遍历token列表"""
    tokens = []
    def traverse(node):
        if isinstance(node, gp.Primitive):
            # 操作符节点
            tokens.append(node.name)
            for child in node.args:
                traverse(child)
        elif isinstance(node, gp.Terminal):
            # 终端节点（变量或常数）
            if isinstance(node.value, str):
                # 变量
                tokens.append(node.value)
            else:
                # 常数
                tokens.append(str(node.value))

    traverse(tree)
    return tokens


def create_snip_compatible_data(X_data, y_data, tree_str):
    """创建与SNIP兼容的数据格式，完全按照test_data的格式"""

    # 简化统计（与test_data格式一致）
    tree_tokens = tree_str.split(',')
    n_unary_ops = str(tree_tokens.count('sqrt') + tree_tokens.count('log') +
                         tree_tokens.count('sin') + tree_tokens.count('cos') +
                         tree_tokens.count('neg') + tree_tokens.count('inv'))
    n_binary_ops = str(tree_tokens.count('add') + tree_tokens.count('sub') +
                          tree_tokens.count('mul') + tree_tokens.count('div') +
                          tree_tokens.count('pow'))

    # 创建正确的数据格式：x_to_fit是二维数组，每行是一个数据点
    x_to_fit = []
    for i in range(len(X_data)):
        row = []
        for j in range(X_data.shape[1]):
            row.append(f"{X_data[i, j]:.6e}")
        x_to_fit.append(row)

    # y_to_fit也是二维数组，每行是一个目标值
    y_to_fit = [[f"{y:.6e}"] for y in y_data]

    # 生成skeleton_tree_encoded（基于tree_str生成）
    skeleton_tree_encoded = []
    for token in tree_str.split(','):
        if token in ['add', 'sub', 'mul', 'div', 'pow', 'pow2', 'pow3', 'sin', 'cos', 'log', 'sqrt', 'neg', 'inv']:
            skeleton_tree_encoded.append(token)
        else:
            # 常数或变量替换为CONSTANT
            skeleton_tree_encoded.append("CONSTANT")

    # 创建完整的JSON数据格式，完全按照test_data的格式
    data = {
        "n_input_points": str(len(X_data)),
        "n_unary_ops": n_unary_ops,
        "n_binary_ops": n_binary_ops,
        "d_in": str(X_data.shape[1]),
        "d_out": "1",
        "input_distribution_type": "0",
        "n_centroids": "1",
        "x_to_fit": x_to_fit,
        "y_to_fit": y_to_fit,
        "tree": tree_str,
        "skeleton_tree_encoded": skeleton_tree_encoded
    }
    return data

def setup_similarity_calculator():
    """设置相似度计算器"""
    global _similarity_calculator

    if _similarity_calculator is not None:
        return _similarity_calculator

    # 创建基本参数，与命令行参数保持一致
    class SimpleParams:
        def __init__(self):
            # 基本设置 - 与命令行参数匹配
            self.cpu = False  # 使用GPU进行计算
            self.eval_only = True
            self.tasks = ["functions"]
            self.eval_from_exp = "weights/snip-10dmax.pth"
            self.eval_data = "/tmp/temp_gp_data.json"
            self.dump_path = "./eval_output"
            self.max_input_dimension = 10
            self.max_output_dimension = 1

            # 关键参数：latent_dim需要与模型匹配（设置为512）
            self.latent_dim = 512

            # 添加必需的SNIP参数 - 与parsers.py保持一致
            self.env_name = "functions"
            self.emb_dim = 256
            self.emb_enc_hidden_dim = 256
            self.emb_enc_layers = 3
            self.transformer_layers = 6
            self.transformer_heads = 8
            self.ffn_dim = 1024

            # 任务参数 - 应该是字符串而不是列表
            self.tasks = "functions"

            # 添加parsers.py中的必要参数
            self.fp16 = False
            self.embedder_type = "LinearPoint"
            self.normalize_y = False
            self.bt_lambda = 0.0051
            self.load_encoder = False
            self.freeze_encoder = False
            self.emb_emb_dim = 64
            self.enc_emb_dim = 512
            self.dec_emb_dim = 512
            self.n_emb_layers = 1
            self.n_enc_layers = 8
            self.n_dec_layers = 8
            self.n_enc_heads = 16
            self.n_dec_heads = 16
            self.emb_expansion_factor = 1
            self.n_enc_hidden_layers = 1
            self.n_dec_hidden_layers = 1
            self.norm_attention = False
            self.dropout = 0
            self.attention_dropout = 0
            self.share_inout_emb = True
            self.enc_positional_embeddings = "learnable"
            self.dec_positional_embeddings = "learnable"
            self.loss_type = "CLIP"
            self.contrastive_weight = 1.0
            self.max_src_len = 200
            self.max_target_len = 200
            self.batch_size = 32
            self.batch_size_eval = 32

            # 训练参数（对于评估不重要）
            self.learning_rate = 1e-4
            self.n_steps = 1000
            self.eval_steps = 100

            # 环境参数 - 来自snip/envs/environment.py的默认值
            self.float_precision = 3
            self.mantissa_len = 1
            self.max_exponent = 100
            self.max_exponent_prefactor = 10
            self.split_with_train_validation = False

            # 生成器参数
            self.prob_const = 0.0
            self.prob_rand = 0.0
            self.max_int = 10
            self.min_binary_ops_per_dim = 0
            self.max_binary_ops_per_dim = 1
            self.max_binary_ops_offset = 4
            self.min_unary_ops = 0
            self.max_unary_ops = 4
            self.min_input_dimension = 1
            self.min_output_dimension = 1

            # 其他环境参数
            self.use_skeleton = False
            self.queue_strategy = None
            self.collate_queue_size = 2000
            self.use_sympy = False
            self.simplify = False
            self.use_abs = False
            self.operators_to_downsample = "div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3"
            self.operators_to_not_repeat = ""
            self.max_unary_depth = 6
            self.required_operators = ""
            self.extra_unary_operators = ""
            self.extra_binary_operators = ""
            self.extra_constants = None
            self.enforce_dim = True
            self.use_controller = True
            self.max_token_len = 0
            self.tokens_per_batch = 10000
            self.pad_to_max_dim = True
            self.max_len = 200
            self.min_op_prob = 0.01
            self.n_input_points_LSO = 1000
            self.min_len_per_dim = 5
            self.max_centroids = 10
            self.reduce_num_constants = True
            self.max_trials = 1
            self.n_prediction_points = 200
            self.prediction_sigmas = "1,2,4,8,16"

            # 系统参数 - 来自parsers.py
            self.local_rank = -1
            self.master_port = -1
            self.windows = False
            self.nvidia_apex = False

            # 环境种子参数
            self.env_base_seed = 0
            self.test_env_seed = 1

            # 数据加载参数
            self.batch_load = False
            self.reload_size = -1

            # 其他必需的参数
            self.num_workers = 0
            self.n_gpu_per_node = 1
            self.global_rank = 0
            self.n_steps_per_epoch = 1000
            self.train_noise_gamma = 0.0
            self.eval_noise_gamma = 0.0
            self.eval_input_length_modulo = -1

            # 其他可能需要的参数
            self.is_proppred = False
            self.property_type = None
            self.reload_model = ""  # 使用空字符串而不是None
            self.reload_checkpoint = ""

    params = SimpleParams()

    # 检查必需的参数
    check_model_params(params)

    # 创建环境和模块
    env = build_env(params)
    modules = build_modules(env, params)
    _similarity_calculator = SimilarityCalculator(modules, env, params)
    print("相似度计算器初始化成功")
    return _similarity_calculator

def protected_div(left, right):
    """保护除法，避免除零错误"""
    if abs(right) < 1e-6:
        return 1.0
    result = left / right
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def protected_sqrt(x):
    """保护平方根，避免负数开方"""
    result = math.sqrt(abs(x))
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def protected_log(x):
    """保护对数，避免负数或零的对数"""
    result = math.log(abs(x) + 1e-6)
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result

def protected_exp(x):
    """保护指数，避免溢出"""
    x = max(min(x, 50), -50)  # 更严格的限制
    result = math.exp(x)
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def protected_pow(left, right):
    """保护幂运算"""
    if abs(left) < 1e-6 and right < 0:
        return 1.0
    if abs(right) > 10:  # 限制指数大小
        right = 10 if right > 0 else -10
    result = abs(left) ** right
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def protected_sin(x):
    """保护正弦函数"""
    result = math.sin(x)
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result

def protected_cos(x):
    """保护余弦函数"""
    result = math.cos(x)
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def load_and_preprocess_data(dataset_name, sample_size=1000):
    """加载并预处理数据"""
    print(f"正在加载数据集: {dataset_name}")
    
    # 构建数据文件路径
    data_path = f'dump/dataset/{dataset_name}'
    
    # 读取指定数量的数据
    data = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            values = line.strip().split()
            if len(values) >= 2:  # 至少需要2列数据
                data.append([float(v) for v in values])
    
    data = np.array(data)
    
    # 检查数据维度
    if data.ndim == 1:
        raise ValueError(f"数据集 {dataset_name} 格式错误：只有一行数据")

    num_cols = data.shape[1]
    print(f"数据形状: {data.shape}, 列数: {num_cols}")

    X = data[:, :-1]  # 前面的列作为输入特征
    y = data[:, -1]  # 最后一列作为目标值
    
    print(f"处理后数据形状: X={X.shape}, y={y.shape}")
    
    # 按8:2比例分割训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"训练集大小: {X_train_scaled.shape[0]}")
    print(f"测试集大小: {X_test_scaled.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def setup_gp(num_inputs=2):
    """设置遗传编程环境"""
    # 创建原语集
    pset = gp.PrimitiveSet("MAIN", num_inputs)  # 动态设置输入变量数量
    
    # 添加基本运算符
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(protected_pow, 2)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(protected_log, 1)
    pset.addPrimitive(protected_exp, 1)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(protected_sin, 1)
    pset.addPrimitive(protected_cos, 1)
    
    # 使用functools.partial替代lambda来避免警告
    pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -2, 2))
    
    # 重命名参数
    arg_mapping = {}
    for i in range(num_inputs):
        arg_mapping[f'ARG{i}'] = f'x{i+1}'
    pset.renameArguments(**arg_mapping)
    
    return pset

def create_fitness_and_individual(pset):
    """创建适应度和个体类型"""
    global _fitness_created, _individual_created
    
    # 只在第一次调用时创建类，避免重复创建警告
    if not _fitness_created:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        _fitness_created = True
    
    if not _individual_created:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        _individual_created = True
    
    return creator.Individual

def setup_toolbox(pset, Individual):
    """设置工具箱"""
    toolbox = base.Toolbox()
    
    # 注册表达式生成器
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    return toolbox

def evaluate_individual_with_similarity(individual, toolbox, X_train, y_train, similarity_calculator):
    """使用相似度计算器评估个体的适应度"""
    # 将树转换为前序遍历字符串
    tree_str = tree_to_prefix_string(individual)
    print(f"生成的树字符串: {tree_str}")

    # 验证生成的树字符串是否有效
    if not tree_str or tree_str.strip() == "":
        # 如果树字符串为空，返回一个很大的适应度值（坏的结果）
        print(f"警告：生成的树字符串为空！")
        return (1e6,)

    # 确保树字符串包含有效的操作符
    valid_ops = ['add', 'sub', 'mul', 'div', 'pow', 'pow2', 'pow3', 'sin', 'cos', 'log', 'sqrt', 'neg', 'inv']
    tokens = tree_str.split(',')
    has_valid_op = any(token in valid_ops for token in tokens if token.strip())

    if not has_valid_op:
        # 如果没有有效的操作符，返回一个很大的适应度值（坏的结果）
        # print(f"警告：生成的树字符串没有有效的操作符！")
        return (1e6,)

    # 创建兼容的数据格式
    data_dict = create_snip_compatible_data(X_train, y_train, tree_str)

    # 创建临时数据文件
    temp_file = "/tmp/temp_gp_data.json"
    import json
    with open(temp_file, 'w') as f:
        json.dump(data_dict, f)

    # 使用相似度计算器
    similarity_matrix = similarity_calculator.enc_dec_step(
        "functions",
        data_path={"functions": [temp_file]}
    )

    # 清理临时文件
    os.remove(temp_file)

    # 从相似度矩阵中提取适应度值
    # 相似度越高，适应度越好（注意适应度是最小化的）
    if similarity_matrix is not None and similarity_matrix.numel() > 0:
        # 取平均相似度，转换为距离（1 - similarity）作为适应度
        avg_similarity = similarity_matrix.mean().item()
        fitness = 1.0 - max(0.0, min(1.0, avg_similarity))  # 确保在[0,1]范围内
        return fitness,

def run_genetic_programming(dataset_name, sample_size=1000, population_size=300, generations=100, random_seed=42):
    """运行遗传编程算法"""
    print(f"开始遗传编程算法... 数据集: {dataset_name}, 随机种子: {random_seed}")
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 加载和预处理数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_preprocess_data(dataset_name, sample_size)
    
    # 设置遗传编程
    input_dim = X_train.shape[1]  # 获取输入特征维度
    pset = setup_gp(input_dim)
    Individual = create_fitness_and_individual(pset)
    toolbox = setup_toolbox(pset, Individual)

    # 初始化相似度计算器
    similarity_calculator = setup_similarity_calculator()

    # 注册评估函数
    toolbox.register("evaluate", evaluate_individual_with_similarity, toolbox=toolbox,
                    X_train=X_train, y_train=y_train, similarity_calculator=similarity_calculator)
    
    # 注册遗传算子
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    
    # 限制树的深度
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    
    # 创建初始种群
    population = toolbox.population(n=population_size)
    
    # 创建名人堂保存最佳个体
    hof = tools.HallOfFame(1)
    
    # 设置统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("开始进化...")
    
    # 运行遗传算法
    population, logbook = algorithms.eaSimple(
        population, toolbox, 
        cxpb=0.5,      # 交叉概率
        mutpb=0.2,     # 变异概率
        ngen=generations,       # 进化代数
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    # 获取最佳个体
    best_individual = hof[0]
    best_func = toolbox.compile(expr=best_individual)
    
    print(f"\n最佳个体: {best_individual}")
    print(f"最佳适应度 (训练RMSE): {best_individual.fitness.values[0]:.6f}")
    
    # 在测试集上评估
    test_predictions = []
    for i in range(len(X_test)):
        # 将X_test[i]的所有特征作为参数传递给best_func
        pred = best_func(*X_test[i])
        if math.isnan(pred) or math.isinf(pred):
            pred = 0.0
        test_predictions.append(pred)
    
    test_predictions = np.array(test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    print(f"测试集RMSE: {test_rmse:.6f}")
    
    return {
        'best_individual': str(best_individual),
        'train_rmse': best_individual.fitness.values[0],
        'test_rmse': test_rmse,
        'logbook': logbook,
        'random_seed': random_seed
    }

def run_multiple_experiments(dataset_name, sample_size=1000, population_size=200, generations=300, num_runs=10):
    """对单个数据集运行多次实验"""
    print(f"\n开始对数据集 {dataset_name} 进行 {num_runs} 次实验...")
    
    results = []
    for seed in range(num_runs):
        print(f"\n--- 运行 {seed + 1}/{num_runs} (随机种子: {seed}) ---")
        result = run_genetic_programming(dataset_name, sample_size, population_size, generations, seed)
        results.append(result)
    
    # 计算统计信息
    test_rmses = [r['test_rmse'] for r in results]
    
    # 添加统计信息
    stats = {
        'dataset_name': dataset_name,
        'sample_size': sample_size,
        'population_size': population_size,
        'generations': generations,
        'num_runs': num_runs,
        'test_rmse_stats': {
            'mean': np.mean(test_rmses),
            'std': np.std(test_rmses),
            'min': np.min(test_rmses),
            'max': np.max(test_rmses)
        },
        'all_results': results
    }
    
    print(f"\n数据集 {dataset_name} 实验完成:")
    print(f"测试RMSE - 平均值: {stats['test_rmse_stats']['mean']:.6f}, 标准差: {stats['test_rmse_stats']['std']:.6f}")
    print(f"最小值: {stats['test_rmse_stats']['min']:.6f}, 最大值: {stats['test_rmse_stats']['max']:.6f}")
    
    return stats

def save_results(results, dataset_name):
    """保存实验结果到文件"""
    os.makedirs('results', exist_ok=True)
    
    # 保存详细结果
    filename = f'results/stGP_{dataset_name}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        # 处理logbook对象，转换为可序列化的格式
        results_copy = results.copy()
        for i, result in enumerate(results_copy['all_results']):
            if 'logbook' in result:
                logbook = result['logbook']
                # 只保存每一代的最佳适应度（最小值），不再保存generations字段
                results_copy['all_results'][i]['logbook'] = {
                    'best_fitness': [record['min'] for record in logbook]
                }
        
        json.dump(results_copy, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {filename}")
    print(f"已保存全部 {len(results['all_results'])} 次实验的每代最佳适应度变化过程，不包含generations字段")

def main(sample_size=1000, population_size=200, generations=300):
    """主函数，运行所有数据集的实验"""
    # 指定要运行的数据集
    datasets=["I.6.2","I.6.2b","I.12.4","I.14.3","I.14.4","I.25.13"]
    
    print("开始批量实验...")
    print(f"数据集: {datasets}")
    print(f"超参数 - 样本数: {sample_size}, 种群大小: {population_size}, 迭代代数: {generations}")
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset}")
        print(f"{'='*60}")
        
        # 运行多次实验
        results = run_multiple_experiments(
            dataset_name=dataset,
            sample_size=sample_size,
            population_size=population_size,
            generations=generations,
            num_runs=10
        )
        
        # 保存结果
        save_results(results, dataset)
        
        # 存储到总结果中
        all_results[dataset] = results
    
    print(f"\n{'='*60}")
    print("所有实验完成！")
    print(f"{'='*60}")
    
    # 打印总结
    print("\n实验总结:")
    for dataset, results in all_results.items():
        stats = results['test_rmse_stats']
        print(f"{dataset}: 平均RMSE={stats['mean']:.6f} (±{stats['std']:.6f}), "
              f"范围[{stats['min']:.6f}, {stats['max']:.6f}]")
    
    return all_results

if __name__ == "__main__":
    # 可以通过修改这些参数来调试不同的超参数设置
    results = main(
        sample_size=100,    # 每个数据集使用的样本数
        population_size=200, # 种群个体数
        generations=300      # 迭代代数
    )