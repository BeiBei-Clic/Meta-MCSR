import numpy as np
import torch
import os
import pickle

import snip
from snip.model import check_model_params, build_modules
from snip.envs import build_env
from src.snip_similarity_calculator import SimilarityCalculator
from parsers import get_parser

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def main(params):
    # 设置设备
    params.device = 'cuda' if not params.cpu and torch.cuda.is_available() else 'cpu'
    snip.utils.CUDA = not params.cpu

    # 设置默认参数
    for attr, default in [('global_rank', 0), ('local_rank', -1), ('master_port', -1),
                          ('n_gpu_per_node', 1), ('n_steps_per_epoch', 1000)]:
        if not hasattr(params, attr):
            setattr(params, attr, default)

    env = build_env(params)
    modules = build_modules(env, params)
    similarity_calculator = SimilarityCalculator(modules, env, params)

    # 计算相似度矩阵
    task = params.tasks[0]

    # 显示数据集信息
    print(f"- 任务: {task}")
    print(f"- 评估数据路径: {params.eval_data}")
    print(f"- 最大输入维度: {params.max_input_dimension}")
    print(f"- 最大输出维度: {params.max_output_dimension}")

    similarity_matrix = similarity_calculator.enc_dec_step(task, data_path={task: [params.eval_data]})
    print(f"\n相似度矩阵形状: {similarity_matrix.shape}")
    print("相似度矩阵:")
    print(similarity_matrix)

    return similarity_matrix



if __name__ == "__main__":
    # 生成解析器并解析参数
    parser = get_parser()
    params = parser.parse_args()

    # 设置必需的默认值
    params.is_slurm_job = getattr(params, 'is_slurm_job', False)
    params.env_base_seed = getattr(params, 'env_base_seed', -1)

    # 处理模型重载
    if params.eval_only and params.eval_from_exp:
        if os.path.isdir(params.eval_from_exp):
            checkpoint_path = f"{params.eval_from_exp}/checkpoint.pth"
            if os.path.exists(checkpoint_path):
                params.reload_checkpoint = checkpoint_path
            else:
                params.reload_model = params.eval_from_exp
        elif os.path.isfile(params.eval_from_exp):
            params.reload_model = params.eval_from_exp

        # 从pickle读取参数
        if os.path.isdir(params.eval_from_exp):
            pickle_file = f"{params.eval_from_exp}/params.pkl"
            if os.path.exists(pickle_file):
                pk = pickle.load(open(pickle_file, "rb"))
                pickled_args = pk.__dict__
                del pickled_args["exp_id"]
                for p in params.__dict__:
                    if p in pickled_args:
                        params.__dict__[p] = pickled_args[p]

    # 检查参数并运行
    check_model_params(params)
    main(params)
