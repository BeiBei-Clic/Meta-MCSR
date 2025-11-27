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
    # CPU / CUDA
    if not params.cpu:
        params.device = 'cuda'
        assert torch.cuda.is_available()
    else:
        params.device = 'cpu'
    snip.utils.CUDA = not params.cpu

    # Set missing parameters
    if not hasattr(params, 'global_rank'):
        params.global_rank = 0
    if not hasattr(params, 'local_rank'):
        params.local_rank = -1
    if not hasattr(params, 'master_port'):
        params.master_port = -1
    if not hasattr(params, 'n_gpu_per_node'):
        params.n_gpu_per_node = 1
    if not hasattr(params, 'n_steps_per_epoch'):
        params.n_steps_per_epoch = 1000

    env = build_env(params)
    modules = build_modules(env, params)
    similarity_calculator = SimilarityCalculator(modules, env, params)

    # Calculate similarity matrix
    task = params.tasks[0]  # Use first task

    # Display dataset information
    print(f"- Task: {task}")
    print(f"- Eval data path: {params.eval_data}")
    print(f"- Max input dimension: {params.max_input_dimension}")
    print(f"- Max output dimension: {params.max_output_dimension}")

    similarity_matrix = similarity_calculator.enc_dec_step(task, data_path={task: [params.eval_data]})
    print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
    print("Similarity matrix:")
    print(similarity_matrix)

    return similarity_matrix



if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # Set required defaults
    if not hasattr(params, 'is_slurm_job'):
        params.is_slurm_job = False
    if not hasattr(params, 'env_base_seed'):
        params.env_base_seed = -1

    # handle model reloading
    if params.eval_only and params.eval_from_exp != "":
        if os.path.isdir(params.eval_from_exp):
            checkpoint_path = params.eval_from_exp + "/checkpoint.pth"
            if os.path.exists(checkpoint_path):
                params.reload_checkpoint = checkpoint_path
            else:
                raise FileNotFoundError(f"No checkpoint found in {params.eval_from_exp}")
        elif os.path.isfile(params.eval_from_exp):
            params.reload_model = params.eval_from_exp
        else:
            raise FileNotFoundError(f"Model path not found: {params.eval_from_exp}")

        # read params from pickle
        if os.path.isdir(params.eval_from_exp):
            pickle_file = params.eval_from_exp + "/params.pkl"
            if os.path.exists(pickle_file):
                pk = pickle.load(open(pickle_file, "rb"))
                pickled_args = pk.__dict__
                del pickled_args["exp_id"]
                for p in params.__dict__:
                    if p in pickled_args:
                        params.__dict__[p] = pickled_args[p]

    # check parameters and run
    check_model_params(params)
    main(params)
