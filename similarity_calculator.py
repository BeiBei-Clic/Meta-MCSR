import random
import numpy as np
import torch
import os
import pickle
from pathlib import Path

import snip
from snip.slurm import init_signal_handler, init_distributed_mode
from snip.utils import bool_flag, initialize_exp
from snip.model import check_model_params, build_modules
from snip.envs import build_env
from src.snip_similarity_calculator import SimilarityCalculator
from parsers import get_parser

np.seterr(all="raise")

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def main(params):

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.cpu:
        params.device = 'cuda'
        assert torch.cuda.is_available()
    else:
        params.device = 'cpu'
    snip.utils.CUDA = not params.cpu

    env = build_env(params)

    modules = build_modules(env, params)
    similarity_calculator = SimilarityCalculator(modules, env, params)

    for task_id in np.random.permutation(len(params.tasks)):
        task = params.tasks[task_id]
        loss = similarity_calculator.enc_dec_step(task)
        print(loss)
        similarity_calculator.iter()



if __name__ == "__main__":

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    if params.eval_only and params.eval_from_exp != "":
        if os.path.isdir(params.eval_from_exp):
            # eval_from_exp is a directory, look for standard checkpoint files
            if os.path.exists(
                params.eval_from_exp + "/best-" + params.validation_metrics + ".pth"
            ):
                params.reload_model = (
                    params.eval_from_exp + "/best-" + params.validation_metrics + ".pth"
                )
            elif os.path.exists(params.eval_from_exp + "/checkpoint.pth"):
                params.reload_model = params.eval_from_exp + "/checkpoint.pth"
            else:
                raise NotImplementedError
        elif os.path.isfile(params.eval_from_exp):
            # eval_from_exp is a direct model file
            params.reload_model = params.eval_from_exp
        else:
            raise NotImplementedError

        eval_data = params.eval_data

        # read params from pickle only if eval_from_exp is a directory
        if os.path.isdir(params.eval_from_exp):
            pickle_file = params.eval_from_exp + "/params.pkl"
            assert os.path.isfile(pickle_file)
            pk = pickle.load(open(pickle_file, "rb"))
            pickled_args = pk.__dict__
            del pickled_args["exp_id"]
            for p in params.__dict__:
                if p in pickled_args:
                    params.__dict__[p] = pickled_args[p]

        params.eval_size = None
        if params.reload_data or params.eval_data:
            params.reload_data = (
                params.tasks + "," + eval_data + "," + eval_data + "," + eval_data
            )
        params.is_slurm_job = False
        params.local_rank = -1
        params.master_port = -1

    # check parameters
    check_model_params(params)

    # run experiment
    main(params)
