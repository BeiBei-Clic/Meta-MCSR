import os
import torch
from torch import nn
from snip.utils import to_cuda
from logging import getLogger

logger = getLogger()


class SimilarityCalculator(object):
    def __init__(self, modules, env, params):
        """
        Initialize similarity calculator.
        """
        self.modules = modules
        self.params = params
        self.env = env

        # load pretrained model if specified
        if self.params.reload_model != "":
            self.reload_model()

        # load checkpoint if specified
        if hasattr(self.params, 'reload_checkpoint') and self.params.reload_checkpoint != "":
            self.reload_checkpoint()

    def reload_model(self):
        """
        Reload a pretrained model.
        """
        if self.params.reload_model != "":
            model_path = self.params.reload_model
            assert os.path.isfile(model_path), f"Model file not found: {model_path}"

        logger.warning(f"Reloading pretrained model from {model_path} ...")
        data = torch.load(model_path, map_location="cpu", weights_only=False)

        modules_to_load = ['embedder', 'encoder_y', 'encoder_f']
        if hasattr(self.params, 'is_proppred') and self.params.is_proppred:
            if self.params.property_type in ['ncr', 'upward', 'yavg', 'oscil']:
                modules_to_load = ['encoder_f']  # symbolic encoder for numeric properties
            else:
                modules_to_load = ['embedder', 'encoder_y']  # numeric encoder for symbolic properties

        for k in modules_to_load:
            if k in self.modules:
                v = self.modules[k]
                weights = data[k]
                try:
                    v.load_state_dict(weights)
                except RuntimeError:  # remove the 'module.' prefix
                    weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                    v.load_state_dict(weights)
                logger.info(f"Loaded {k} from {model_path}")

    def reload_checkpoint(self):
        """
        Reload a checkpoint.
        """
        checkpoint_path = self.params.reload_checkpoint
        assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        for k, v in self.modules.items():
            if k in data:
                weights = data[k]
                try:
                    v.load_state_dict(weights)
                except RuntimeError:  # remove the 'module.' prefix
                    weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                    v.load_state_dict(weights)
                logger.info(f"Loaded {k} from checkpoint")

    def enc_dec_step(self, task):
        """
        Encoding step to compute similarity matrix.
        Returns the similarity matrix between encoded symbolic and numeric representations.
        """
        embedder, encoder_y, encoder_f = (
            self.modules["embedder"],
            self.modules["encoder_y"],
            self.modules["encoder_f"],
        )

        embedder.eval()
        encoder_y.eval()
        encoder_f.eval()

        with torch.no_grad():
            # get data batch
            dataloader = self.env.create_train_iterator(task, None, self.params)
            samples, errors = next(iter(dataloader))

            x_to_fit = samples["x_to_fit"]
            y_to_fit = samples["y_to_fit"]

            # prepare input for embedder
            x1 = []
            for seq_id in range(len(x_to_fit)):
                x1.append([])
                for seq_l in range(len(x_to_fit[seq_id])):
                    x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])

            # embed the sequences
            x1, len1 = embedder(x1)

            # prepare skeleton/tree input
            if hasattr(self.params, 'use_skeleton') and self.params.use_skeleton:
                x2, len2 = self.env.batch_equations(
                    self.env.word_to_idx(
                        samples["skeleton_tree_encoded"], float_input=False
                    )
                )
            else:
                x2, len2 = self.env.batch_equations(
                    self.env.word_to_idx(samples["tree_encoded"], float_input=False)
                )

            x2, len2 = to_cuda(x2, len2)

            # encode both representations
            encoded_y = encoder_y("fwd", x=x1, lengths=len1, causal=False)  # bx512
            encoded_f = encoder_f("fwd", x=x2, lengths=len2, causal=False)  # bx512

            # compute similarity matrix
            similarity_matrix = encoded_f @ encoded_y.T

            return similarity_matrix