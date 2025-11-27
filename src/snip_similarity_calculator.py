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

    def enc_dec_step(self, task, data_path=None):
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
            # For evaluation, create dataset directly with train=False
            from snip.envs.environment import EnvDataset
            dataset = EnvDataset(
                self.env,
                task,
                train=False,  # Use False for evaluation
                skip=False,
                params=self.params,
                path=data_path[task][0] if data_path else None,
                type="eval",  # Set type to avoid NoneType error
            )

            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.params.batch_size_eval if hasattr(self.params, 'batch_size_eval') else self.params.batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )

            samples, errors = next(iter(dataloader))

            # Display raw data before processing
            print(f"\nRaw sample information:")
            print(f"- Keys in samples: {list(samples.keys())}")
            if 'infos' in samples:
                print(f"- Sample infos: {samples['infos']}")
            if 'tree' in samples:
                print(f"- Tree structure: {samples['tree'][:1]}")
            if 'tree_encoded' in samples:
                print(f"- Tree encoded sample: {samples['tree_encoded'][:1]}")

            x_to_fit = samples["x_to_fit"]
            y_to_fit = samples["y_to_fit"]

            # Display data loading information
            print(f"\nData loading details:")
            print(f"- Number of samples: {len(x_to_fit)}")
            print(f"- First sample x shape: {x_to_fit[0].shape}")
            print(f"- First sample y shape: {y_to_fit[0].shape}")
            print(f"- First sample x[:2]: {x_to_fit[0][:2]}")
            print(f"- First sample y[:2]: {y_to_fit[0][:2]}")

            # prepare input for embedder
            x1 = []
            print(f"\nPreparing embedder input:")
            for seq_id in range(len(x_to_fit)):
                x1.append([])
                print(f"  Sample {seq_id}: x_to_fit shape = {x_to_fit[seq_id].shape}, y_to_fit shape = {y_to_fit[seq_id].shape}")
                for seq_l in range(len(x_to_fit[seq_id])):
                    x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])
                print(f"    After processing: x1[{seq_id}] length = {len(x1[seq_id])}, each element shape = {len(x1[seq_id][0])}")

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