import os
import torch
from snip.utils import to_cuda
from logging import getLogger

logger = getLogger()


class SimilarityCalculator(object):
    def __init__(self, modules, env, params):
        self.modules = modules
        self.params = params
        self.env = env

        # 加载预训练模型
        if self.params.reload_model:
            self.reload_model()

        # 加载检查点
        if hasattr(self.params, 'reload_checkpoint') and self.params.reload_checkpoint:
            self.reload_checkpoint()

    def reload_model(self):
        model_path = self.params.reload_model
        logger.warning(f"从 {model_path} 重载预训练模型...")
        data = torch.load(model_path, map_location="cpu", weights_only=False)

        modules_to_load = ['embedder', 'encoder_y', 'encoder_f']
        if hasattr(self.params, 'is_proppred') and self.params.is_proppred:
            if self.params.property_type in ['ncr', 'upward', 'yavg', 'oscil']:
                modules_to_load = ['encoder_f']  # 数值属性的符号编码器
            else:
                modules_to_load = ['embedder', 'encoder_y']  # 符号属性的数值编码器

        for k in modules_to_load:
            if k in self.modules:
                v = self.modules[k]
                weights = data[k]
                try:
                    v.load_state_dict(weights)
                except RuntimeError:  # 移除'module.'前缀
                    weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                    v.load_state_dict(weights)
                logger.info(f"从 {model_path} 加载 {k}")

    def reload_checkpoint(self):
        checkpoint_path = self.params.reload_checkpoint
        logger.warning(f"从 {checkpoint_path} 重载检查点...")
        data = torch.load(checkpoint_path, map_location="cpu")

        # 重载模型参数
        for k, v in self.modules.items():
            if k in data:
                weights = data[k]
                try:
                    v.load_state_dict(weights)
                except RuntimeError:  # 移除'module.'前缀
                    weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                    v.load_state_dict(weights)
                logger.info(f"从检查点加载 {k}")

    def enc_dec_step(self, task, data_path=None):
        # 编码步骤计算相似度矩阵
        embedder, encoder_y, encoder_f = (
            self.modules["embedder"],
            self.modules["encoder_y"],
            self.modules["encoder_f"],
        )

        embedder.eval()
        encoder_y.eval()
        encoder_f.eval()

        with torch.no_grad():
            from snip.envs.environment import EnvDataset
            dataset = EnvDataset(
                self.env,
                task,
                train=False,
                skip=False,
                params=self.params,
                path=data_path[task][0] if data_path else None,
                type="eval",
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=getattr(self.params, 'batch_size_eval', self.params.batch_size),
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )

            samples, errors = next(iter(dataloader))

            x_to_fit = samples["x_to_fit"]
            y_to_fit = samples["y_to_fit"]

            # print(f"- 第一个样本 x[:2]: {x_to_fit[0][:2]}")
            # print(f"- 第一个样本 y[:2]: {y_to_fit[0][:2]}")
            # print(f"- 最后样本 x[:2]: {x_to_fit[-1][:2]}")
            # print(f"- 最后样本 y[:2]: {y_to_fit[-1][:2]}")

            # 准备嵌入器输入
            x1 = []
            print(f"\n准备嵌入器输入:")
            for seq_id in range(len(x_to_fit)):
                x1.append([])
                for seq_l in range(len(x_to_fit[seq_id])):
                    x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])

            x1, len1 = embedder(x1)

            # 准备骨架/树输入
            if hasattr(self.params, 'use_skeleton') and self.params.use_skeleton:
                x2, len2 = self.env.batch_equations(
                    self.env.word_to_idx(samples["skeleton_tree_encoded"], float_input=False)
                )
            else:
                x2, len2 = self.env.batch_equations(
                    self.env.word_to_idx(samples["tree_encoded"], float_input=False)
                )

            x2, len2 = to_cuda(x2, len2)

            # 编码两种表示
            encoded_y = encoder_y("fwd", x=x1, lengths=len1, causal=False)
            encoded_f = encoder_f("fwd", x=x2, lengths=len2, causal=False)

            # 计算相似度矩阵
            similarity_matrix = encoded_f @ encoded_y.T

            return similarity_matrix