# `SNIP`: 数学符号与数值统一预训练的多模态模型 (MathCLIP)

**ICLR 2024 Spotlight** 论文 **[SNIP: Bridging Mathematical Symbolic and Numeric Realms with Unified Pre-training](https://openreview.net/forum?id=KZSEgJGPxu)** 的官方实现。

[论文](https://openreview.net/forum?id=KZSEgJGPxu) | [模型](https://drive.google.com/drive/folders/1oGVQPAuTwWQnhX_pxN3OdKDt9-rmCfV3?usp=share_link) | [数据](https://github.com/deep-symbolic-mathematics/Multimodal-Math-Pretraining/blob/main/run_export_data.sh) | [结果](https://github.com/deep-symbolic-mathematics/Multimodal-Symbolic-Regression/tree/main/srbench_results)

## 概述
受到[CLIP](https://arxiv.org/abs/2310.02227)在视觉-语言表示学习中卓越表现的启发，我们引入了一个用于符号数学的多模态预训练模型，称为**SNIP**（**Symbolic-Numeric Integrated Pre-training**，符号-数值集成预训练），该模型强调了数值增强表示在数学表示学习中的重要性。

<p align="center">
<img src="./images/SNIP3.gif" width="100%" /> 
 <br>
<b>SNIP：一个多模态transformer模型，使用对比学习连接数学符号方程与数值数据表示</b>
</p>

## 安装
代码需要`environment.yml`中指定的依赖项。请按照相关库进行安装或运行：

```
conda env create -f environment.yml
```
此库需要`python>3.7`

## 预训练模型
我们已经发布了两个预训练的SNIP模型，每个模型都为不同类型的分析而设计。在[这里](https://drive.google.com/drive/folders/1oGVQPAuTwWQnhX_pxN3OdKDt9-rmCfV3?usp=share_link)下载它们。你会发现：

- **[SNIP-10dmax](https://drive.google.com/drive/u/1/folders/1oGVQPAuTwWQnhX_pxN3OdKDt9-rmCfV3):** 这个模型处理**最多10维输入**。更多信息见[论文](https://arxiv.org/pdf/2310.02227.pdf)第5节和附录D第3页。

- **[SNIP-1d-normalized](https://drive.google.com/drive/u/1/folders/1oGVQPAuTwWQnhX_pxN3OdKDt9-rmCfV3):** 这个模型用于**1维输入**和**标准化目标**，非常适合专注于函数模式。更多细节见[论文](https://arxiv.org/pdf/2310.02227.pdf)第4节和附录D。

要使用它们，在你的项目中创建一个`weights/`文件夹，在那里下载检查点，并使用带有模型路径的`--reload_model`参数，如`--reload_model ./weights/snip-1d-normalized.pth`。

## 预训练数据生成
对于预训练，我们基于[SymbolicMathematics](https://openreview.net/forum?id=S1eZYeHFDS)的方法生成数学函数的(符号，数值)对合成数据。每对包括数据点$(x,y)$和一个数学函数$f$，使得$y=f(x)$。更多信息请参见[这里](https://github.com/deep-symbolic-mathematics/Multimodal-Math-Pretraining/tree/main/snip/envs/generators.py)的`generate_datapoints`函数。你也可以在[这里](https://github.com/deep-symbolic-mathematics/Multimodal-Math-Pretraining/tree/main/snip/envs/environment.py)调整数据生成设置。

数据在训练期间即时生成，但如果你想提前创建和分析它，使用`run_export_data.sh`：
```
python train.py --export_data True --dump_path ./dump --max_input_dimension 10
```
你导出的数据将保存在`data.prefix`文件中。

## SNIP预训练
SNIP的所有训练设置都在`parsers.py`中。SNIP对符号和数值头都使用Transformer编码器，你可以在[这里](./snip/model/__init__.py)的`encoder_f`和`encoder_y`模块中找到它们。有关对比学习和训练的信息，请查看[训练器](./snip/trainer.py)文件。以下是开始训练的方法：
```
python train.py --loss_type CLIP \
                --batch_size 256 \
                --dump_path ./dump \
                --max_input_dimension 10 \
                --exp_id run1-10d \
                --lr 4e-5 \
                --latent_dim 512 \
                --save_periodic 10
```
你可以随时在`snip/envs/`下的`parsers.py`和`environment.py`中调整训练和数据设置。运行命令后，每10个(`save_periodic`)周期训练的模型保存在`dump/`路径中。

## 使用SNIP进行跨模态属性预测
我们提供了代码来测试SNIP表示在跨模态符号到数值属性预测任务上的性能，这意味着在这些任务中，输入是符号数学方程，标签是基于数值数据观察定义的属性。

### 数据生成
要尝试它，首先生成数据。例如，要为**非凸比(NCR)**预测任务生成10k训练示例（如[论文](https://arxiv.org/pdf/2310.02227.pdf)中所述），使用此命令：
```
python train.py --export_data True --is_proppred True --property_type ncr --dump_path ./dump --max_input_dimension 1 --n_steps_per_epoch 625  --exp_name data --exp_id ncr
```

这会将`ncr`属性的数据保存在`dump/data/ncr/`中。要为其他属性生成数据，只需更改`--property_type`参数。

### 训练
对于此任务，我们使用Transformer编码器架构（编码符号方程输入），后跟一个回归预测器头（预测属性）。训练使用均方误差(MSE)损失进行。以下是训练[论文](https://arxiv.org/pdf/2310.02227.pdf)第4节中定义的不同模型变体的命令。

**监督模型（无预训练）**：
```
python train.py --is_proppred True \
                --property_type ncr \
                --reload_data functions,dump/data/ncr/train.prefix,dump/data/ncr/train.prefix, \
                --normalize_y True \
                --batch_size 64 \
                --dump_path ./dump \
                --max_input_dimension 1 \
                --exp_name NCR_pred \
                --exp_id run1 \
                --lr 1e-5 \
                --latent_dim 512 \
                --save_periodic 10
```

**SNIP编码器（冻结）**：
```
python train.py --reload_model ./weights/snip-1d-normalized.pth --freeze_encoder True [其他参数] 
```

**SNIP编码器（微调）**：
```
python train.py --reload_model ./weights/snip-1d-normalized.pth --freeze_encoder False [其他参数] 
```

使用这些命令，模型每10个周期自动保存一次。要使用SNIP的编码器，你应该使用模型权重的路径激活`--reload_model`参数。你也可以使用`--freeze_encoder True`冻结编码器。

### 推理
要测试你的模型在每个属性预测任务上的表现如何，使用`run_eval_proppred.sh`脚本。例如，如果你想测试NCR属性任务，使用此命令：
```
python eval_proppred.py --is_proppred True \
                        --property_type ncr \
                        --reload_model dump/NCR/model.pth \
                        --reload_data functions,dump/data/ncr/test.prefix,dump/data/ncr/test.prefix,
```
此命令将使用`--reload_model`参数加载你训练模型的权重，并针对`--reload_data`路径中指定的数据集进行测试。

## 使用SNIP进行符号回归
要将SNIP用于更复杂的任务，如符号回归（从数据中发现符号数学方程：数值到符号生成任务），请查看**[Multimodal-Symbolic-Regression](https://github.com/deep-symbolic-mathematics/Multimodal-Symbolic-Regression)**存储库。

### SRBench上的最终结果
我们在SRBench数据集上进行符号回归的SNIP实验结果在[Multimodal-Symbolic-Regression](https://github.com/deep-symbolic-mathematics/Multimodal-Symbolic-Regression)存储库的[srbench_results/](https://github.com/deep-symbolic-mathematics/Multimodal-Symbolic-Regression/tree/main/srbench_results)目录中提供。这些结果共享以帮助研究社区重现我们论文的发现并作为参考基准。每个结果文件包含我们评估中使用的详细性能指标和实验配置。

## 引用
如果你发现论文或这个存储库有帮助，请引用它：
<pre>
@inproceedings{
meidani2024snip,
title={{SNIP}: Bridging Mathematical Symbolic and Numeric Realms with Unified Pre-training},
author={Kazem Meidani and Parshin Shojaee and Chandan K. Reddy and Amir Barati Farimani},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=KZSEgJGPxu}
}
</pre>

## 许可证
此存储库在MIT许可证下授权。

这项工作建立在其他开源项目之上，包括[Deep Learning for Symbolic Mathematics](https://github.com/facebookresearch/SymbolicMathematics)和[Contrastive Language-Image Pretraining](https://github.com/openai/CLIP)。我们感谢这些工作的原始贡献者开源他们宝贵的源代码。

## 联系我们
如有任何问题或问题，欢迎在此存储库中打开issue，或通过mmeidani@andrew.cmu.edu和parshinshojaee@vt.edu联系我们。