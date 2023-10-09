<!--版权2022 HuggingFace团队保留所有权利。

根据Apache许可证第2.0版（“许可证”）进行许可；除非符合许可证，否则您不能使用此文件。
您可以在以下位置获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是按"原样"分发的，不附带任何明示或暗示的担保或条件。详细了解许可证的特定语言，请参阅许可证

⚠️请注意，此文件采用Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），在您的Markdown查看器中可能无法正确呈现。-->

# ViTMSN

## 概述

ViTMSN模型由Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas在[Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141)中提出。该论文提出了一种联合嵌入架构，用于将掩蔽补丁的原型与未掩蔽补丁的原型进行匹配。在这种设置下，他们的方法在低样本和极低样本情况下获得了出色的性能。

论文中的摘要如下：

*我们提出了掩蔽孪生网络（MSN），这是一种自监督学习框架，用于学习图像表示。我们的方法将包含随机掩蔽补丁的图像视图的表示与原始未掩蔽图像的表示进行匹配。当应用于Vision Transformers时，这种自监督预训练策略尤其具有可伸缩性，因为网络只处理未掩蔽的补丁。结果，MSN提高了联合嵌入架构的可伸缩性，同时生成高语义级别的表示，在低样本图像分类上具有竞争力的性能。例如，在ImageNet-1K上，只有5000张标注图像时，我们的基本MSN模型达到了72.4%的top-1精度，当使用1%的ImageNet-1K标签时，我们达到了75.7%的top-1精度，这为自监督学习在这个基准测试中设立了新的技术水平。*

提示：

- MSN（掩蔽孪生网络）是自监督预训练Vision Transformers（ViTs）的方法。预训练目标是将未掩蔽视图的原型与相同图像的掩蔽视图的原型进行匹配。
- 作者只发布了骨干网络的预训练权重（ImageNet-1k预训练）。因此，要在自己的图像分类数据集上使用这个权重，使用[`ViTMSNForImageClassification`]类，该类是从[`ViTMSNModel`]初始化的。请参考[此笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb)详细了解微调的教程。
- MSN在低样本和极低样本情况下特别有用。值得注意的是，当进行微调时，它在只有1%的ImageNet-1K标签时达到了75.7%的top-1精度。

<img src="https://i.ibb.co/W6PQMdC/Screenshot-2022-09-13-at-9-08-40-AM.png" alt="drawing" width="600"/>

<small> MSN架构。取自<a href="https://arxiv.org/abs/2204.07141">原始论文。</a> </small>

此模型由[sayakpaul](https://huggingface.co/sayakpaul)贡献。原始代码可以在[此处](https://github.com/facebookresearch/msn)找到。

## 资源

下面是官方Hugging Face和社区（通过🌎标识）的资源列表，可帮助您开始使用ViT MSN。

<PipelineTag pipeline="image-classification"/>

- [`ViTMSNForImageClassification`]支持使用这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果您有兴趣提交要包含在此处的资源，请随时提出Pull Request，我们将进行审查！这个资源应该展示一些新的东西，而不是重复现有的资源。

## ViTMSNConfig

[[autodoc]] ViTMSNConfig

## ViTMSNModel

[[autodoc]] ViTMSNModel
    - 前向传播

## ViTMSNForImageClassification

[[autodoc]] ViTMSNForImageClassification
    - 前向传播