<!--版权所有2021 The HuggingFace团队。保留所有权利。

根据Apache License, Version 2.0（“许可证”）进行许可；除非符合许可证的规定
你可能不使用此文件。你可以在以下位置获取许可证副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件都是
基于“AS IS” BASIS，不附带任何担保或条件。有关许可证下的
特定语言管理权限和限制。-->

# ImageGPT

## 概述

ImageGPT模型在马克·陈（Mark Chen）、亚历克·拉德福德（Alec Radford）、Rewon Child、Jeffrey Wu、Heewoo Jun、大卫·鲁安（David Luan）和Ilya Sutskever的[像素生成预训练](https://openai.com/blog/image-gpt)一文中提出。ImageGPT（iGPT）是一个类似于GPT-2的模型，用于预测下一个像素值，可以进行无条件和有条件的图像生成。

论文中的摘要如下：

*受到自然语言无监督表示学习的进展的启发，我们研究类似的模型能否为图像学习有用的表示。我们训练一个序列Transformer来自回归地预测像素，而不涉及2D输入结构的知识。尽管在没有标签的低分辨率ImageNet上进行训练，但我们发现一个GPT-2规模的模型通过线性探测、微调和低数据分类的度量学习到了强大的图像表示。在CIFAR-10上，我们使用线性探测实现了96.3%的准确率，超过了一个有监督的Wide ResNet，并且使用完全微调实现了99.0%的准确率，与顶级的有监督预训练模型相匹配。在使用像素替代VQVAE编码的ImageNet上，我们与自监督基准方法相竞争，在线性探测中实现了69.0%的top-1准确率。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/imagegpt_architecture.png"
alt="绘图" width="600"/>

<small> 方法概述。取自[原论文](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)。 </small>

此模型由[nielsr](https://huggingface.co/nielsr)贡献，基于[这个问题](https://github.com/openai/image-gpt/issues/7)。原始代码可在此处找到
[here](https://github.com/openai/image-gpt)。

提示:

- ImageGPT与[GPT-2](gpt2)几乎完全相同，唯一的区别是使用了不同的激活函数（即“quick gelu”），并且层归一化层不对输入进行均值中心化。ImageGPT
  也没有输入和输出嵌入的绑定。
- 由于Transformer的注意机制的时间和内存需求与序列长度呈二次关系，作者在较小的输入分辨率（例如32x32和64x64）上预训练了ImageGPT。然而，将大小为32x32x3=3072的序列从0到255的范围的整数输入到Transformer仍然是不可行的。因此，作者对（R，G，B）像素值应用了k-means聚类，其中k = 512。这样，我们只有一个长度为32*32 = 1024的序列，但是现在是0..511范围内的整数。因此，我们通过增大嵌入矩阵的开销来缩小序列长度。换句话说，ImageGPT的词汇表大小为512，+ 1表示特殊的“句子开始”（SOS）令牌，在每个序列的开头使用。你可以使用[`ImageGPTImageProcessor`]来准备
  图像以供模型使用。
- 尽管ImageGPT完全无监督地进行预训练（即没有使用任何标签），但它产生了对下游任务（如图像分类）有用的性能良好的图像特征。作者表明，网络中间的特征最为有效，并且可以直接用作训练线性模型的输入（例如sklearn的逻辑回归模型）。这也被称为“线性探测”。可以通过首先将图像输入模型，然后指定`output_hidden_states=True`，然后在所需的任何层上对隐藏状态进行平均池化来轻松获取特征。
- 另外，可以类似于BERT的方式对整个模型进行进一步的下游数据集微调。为此，可以使用[`ImageGPTForImageClassification`]。
- ImageGPT有不同的大小：有ImageGPT-small、ImageGPT-medium和ImageGPT-large。作者还训练了一个XL变体，但未发布。下表总结了这些大小的区别：

| **模型变体** | **深度** | **隐藏大小** | **解码器隐藏大小** | **参数（M）** | **ImageNet-1k Top 1** |
|---|---|---|---|---|---|
| MiT-b0 | [2, 2, 2, 2] | [32, 64, 160, 256] | 256 | 3.7 | 70.5 |
| MiT-b1 | [2, 2, 2, 2] | [64, 128, 320, 512] | 256 | 14.0 | 78.7 |
| MiT-b2 | [3, 4, 6, 3] | [64, 128, 320, 512] | 768 | 25.4 | 81.6 |
| MiT-b3 | [3, 4, 18, 3] | [64, 128, 320, 512] | 768 | 45.2 | 83.1 |
| MiT-b4 | [3, 8, 27, 3] | [64, 128, 320, 512] | 768 | 62.6 | 83.6 |
| MiT-b5 | [3, 6, 40, 3] | [64, 128, 320, 512] | 768 | 82.0 | 83.8 |

## 资源

这是一个官方Hugging Face和社区（由🌎表示）资源列表，以帮助你开始使用ImageGPT。

<PipelineTag pipeline="image-classification"/>

- ImageGPT的演示笔记本可以在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ImageGPT)找到。
- 通过此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持[`ImageGPTForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果你有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将对其进行审查！资源应该尽量展示新内容，而不是重复现有资源。

## ImageGPTConfig

[[autodoc]] ImageGPTConfig

## ImageGPTFeatureExtractor

[[autodoc]] ImageGPTFeatureExtractor

    - __call__

## ImageGPTImageProcessor

[[autodoc]] ImageGPTImageProcessor
    - 预处理

## ImageGPTModel

[[autodoc]] ImageGPTModel

    - forward

## ImageGPTForCausalImageModeling

[[autodoc]] ImageGPTForCausalImageModeling

    - forward

## ImageGPTForImageClassification

[[autodoc]] ImageGPTForImageClassification

    - forward