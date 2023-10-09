<!--版权所有2022 The HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）进行许可；除非符合许可证的规定，否则你不得使用此文件。
你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，
不附带任何明示或暗示的担保或条件。有关许可证下的特定语言，请参阅许可证。

⚠️注意，此文件是以Markdown格式编写的，但包含特定的语法，
用于我们的doc-builder（类似于MDX），可能无法在你的Markdown查看器中正确呈现。

-->

# 大型转移（BiT）

## 概述

BiT模型由Alexander Kolesnikov，Lucas Beyer，Xiaohua Zhai，Joan Puigcerver，Jessica Yung，Sylvain Gelly，Neil Houlsby在《Big Transfer (BiT): General Visual Representation Learning》（https://arxiv.org/abs/1912.11370）中提出。
BiT是一种简单的方法，用于扩展[ResNet](resnet)-like架构（具体而言，是ResNetv2）的预训练。该方法在迁移学习方面取得了显著的改进。

论文中的摘要如下：

*通过转移预训练表示提高样本效率并简化深度神经网络视觉训练中的超参数调整。我们回顾了在大型有监督数据集上进行预训练并对模型进行微调的范式。我们扩展了预训练，并提出了一个简单的方法，称为Big Transfer（BiT）。通过组合几个精心选择的组件，并使用简单的启发式方法进行转移，我们在20多个数据集上取得了强大的性能。BiT在各种数据范围上表现良好，从每类1个示例到总共100万个示例。BiT在ILSVRC-2012上达到了87.5%的top-1准确率，在CIFAR-10上达到了99.4%的准确率，在19个任务的视觉任务自适应基准(VTAB)上达到了76.3%的准确率。在小型数据集上，BiT在ILSVRC-2012上达到了76.8%的准确率（每类10个示例），在CIFAR-10上达到了97.0%的准确率（每类10个示例）。我们对导致高转移性能的主要组件进行了详细分析。*

提示：

- BiT模型在架构方面等效于ResNetv2，唯一的区别是：1）所有批归一化层都被[组归一化](https://arxiv.org/abs/1803.08494)取代，
2）[权重标准化](https://arxiv.org/abs/1903.10520)用于卷积层。作者表明，两者的结合对于使用大批量大小进行训练非常有用，并对迁移学习产生了重要影响。

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[此处](https://github.com/google-research/big_transfer)找到。

## 资源

以下是官方Hugging Face和社区（用🌎表示）资源列表，可帮助你开始使用BiT。

<PipelineTag pipeline="image-classification"/>

- 该[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 支持使用 `BitForImageClassification` 进行图像分类。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果你有兴趣提交资源以包含在此处，请随时打开Pull请求，我们将对其进行审核！该资源应该理想地展示一些新的东西，而不是重复现有的资源。

## BitConfig

[[autodoc]] BitConfig

## BitImageProcessor

[[autodoc]] BitImageProcessor
    - preprocess

## BitModel

[[autodoc]] BitModel
    - forward

## BitForImageClassification

[[autodoc]] BitForImageClassification
    - forward
