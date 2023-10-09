<!--版权所有2022 The HuggingFace团队。保留所有权利。

根据Apache License，Version 2.0（“许可证”）许可; 除非符合许可证的规定，否则您不得使用此文件。
您可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或以书面形式同意，根据许可证分发的软件是基于“原样”的，
没有任何明示或暗示的担保或条件。参见许可证中的特定语言以及许可证下的限制。

⚠️ 请注意，此文件是Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），可能在您的Markdown查看器中无法正确显示。

-->

# RegNet

## 概述

RegNet模型是由Ilija Radosavovic、Raj Prateek Kosaraju、Ross Girshick、Kaiming He和Piotr Dollár在[《Designing Network Design Spaces》](https://arxiv.org/abs/2003.13678)中提出的。
作者设计了搜索空间以进行神经网络架构搜索（NAS）。他们首先从高维搜索空间开始，通过根据当前搜索空间中采样的表现最佳模型的经验性约束来迭代地减小搜索空间。

论文中的摘要如下：

*在这项工作中，我们提出了一种新的网络设计范式。我们的目标是推进对网络设计的理解，并发现适用于各种设置的设计原则。我们不专注于设计单个网络实例，而是设计将网络总体设计空间参数化的网络设计空间。这个整个过程类似于经典手动设计网络，但升级到了设计空间级别。使用我们的方法，我们探索了网络设计的结构方面，并得出了一个由简单、规则网络组成的低维设计空间，我们称之为RegNet。RegNet参数化的核心见解非常简单：优秀网络的宽度和深度可以用量化的线性函数解释。我们分析RegNet设计空间并得出了有趣的发现，这些发现与当前的网络设计实践不符。在可比的训练设置和计算量下，RegNet模型在GPU上的性能优于流行的EfficientNet模型，同时速度最高可提高5倍。*

提示：

- 您可以使用[`AutoImageProcessor`](https://huggingface.co/docs/datasets/package_reference/main_classes/hf_datasets.transforms.AutoImageProcessor.html)为模型准备图像。
- 来自[《Self-supervised Pretraining of Visual Features in the Wild》](https://arxiv.org/abs/2103.01988)的巨大的10B模型，它在10亿个Instagram图像上进行了训练，可在[huggingface.co](https://huggingface.co/facebook/regnet-y-10b-seer)上获取。

此模型由[Francesco](https://huggingface.co/Francesco)贡献。模型的TensorFlow版本由[sayakpaul](https://huggingface.com/sayakpaul)和[ariG23498](https://huggingface.com/ariG23498)贡献。
原始代码可在[此处](https://github.com/facebookresearch/pycls)找到。

## 资源

以下是官方Hugging Face资源和社区（标有🌎）资源的列表，可帮助您开始使用RegNet。

<PipelineTag pipeline="image-classification"/>

- [`RegNetForImageClassification`](https://huggingface.co/models?pipeline_tag=image-classification)的使用示例可以在[此处](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)找到。
- 参阅：[图像分类任务指南](../tasks/image_classification)

如果您有兴趣提交资源以包含在这里，请随时打开拉取请求，我们将进行审核！资源应该展示出新的东西，而不是重复现有的资源。

## RegNetConfig

[[autodoc]] RegNetConfig


## RegNetModel

[[autodoc]] RegNetModel
    - forward


## RegNetForImageClassification

[[autodoc]] RegNetForImageClassification
    - forward

## TFRegNetModel

[[autodoc]] TFRegNetModel
    - call


## TFRegNetForImageClassification

[[autodoc]] TFRegNetForImageClassification
    - call


## FlaxRegNetModel

[[autodoc]] FlaxRegNetModel
    - __call__


## FlaxRegNetForImageClassification

[[autodoc]] FlaxRegNetForImageClassification
    - __call__