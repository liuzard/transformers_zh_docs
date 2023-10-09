<!--版权所有2022 The HuggingFace团队。

根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证，否则不得使用此文件。
您可以在下面的网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则依照许可证分发的软件是基于“原样”分发的，无论明示或暗示，均不附带任何担保或条件。请详阅许可证中的详细信息和限制。

⚠️请注意，这个文件是以Markdown格式编写的，但是包含特定的语法，用于我们的文档构建器（类似于MDX），可能无法在您的Markdown查看器中正确显示。

-->

# 混合视觉变形器（ViT混合）

## 概述

混合视觉变形器（ViT）模型是由Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov、Dirk Weissenborn、Xiaohua Zhai、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly、Jakob Uszkoreit、Neil Houlsby在《一张图片值16×16个单词：用于图像识别的Transformer》一文中提出的。这是第一篇成功训练Transformer编码器在ImageNet上的论文，相较于常见的卷积架构，取得了非常好的结果。ViT混合是[普通视觉变形器](vit)的一种轻微变种，通过利用卷积主干（具体来说，[BiT](bit)）的特征作为Transformer的初始“标记”。

论文摘要如下：

*虽然Transformer架构已经成为自然语言处理任务的标准，但它在计算机视觉中的应用还相对有限。在视觉领域，注意力要么与卷积网络配合使用，要么用于替换卷积网络的某些组件，同时保持其整体结构。本文表明，这种对CNN的依赖是不必要的，直接将纯Transformer应用于图像块序列在图像分类任务上可以表现得很好。在大量数据上进行预训练，并迁移到多个中型或小型图像识别基准（ImageNet，CIFAR-100，VTAB等）上时，Vision Transformer（ViT）相对于最先进的卷积网络取得了出色的结果，而且训练时所需的计算资源要少得多。*

此模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码（使用JAX编写）可以在[这里](https://github.com/google-research/vision_transformer)找到。

## 资源

为帮助您开始使用ViT混合，下面是一些官方的Hugging Face和社区（由🌎表示）资源。

<PipelineTag pipeline="image-classification"/>

- 此项目支持[`ViTHybridForImageClassification`]，该示例脚本（[example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)）和 [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 示例。
- 参见：[图像分类任务指南](../tasks/image_classification)

如果您有兴趣提交资源以包含在此处，请随时发起拉取请求，我们将进行审核！该资源应该展示一些新的内容，而不是重复现有的资源。

## ViTHybridConfig

[[autodoc]] ViTHybridConfig

## ViTHybridImageProcessor

[[autodoc]] ViTHybridImageProcessor
    - preprocess

## ViTHybridModel

[[autodoc]] ViTHybridModel
    - forward

## ViTHybridForImageClassification

[[autodoc]] ViTHybridForImageClassification
    - forward