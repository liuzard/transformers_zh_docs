<!--版权所有2022年的HuggingFace团队。

根据Apache License，Version 2.0（“许可证”）许可; 除非符合许可证的规定，否则您不得使用此文件。
您可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按
“现状”分发，不附带任何明示或暗示的保证或条件。有关的许可证限制
请参阅许可证中特定的语言。

⚠️ 请注意，此文件为Markdown格式，但包含特定于我们的文档生成器（类似于MDX）的语法，可能在您的Markdown查看器中无法正确渲染。

-->

# ConvNeXT

## 概述

ConvNeXT模型是由Zhuang Liu、Hanzi Mao、Chao-Yuan Wu、Christoph Feichtenhofer、Trevor Darrell和Saining Xie在《A ConvNet for the 2020s》论文中提出的。ConvNeXT是一种纯卷积模型（ConvNet），受到了Vision Transformers设计的启发，并声称胜过它们。

论文中的摘要如下：

*视觉识别的“热闹二十年”始于Vision Transformers（ViTs）的推出，它们很快取代了ConvNets成为最先进的图像分类模型。
然而，普通的ViT在应用到一般的计算机视觉任务（如物体检测和语义分割）时面临困难。正是分层Transformers（如Swin Transformers）重新引入了几个ConvNet的先验知识，使得Transformers在作为通用视觉骨干的实际可行性和在各种视觉任务上表现出色。
然而，这种混合方法的效果很大程度上仍然归功于Transformers本身的优越性，而不是卷积的固有归纳偏置。在这项工作中，我们重新审视设计空间并测试纯ConvNet可以实现的极限。
我们逐步将标准ResNet向视觉Transformer的设计方向“现代化”，并发现了一些在这个过程中对性能差异有贡献的关键组件。
这次探索的结果是一系列名为ConvNeXt的纯ConvNet模型。ConvNeXt完全由标准的ConvNet模块构建，与Transformers在精确性和可扩展性方面相比大胜，达到了87.8%的ImageNet top-1准确率，并在COCO检测和ADE20K分割上胜过Swin Transformers，同时保持了标准ConvNet的简单性和效率。*

提示：
- 查看每个模型下的代码示例以了解使用方法。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnext_architecture.jpg"
alt="drawing" width="600"/>

<small> ConvNeXT架构。摘自<a href="https://arxiv.org/abs/2201.03545">原始论文</a>。</small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。模型的TensorFlow版本由[ariG23498](https://github.com/ariG23498)、[gante](https://github.com/gante)和[sayakpaul](https://github.com/sayakpaul)（同等贡献）贡献。原始代码位于[此处](https://github.com/facebookresearch/ConvNeXt)。

## 资源

以下是官方Hugging Face和社区（通过🌎标识）的资源列表，帮助您快速开始使用ConvNeXT。

<PipelineTag pipeline="image-classification"/>

- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持`ConvNextForImageClassification`。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果您有兴趣提交一个资源以包含在这里，请随时发起拉取请求，我们将进行评估！资源理想情况下应该展示出一些新的内容，而不是重复现有资源。

## ConvNextConfig

[[autodoc]] ConvNextConfig

## ConvNextFeatureExtractor

[[autodoc]] ConvNextFeatureExtractor

## ConvNextImageProcessor

[[autodoc]] ConvNextImageProcessor
    - preprocess

## ConvNextModel

[[autodoc]] ConvNextModel
    - forward

## ConvNextForImageClassification

[[autodoc]] ConvNextForImageClassification
    - forward

## TFConvNextModel

[[autodoc]] TFConvNextModel
    - call

## TFConvNextForImageClassification

[[autodoc]] TFConvNextForImageClassification
    - call