<!--
版权© 2022 HuggingFace团队。版权所有。

根据Apache许可证第2版（“许可证”），除非符合许可证的规定，否则你不得使用此文件。
你可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件分发在"根据现状"的基础上，
无论是明示的还是暗示的，都没有任何形式的保证或条件。有关详细信息，请参阅许可证中的特定语言。

⚠️ 请注意，此文件是Markdown格式，但包含我们doc-builder（类似于MDX的）的特定语法，
可能无法在你的Markdown查看器中正确呈现。

-->

# TimeSformer

## 概述

TimeSformer模型是由Facebook Research在[TimeSformer：时空注意力是否就是你对视频理解所需要的全部？](https://arxiv.org/abs/2102.05095)一文中提出的。
这项工作是行动识别领域的里程碑，是第一个基于转换器（Transformer）的视频模型。它启发了许多基于转换器的视频理解和分类论文。

论文摘要如下所示：

*我们提出了一种基于自注意力模型的无卷积视频分类方法，它在空间和时间上进行。我们的方法称为“TimeSformer”，通过直接从一系列帧级补丁中学习时空特征，使标准转换器体系结构适应于视频。我们的实验研究比较了不同的自注意力方案，并表明在考虑的设计选择中，“分割注意力”（即分别在每个块内应用时间注意力和空间注意力）可以获得最佳的视频分类准确性。尽管设计与传统方法截然不同，但TimeSformer在多个行动识别基准数据集上取得了最先进的结果，包括Kinetics-400和Kinetics-600的最高报告准确性。最后，与3D卷积网络相比，我们的模型训练速度更快，在测试效率方面可以取得显著的提高（准确率稍有下降），并且还可以应用于更长的视频片段（超过一分钟）。源代码和模型可在以下网址获取：[this https URL](https://github.com/facebookresearch/TimeSformer)。*

提示：

有许多预训练的变体。根据其训练的数据集选择预训练模型。此外，每个剪辑的输入帧数会根据模型大小而改变，因此在选择预训练模型时应考虑此参数。

此模型由[fcakyon](https://huggingface.co/fcakyon)贡献。
原始代码可在[此处](https://github.com/facebookresearch/TimeSformer)找到。

## 文档资源

- [视频分类任务指南](../tasks/video_classification)

## TimesformerConfig

[[autodoc]] TimesformerConfig

## TimesformerModel

[[autodoc]] TimesformerModel
    - forward

## TimesformerForVideoClassification

[[autodoc]] TimesformerForVideoClassification
    - forward
-->