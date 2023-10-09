版权所有 © 2022 HuggingFace团队。

根据Apache许可证2.0版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以“按原样”（AS-IS）方式分发的软件，不附带任何明示或暗示的保证或条件。详细了解许可证下的特定语言条款和限制，请参阅许可证。

⚠️请注意，此文件是Markdown格式的，但包含特定的语法，是我们的文档生成器（类似于MDX）可能无法正确渲染的。

# VAN

<Tip warning={true}>


此模型只处于维护模式，因此我们不会接受任何更改其代码的新PR。

如果在运行此模型时遇到任何问题，请将其降级为上一个支持此模型的版本v4.30.0。可以通过运行以下命令实现：`pip install -U transformers==4.30.0`。

</Tip>

## 概述

VAN模型是由Meng-Hao Guo、Cheng-Ze Lu、Zheng-Ning Liu、Ming-Ming Cheng和Shi-Min Hu提出的[Visual Attention Network](https://arxiv.org/abs/2202.09741)中提出的。

该论文介绍了一种基于卷积操作的新型注意力层，能够捕捉局部和远程关系。这是通过组合正常和大内核卷积层实现的。后者使用扩张卷积来捕捉远程相关性。

论文中的摘要如下所示：

*尽管最初设计用于自然语言处理任务，但自注意机制最近在各个计算机视觉领域中取得了巨大的成功。然而，图像的二维性为在计算机视觉中应用自注意力机制带来了三个挑战。(1)将图像视为1D序列忽略了它们的2D结构。(2)二次复杂度对于高分辨率图像来说太昂贵。(3)它只捕捉了空间适应性，而忽视了通道适应性。在本文中，我们提出了一种新型的大内核注意力（LKA）模块，以在自我注意力中实现自适应和长程关系，同时避免上述问题。我们进一步引入了一种基于LKA的新型神经网络，即Visual Attention Network (VAN)。VAN非常简单，但在广泛的实验证明中，它在图像分类、目标检测、语义分割、实例分割等方面明显优于最先进的视觉变换器和卷积神经网络。代码可在[此网址](https://github.com/Visual-Attention-Network/VAN-Classification)上获得。*

提示：

- VAN没有嵌入层，因此`hidden_states`的长度等于阶段数。

下图显示了Visual Attention Layer的架构。摘自[原论文](https://arxiv.org/abs/2202.09741)。

![VAN架构图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/van_architecture.png)

此模型由[Francesco](https://huggingface.co/Francesco)贡献。原始代码可在[此处](https://github.com/Visual-Attention-Network/VAN-Classification)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）资源的列表，可帮助您开始使用VAN。

<PipelineTag pipeline="image-classification"/>

- [`VanForImageClassification`]受到此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)的支持。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果您有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将对其进行审核！该资源应该展示出一些新内容，而不是重复现有的资源。

## VanConfig

[[autodoc]] VanConfig


## VanModel

[[autodoc]] VanModel
    - forward


## VanForImageClassification

[[autodoc]] VanForImageClassification
    - forward