<!--版权所有2023年HuggingFace团队。保留所有权利。

根据Apache License，Version 2.0（“许可证”）许可使用此文件；除非符合许可证的规定，否则您不能使用此文件。
您可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，没有任何明示或暗示的担保或条件。请参阅许可证以获取许可证下的特定语言和限制。-->

# VitMatte

## 概述

[VitMatte](https://arxiv.org/abs/2305.15272)模型由Jingfeng Yao，Xinggang Wang，Shusheng Yang和Baoyuan Wang在文章《使用预训练的普通视觉变换器增强图像抠图》中提出。
VitMatte利用普通[视觉变换器](vit)来进行图像抠图任务，即准确估计图像和视频中的前景对象。

论文摘要如下：

*最近，普通视觉变换器（ViTs）在各种计算机视觉任务中展示了令人印象深刻的性能，这要归功于它们强大的建模能力和大规模预训练。然而，它们尚未解决图像抠图问题。我们假设ViTs也可以提升图像抠图，并提出了一种新的高效而稳健的基于ViT的抠图系统，命名为ViTMatte。我们的方法利用了以下特点：（i）混合注意机制与卷积脖领相结合，帮助ViTs在抠图任务中实现出色的性能与计算权衡。（ii）此外，我们引入了详细捕捉模块，该模块仅由简单的轻量级卷积组成，以补充抠图所需的详细信息。据我们所知，ViTMatte是第一项在图像抠图中释放ViT潜力的工作，其具有简洁的适应性，包括各种预训练策略，简洁的架构设计和灵活的推理策略。我们在Composition-1k和Distinctions-646上评估了ViTMatte，这是图像抠图最常用的基准，在大幅度超过先前的抠图工作上实现了最先进的性能。*

提示：

- 该模型预期图像和修剪图（连接在一起）作为输入。可以使用[`ViTMatteImageProcessor`]来实现此目的。

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[此处](https://github.com/hustvl/ViTMatte)找到。


## VitMatteConfig

[[autodoc]] VitMatteConfig

## VitMatteImageProcessor

[[autodoc]] VitMatteImageProcessor
    - 预处理

## VitMatteForImageMatting

[[autodoc]] VitMatteForImageMatting
    - 正向传播