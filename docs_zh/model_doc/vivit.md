版权所有，2023年 HuggingFace 团队保留所有权利。

根据 Apache 许可证，版本 2.0（“许可证”），除非符合许可证的规定，否则您不得使用此文件。
您可以在下面的链接处获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非有适用的法律规定或以书面形式达成协议，根据许可证分发的软件是基于“AS IS”原则分发的，没有任何明示或暗示的保证或条件。请参阅许可证以获得具体语言的权限和限制。

# 视频视觉变压器（ViViT）

## 概述

Vivit 模型是由 Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid 在 [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) 中提出的。
该论文提出了一套基于纯变压器的视频理解模型，这是首批取得成功的纯变压器模型。

论文中的摘要如下：

*我们提出了基于纯变压器的视频分类模型，借鉴了这些模型在图像分类中的最新成功。我们的模型从输入视频中提取时空标记，然后通过一系列变压器层进行编码。为了处理视频中遇到的长序列标记，我们提出了几种高效的模型变体，分解了输入的空间和时间维度。虽然已知基于变压器的模型只在有大型训练数据集可用时有效，但我们展示了如何在训练过程中有效地对模型进行正则化，并利用预训练的图像模型，能够在相对较小的数据集上进行训练。我们进行了彻底的消融研究，并在多个视频分类基准测试中取得了最先进的结果，包括 Kinetics 400 和 600、Epic Kitchens、Something-Something v2 和 Moments in Time，优于基于深度 3D 卷积网络的先前方法。*

此模型由 [jegormeister](https://huggingface.co/jegormeister) 贡献。可以在 [这里](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit) 找到原始代码（使用 JAX 编写）。

## VivitConfig

[[autodoc]] VivitConfig

## VivitImageProcessor

[[autodoc]] VivitImageProcessor
    - preprocess

## VivitModel

[[autodoc]] VivitModel
    - forward

## VivitForVideoClassification

[[autodoc]] transformers.VivitForVideoClassification
    - forward