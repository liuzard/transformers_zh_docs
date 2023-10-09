版权所有 (2023) HuggingFace 团队。

根据 Apache 许可证，第 2.0 版 (以下简称“许可证”) 许可；除非遵循许可证，否则不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按“原样”分发，不附带任何形式的明示或暗示的保证或条件。请参阅许可证以获取特定语言下的权限和限制。

# 金字塔视觉变换器 (PVT)

## 概述

PVT 模型是由 Wang,Wenhai, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao 在 [金字塔视觉变换器: 一种无卷积稠密预测的多功能骨干](https://arxiv.org/abs/2102.12122) 中提出的。PVT 是一种利用金字塔结构的视觉变换器，可用作密集预测任务的有效骨干。具体来说，它允许使用更精细的输入 (每个补丁 4 x 4 像素)，同时随着变深，缩短变换器的序列长度，从而降低计算成本。此外，还使用了空间缩减注意力 (SRA) 层，用于进一步减少学习高分辨率特征时的资源消耗。

以下是论文摘要：

*尽管卷积神经网络 (CNN) 在计算机视觉领域取得了巨大成功，但本研究探讨了一种更简单、无卷积的骨干网络，对许多密集预测任务很有用。与最近提出的专用于图像分类的视觉变换器 (ViT) 不同，我们介绍了金字塔视觉变换器 (PVT)，克服了将变换器移植到各种密集预测任务中的困难。与通常产生低分辨率输出并导致较高计算和内存成本的 ViT 不同，PVT 不仅可以在图像的密集分区上进行训练，以实现高输出分辨率(对于密集预测很重要)，还使用逐步缩小的金字塔来减少大型特征图的计算量。PVT 继承了 CNN 和变换器的优点，使其成为各种视觉任务的统一骨干，无需卷积，可直接替换 CNN 骨干。我们通过广泛的实验证实了 PVT 的有效性，表明它提高了许多下游任务的性能，包括目标检测、实例和语义分割。例如，使用相应数量的参数，PVT+RetinaNet 在 COCO 数据集上达到 40.4 AP，超过了 ResNet50+RetinNet (36.3 AP) 的 4.1 绝对 AP (见图 2)。我们希望 PVT 能成为像素级预测的替代和有用的骨干，推动未来的研究。*

此模型由 [Xrenya](<https://huggingface.co/Xrenya>) 提供。原始代码可在 [此处](https://github.com/whai362/PVT) 找到。

- PVTv1 在 ImageNet-1K 上

| **模型变体**  | **大小** | **Acc@1** | **参数 (M)** |
|--------------|:-------:|:---------:|:------------:|
| PVT-Tiny     |   224   |   75.1    |     13.2     |
| PVT-Small    |   224   |   79.8    |     24.5     |
| PVT-Medium   |   224   |   81.2    |     44.2     |
| PVT-Large    |   224   |   81.7    |     61.4     |

## PvtConfig

[[autodoc]] PvtConfig

## PvtImageProcessor

[[autodoc]] PvtImageProcessor
    - preprocess

## PvtForImageClassification

[[autodoc]] PvtForImageClassification
    - forward

## PvtModel

[[autodoc]] PvtModel
    - forward