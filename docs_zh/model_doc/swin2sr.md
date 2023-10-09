版权所有 2022 The HuggingFace Team.

根据 Apache License, Version 2.0 (the "License") 的许可，除非符合 License 你不得使用此文件。你可以在以下网址获取 License 的副本：http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或者书面同意，按"AS IS" BASIS 分发的软件被在没有任何担保或者条件的情况下分发，不论是明示还是暗示。详情请参阅 License 的具体语言和限制。

⚠️ 请注意，此文件是 Markdown 文件，但包含了特定的语法给我们的文档生成器(doc-builder)使用（类似于 MDX），可能在你的 Markdown 查看器中无法正确显示。

# Swin2SR

## 概述

[Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345) 是由 Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, Radu Timofte 提出的 Swin2SR 模型。Swin2SR 在 [SwinIR](https://github.com/JingyunLiang/SwinIR/) 模型的基础上加入了 [Swin Transformer v2](swinv2) 层，以减轻训练不稳定性、预训练和微调之间的分辨率差距以及数据上的限制等问题。

论文中的摘要如下所示：
*压缩在流媒体服务、虚拟现实或视频游戏等有限带宽系统中对于有效传输和存储图像和视频起着重要作用。然而，压缩不可避免地导致伪影和原始信息的丧失，这可能严重降低视觉质量。出于这些原因，压缩图像的质量增强已经成为一个热门的研究课题。虽然大多数最先进的图像恢复方法是基于卷积神经网络的，但一些基于 Transformer 的方法，如 SwinIR，在这些任务中表现出令人印象深刻的性能。
在本文中，我们探索了新颖的 Swin Transformer V2，以改进 SwinIR 用于图像超分辨率，特别是压缩输入的场景。使用这种方法，我们可以解决 Transformer 视觉模型训练中的主要问题，例如训练不稳定性、预训练和微调之间的分辨率差距以及数据上的限制。我们在三个代表性任务上进行了实验：JPEG 压缩伪影去除、图像超分辨率（经典和轻量级）以及压缩图像超分辨率。实验结果表明，我们的方法 Swin2SR 能够提高 SwinIR 的训练收敛性和性能，并且在 "AIM 2022 挑战赛：压缩图像和视频的超分辨率" 中是前五的解决方案。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/swin2sr_architecture.png"
alt="drawing" width="600"/>

<small> Swin2SR 模型架构。来源于 <a href="https://arxiv.org/abs/2209.11345">原始论文</a>。 </small>

此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。
原始代码可以在 [这里](https://github.com/mv-lab/swin2sr) 找到。

## 资源

Swin2SR 的演示笔记本可以在 [这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Swin2SR) 找到。

使用 SwinSR 进行图像超分辨率的演示空间可以在 [这里](https://huggingface.co/spaces/jjourney1125/swin2sr) 找到。

## Swin2SRImageProcessor

[[autodoc]] Swin2SRImageProcessor
    - preprocess

## Swin2SRConfig

[[autodoc]] Swin2SRConfig

## Swin2SRModel

[[autodoc]] Swin2SRModel
    - forward

## Swin2SRForImageSuperResolution

[[autodoc]] Swin2SRForImageSuperResolution
    - forward