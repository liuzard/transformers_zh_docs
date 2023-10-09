<!--版权2022年HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”）许可。除非符合许可证，否则不得使用此文件。你可以在以下位置获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于“原样”分发，不附带任何明示或暗示的保证或条件。请参阅许可证以了解许可下的特定语言和限制。

⚠️请注意，此文件是Markdown文件，但包含特定于doc-builder的语法（类似于MDX），可能无法在你的Markdown查看器中正确显示。-->

# X-CLIP

## 概述

X-CLIP模型的提出是在Bolin Ni、Houwen Peng、Minghao Chen、Songyang Zhang、Gaofeng Meng、Jianlong Fu、Shiming Xiang和Haibin Ling的论文[《扩展用于通用视频识别的语言-图像预训练模型》](https://arxiv.org/abs/2208.02816)中。X-CLIP是[CLIP](clip)在视频领域的最小扩展。该模型由文本编码器、跨帧视觉编码器、多帧集成Transformer和视频特定提示生成器组成。

论文摘要如下所示：

*对比语言-图像预训练已经在从互联网规模数据中学习视觉-文本联合表示方面取得了巨大的成功，展示了在各种图像任务中显着的“零样本”泛化能力。然而，如何有效地将这种新的语言-图像预训练方法扩展到视频领域仍然是一个开放的问题。在这项工作中，我们提出了一种简单而有效的方法，直接将预训练的语言-图像模型应用于视频识别，而不是从头开始预训练一个新模型。更具体地说，为了捕捉帧之间沿时间维度的长程依赖性，我们提出了一种跨帧注意机制，明确地在帧之间交换信息。这种模块轻巧且可以无缝地插入预训练的语言-图像模型中。此外，我们提出了一种视频特定的提示方案，利用视频内容信息生成有区分度的文本提示。广泛的实验证明我们的方法是有效的，并且可以推广到不同的视频识别场景。特别是，在全监督设置下，我们的方法在Kinectics-400数据集上达到了87.1%的top-1准确率，与Swin-L和ViViT-H相比FLOPs使用了12倍更少。在零样本实验中，我们的方法在两个常用协议的top-1准确率方面超过了当前最先进的方法+7.6%和+14.9%。在少样本场景下，我们的方法在标记数据非常有限的情况下，比之前的最佳方法提高了+32.1%和+23.1%。*

提示：

- 使用X-CLIP与[CLIP](clip)相同。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/xclip_architecture.png"
alt="drawing" width="600"/> 

<small> X-CLIP架构。来自<a href="https://arxiv.org/abs/2208.02816">原论文</a>。 </small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可以在[此处](https://github.com/microsoft/VideoX/tree/master/X-CLIP)找到。

## 资源

以下是官方的Hugging Face资源和社区（用🌎表示）资源列表，可帮助你开始使用X-CLIP。

- X-CLIP的演示笔记本可以在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/X-CLIP)找到。

如果你有兴趣提交资源以纳入此处，请随时提出拉取请求，我们将对其进行审查！资源应该展示出新的东西，而不是重复现有的资源。

## XCLIPProcessor

[[autodoc]] XCLIPProcessor

## XCLIPConfig

[[autodoc]] XCLIPConfig
    - from_text_vision_configs

## XCLIPTextConfig

[[autodoc]] XCLIPTextConfig

## XCLIPVisionConfig

[[autodoc]] XCLIPVisionConfig

## XCLIPModel

[[autodoc]] XCLIPModel
    - forward
    - get_text_features
    - get_video_features

## XCLIPTextModel

[[autodoc]] XCLIPTextModel
    - forward

## XCLIPVisionModel

[[autodoc]] XCLIPVisionModel
    - forward