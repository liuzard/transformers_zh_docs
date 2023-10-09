版权所有2022年背爱抚团队。
根据Apache许可证2.0版（“许可证”），除非遵守许可证，否则不得使用此文件。你可以在以下位置获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律规定或书面同意，根据许可证分发的软件基于“按原样”提供，没有任何明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️请注意，此文件是Markdown格式，但包含我们文档生成器（类似于MDX）的特定语法，可能无法在你的Markdown查看器中正确显示。

# FLAVA

## 概述

《FLAVA: A Foundational Language And Vision Alignment Model》的作者为Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, Douwe Kiela，该模型已被CVPR 2022接受。

该论文旨在创建一个统一的基础模型，可同时用于视觉、语言和视觉语言多模式任务。

论文的摘要如下：

“最先进的视觉和视觉语言模型依赖于大规模的视觉语言预训练，以在各种下游任务中获得良好的性能。通常，这些模型通常是跨模态的（对比）或多模态的（具有较早的融合），但不是同时都具备；他们通常只针对特定的模态或任务。一个有希望的方向是使用一个单一的整体通用模型作为“基础”，它可以同时面向所有模态---真正的视觉和语言基础模型应该能够很好地处理视觉任务、语言任务以及跨模态和多模态的视觉和语言任务。我们将FLAVA引入为这样的模型，并在包含这些目标模态的广泛范围的35个任务上展示了令人印象深刻的性能。”

该模型由aps贡献。原始代码可以在[此处](https://github.com/facebookresearch/multimodal/tree/main/examples/flava)找到。

## FlavaConfig

[[autodoc]] FlavaConfig

## FlavaTextConfig

[[autodoc]] FlavaTextConfig

## FlavaImageConfig

[[autodoc]] FlavaImageConfig

## FlavaMultimodalConfig

[[autodoc]] FlavaMultimodalConfig

## FlavaImageCodebookConfig

[[autodoc]] FlavaImageCodebookConfig

## FlavaProcessor

[[autodoc]] FlavaProcessor

## FlavaFeatureExtractor

[[autodoc]] FlavaFeatureExtractor

## FlavaImageProcessor

[[autodoc]] FlavaImageProcessor
    - preprocess

## FlavaForPreTraining

[[autodoc]] FlavaForPreTraining
    - forward

## FlavaModel

[[autodoc]] FlavaModel
    - forward
    - get_text_features
    - get_image_features

## FlavaImageCodebook

[[autodoc]] FlavaImageCodebook
    - forward
    - get_codebook_indices
    - get_codebook_probs

## FlavaTextModel

[[autodoc]] FlavaTextModel
    - forward

## FlavaImageModel

[[autodoc]] FlavaImageModel
    - forward

## FlavaMultimodalModel

[[autodoc]] FlavaMultimodalModel
    - forward