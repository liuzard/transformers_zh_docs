<!--版权 2023 The HuggingFace Team. All rights reserved.

根据Apache许可证，版本2.0（“许可证”），你不得使用此文件，除非符合 许可证 。你可以在以下网址获取许可证的副本 http://www.apache.org/licenses/LICENSE-2.0

除非法律要求或书面同意，根据许可证分发的软件是基于“原样”的原则，不附带任何形式的保证条款或条件，无论是明示的还是暗示的。请参阅许可证以了解许可证下的特定语言 条件和限制。

⚠️ 请注意，该文件为Markdown格式，但包含了特定于我们的文档构建器的语法（类似于MDX），你的Markdown查看器可能无法正确 渲染。

-->

# BLIP

## 概述

《BLIP: 引导式语言-图像预训练用于统一的视觉-语言理解与生成》一文中提出了BLIP模型（由作者Junnan Li，Dongxu Li，Caiming Xiong，Steven Hoi共同贡献）。

BLIP是一个能够执行各种多模态任务的模型，包括：
- 视觉问答
- 图像-文本检索（图像-文本匹配）
- 图像字幕生成

该论文的摘要如下：

“视觉-语言预训练（VLP）极大地提升了许多视觉-语言任务的性能。然而，大多数现有的预训练模型在理解型任务和生成型任务中都表现出色。此外，性能改进主要通过扩大从网络收集到的带有噪声的图像-文本对数据集来实现，这是一种次优的监督来源。在本论文中，我们提出了BLIP，这是一个新的VLP框架，可以自由地转移到视觉-语言理解和生成任务。BLIP通过引导字幕来有效利用嘈杂的网络数据，其中一个标题生成器生成合成字幕，一个过滤器删除噪声字幕。我们在广泛的视觉-语言任务上取得了最先进的结果，如图像-文本检索（平均精确率@1提高2.7%）、图像字幕生成（CIDEr提高2.8%）和VQA（VQA分数提高1.6%）。BLIP在直接转移到视觉-语言任务中的零射程方式中也表现出良好的泛化能力。代码、模型和数据集已公开发布。”

![BLIP.gif](https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif)

此模型由[ybelkada](https://huggingface.co/ybelkada)提供。原始代码可在[此处](https://github.com/salesforce/BLIP)找到。

## 资源

- [Jupyter笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb)：关于如何在自定义数据集上精调BLIP进行图像字幕生成的示例。

## BlipConfig

[[autodoc]] BlipConfig
    - from_text_vision_configs

## BlipTextConfig

[[autodoc]] BlipTextConfig

## BlipVisionConfig

[[autodoc]] BlipVisionConfig

## BlipProcessor

[[autodoc]] BlipProcessor

## BlipImageProcessor

[[autodoc]] BlipImageProcessor
    - preprocess

## BlipModel

[[autodoc]] BlipModel
    - forward
    - get_text_features
    - get_image_features

## BlipTextModel

[[autodoc]] BlipTextModel
    - forward

## BlipVisionModel

[[autodoc]] BlipVisionModel
    - forward

## BlipForConditionalGeneration

[[autodoc]] BlipForConditionalGeneration
    - forward

## BlipForImageTextRetrieval

[[autodoc]] BlipForImageTextRetrieval
    - forward

## BlipForQuestionAnswering

[[autodoc]] BlipForQuestionAnswering
    - forward

## TFBlipModel

[[autodoc]] TFBlipModel
    - call
    - get_text_features
    - get_image_features

## TFBlipTextModel

[[autodoc]] TFBlipTextModel
    - call

## TFBlipVisionModel

[[autodoc]] TFBlipVisionModel
    - call

## TFBlipForConditionalGeneration

[[autodoc]] TFBlipForConditionalGeneration
    - call

## TFBlipForImageTextRetrieval

[[autodoc]] TFBlipForImageTextRetrieval
    - call

## TFBlipForQuestionAnswering

[[autodoc]] TFBlipForQuestionAnswering
    - call