版权所有 2021 年 HuggingFace 团队。

根据 Apache 许可证第 2.0 版（“许可证”）的条款；除非遵守许可证，否则不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

请注意，该文件以 Markdown 格式呈现，但包含我们的 doc-builder 的特定语法（类似于 MDX），可能无法在你的 Markdown 查看器中正确呈现。

# VisionTextDualEncoder

## 概述

[`VisionTextDualEncoderModel`] 可用于使用任何预训练的视觉自编码模型作为视觉编码器（例如 [ViT](vit)、[BEiT](beit)、[DeiT](deit)）和任何预训练的文本自编码模型作为文本编码器（例如 [RoBERTa](roberta)、[BERT](bert)）来初始化视觉文本双编码器模型。在视觉编码器和文本编码器之上添加了两个投影层，将输出嵌入投影到共享潜空间。投影层随机初始化，因此该模型应在下游任务上进行微调。该模型可用于使用 CLIP 类似的对比图像文本训练来对齐视觉文本嵌入，然后可用于零样本视觉任务，如图像分类或检索。

在《LiT: Zero-Shot Transfer with Locked-image Text Tuning》中，展示了如何利用预训练的（锁定/冻结）图像和文本模型进行对比学习，显著改进了新的零样本视觉任务，如图像分类或检索。

## VisionTextDualEncoderConfig

[[autodoc]] VisionTextDualEncoderConfig

## VisionTextDualEncoderProcessor

[[autodoc]] VisionTextDualEncoderProcessor

## VisionTextDualEncoderModel

[[autodoc]] VisionTextDualEncoderModel
    - forward

## FlaxVisionTextDualEncoderModel

[[autodoc]] FlaxVisionTextDualEncoderModel
    - __call__

## TFVisionTextDualEncoderModel

[[autodoc]] TFVisionTextDualEncoderModel
    - call