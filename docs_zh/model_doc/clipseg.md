<!--版权2022年HuggingFace团队保留所有权利。

根据Apache License, Version 2.0许可证（以下称“许可证”），您不能使用此文件，除非符合许可证的规定。
您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律另有规定或书面同意，根据许可证分发的软件是按"原样"提供的，不附带任何明示或暗示的保证或条件。有关许可证的特定语言和限制的详细信息，请参阅许可证。

⚠️请注意，此文件采用Markdown格式，但包含特定语法用于我们的doc-builder（类似MDX），可能在您的Markdown查看器中无法正确显示。

-->

# CLIPSeg

## 概述

CLIPSeg模型是由Timo Lüddecke和Alexander Ecker在《使用文本和图像提示进行图像分割》(https://arxiv.org/abs/2112.10003)中提出的。CLIPSeg在冻结的[CLIP](clip)模型之上添加了一个最小的解码器，用于零次和一次图像分割。

以下是论文的摘要：

*通常通过训练一个模型来处理固定对象类的图像分割。随后，在测试时将其他类或更复杂的查询合并进来是非常昂贵的，因为它需要在包含这些表达的数据集上重新训练模型。在这里，我们提出了一种可以根据任意提示在测试时生成图像分割的系统。提示可以是文本或图像。这种方法使我们能够为三种常见的分割任务创建一个统一的模型（仅训练一次），这些任务有着不同的挑战：指称表达式分割、零次分割和一次分割。我们在CLIP模型的基础上构建了一个骨干，并在其上扩展了一个基于Transformer的解码器，以实现密集预测。在PhraseCut数据集的扩展版本上进行训练后，我们的系统可以根据自由文本提示或表达查询的附加图像为图像生成二进制分割图。我们详细分析了不同变体的后一类基于图像的提示。这种新颖的混合输入不仅能够动态适应上面提到的三种分割任务，还能够适应任何可以用文本或图像查询表达的二进制分割任务。最后，我们发现我们的系统在涉及捷径或属性的广义查询中表现出良好的适应性*

提示：

- [`CLIPSegForImageSegmentation`] 在[`CLIPSegModel`]之上添加一个解码器。后者与[`CLIPModel`]相同。
- [`CLIPSegForImageSegmentation`] 可以根据测试时的任意提示生成图像分割。提示可以是文本（作为`input_ids`提供给模型）或图像（作为`conditional_pixel_values`提供给模型）。您还可以提供自定义的条件嵌入（作为`conditional_embeddings`提供给模型）。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/clipseg_architecture.png" alt="drawing" width="600"/> 

<small> CLIPSeg概述。摘自<a href="https://arxiv.org/abs/2112.10003">原始论文</a>。</small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可以在[此处](https://github.com/timojl/clipseg)找到。

## 资源

官方Hugging Face和社区（以🌎表示）资源列表，以帮助您开始使用CLIPSeg。如果您有兴趣提交资源以包含在此处，请随时提出拉取请求，我们将进行审核！该资源应该体现出一些新的东西，而不是重复现有的资源。

<PipelineTag pipeline="image-segmentation"/>

- [使用CLIPSeg进行零次图像分割的示例笔记本](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb).

## CLIPSegConfig

[[autodoc]] CLIPSegConfig
    - from_text_vision_configs

## CLIPSegTextConfig

[[autodoc]] CLIPSegTextConfig

## CLIPSegVisionConfig

[[autodoc]] CLIPSegVisionConfig

## CLIPSegProcessor

[[autodoc]] CLIPSegProcessor

## CLIPSegModel

[[autodoc]] CLIPSegModel
    - forward
    - get_text_features
    - get_image_features

## CLIPSegTextModel

[[autodoc]] CLIPSegTextModel
    - forward

## CLIPSegVisionModel

[[autodoc]] CLIPSegVisionModel
    - forward

## CLIPSegForImageSegmentation

[[autodoc]] CLIPSegForImageSegmentation
    - forward