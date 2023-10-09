<!--
版权所有2023 HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”），你除非符合许可证规定，否则不得使用此文件。
你可以获取许可证副本，请访问

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按“按原样”基础分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以获取
有关许可证下特定语言的权限和限制的详细信息。

⚠️请注意，该文件是Markdown格式，但包含我们文档生成器的特定语法（类似于MDX），可能在你的Markdown查看器中无法正确呈现。
-->

# Pix2Struct

## 概述

[Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347)是由Kent Lee、Mandar Joshi、Iulia Turc、Hexiang Hu、Fangyu Liu、Julian Eisenschlos、Urvashi Khandelwal、Peter Shaw、Ming-Wei Chang和Kristina Toutanova提出的。其中Pix2Struct模型的论文摘要如下：

> 视觉语言无处不在 - 数据源范围从带有图表的教科书到带有图像和表格的网页，再到具有按钮和表格的移动应用程序。也许是由于这种多样性，先前的工作通常依赖于领域特定的方法，对底层数据、模型架构和目标的共享有限。我们提出了Pix2Struct，这是一个仅用于视觉语言理解的预训练图像到文本模型，可以在包含视觉定位语言的任务上进行微调。Pix2Struct的预训练目标是学习将屏幕截图中的遮罩解析成简化的HTML。Web具有对HTML结构清晰反映的丰富的视觉元素，为下游任务的多样性提供了一个非常适合的预训练数据源。直观地说，这个目标包括了识别文字、语言建模、图像字幕等通用的预训练信号。除了新颖的预训练策略，我们还引入了可变分辨率的输入表示和更灵活的语言和视觉输入集成方式，其中诸如问题的语言提示直接呈现在输入图像上。我们首次展示了，单个预训练模型在四个领域的九个任务中有六个任务达到了最新的结果：文档、插图、用户界面和自然图片。

技巧：

Pix2Struct已经在各种任务和数据集上进行了微调，涵盖了从图像字幕、视觉问答（VQA）到不同输入（书籍、图表、科学图表）、说明UI组件的字幕等任务。完整列表可以在论文的表1中找到。
因此，我们建议你将这些模型用于它们已经进行细调的任务。例如，如果你想使用Pix2Struct进行UI的字幕，你应该使用对UI数据集进行了精调的模型。如果你想使用Pix2Struct进行图像字幕，你应该使用对自然图像字幕数据集进行了精调的模型，依此类推。

如果你想使用该模型进行有条件的文本生成，请确保使用带有`add_special_tokens=False`的处理器。

该模型由[ybelkada](https://huggingface.co/ybelkada)贡献。
原始代码可以在[这里](https://github.com/google-research/pix2struct)找到。

## 资源

- [微调示例笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb)
- [所有模型](https://huggingface.co/models?search=pix2struct)


## Pix2StructConfig

[[autodoc]] Pix2StructConfig
    - from_text_vision_configs

## Pix2StructTextConfig

[[autodoc]] Pix2StructTextConfig

## Pix2StructVisionConfig

[[autodoc]] Pix2StructVisionConfig

## Pix2StructProcessor

[[autodoc]] Pix2StructProcessor

## Pix2StructImageProcessor

[[autodoc]] Pix2StructImageProcessor
    - preprocess

## Pix2StructTextModel

[[autodoc]] Pix2StructTextModel
    - forward

## Pix2StructVisionModel

[[autodoc]] Pix2StructVisionModel
    - forward

## Pix2StructForConditionalGeneration

[[autodoc]] Pix2StructForConditionalGeneration
    - forward