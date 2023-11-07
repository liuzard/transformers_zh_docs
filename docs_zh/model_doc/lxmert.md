<!--版权 2020 HuggingFace团队。保留所有权利。

根据Apache许可证，版本2.0（“许可证”）许可；你不得在未遵守许可证的情况下使用此文件。
你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，
没有任何明示或暗示的保证或条件。有关许可证下所涉及的特定语言的权限和限制，请查看许可证。

⚠️请注意，这个文件是以Markdown格式编写的，但包含了我们文档生成器的特定语法（类似于MDX），可能无法在你的Markdown查看器中正确显示。-->

# LXMERT

## 概述

LXMERT模型的提出可参见Hao Tan和Mohit Bansal发表的研究论文[LXMERT:Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490)。该模型是一系列双向Transformer编码器的组合（一个负责图像模态，一个负责语言模态，最后一个用于融合两个模态），通过在预训练阶段进行掩码语言建模、视觉-语言文本对齐、ROI特征回归、掩码视觉属性建模、掩码视觉对象建模和视觉问答目标训练来完成。预训练过程包括多个多模态数据集：MSCOCO，Visual-Genome + Visual-Genome Question Answering，VQA 2.0和GQA。

论文中的摘要如下:

*视觉与语言推理需要对视觉概念、语言语义以及两种模态之间的对齐与关系有一定的理解。为此，我们提出了名为LXMERT（Learning Cross-Modality Encoder Representations from Transformers）的框架来学习视觉与语言之间的连接。在LXMERT中，我们构建了一个大规模的Transformer模型，包含三个编码器：对象关系编码器、语言编码器和跨模态编码器。此外，为了使我们的模型能够连接视觉和语言语义，我们使用大量的图像-句子对进行预训练，在其中执行五个不同的预训练任务：掩码语言建模、掩码对象预测（特征回归和标签分类）、跨模态匹配和图像问答。这些任务有助于学习单模态和跨模态关系。在我们的预训练参数微调后，我们的模型在两个视觉问答数据集（即VQA和GQA）上取得了最先进的结果。我们还通过将预训练的跨模态模型应用于具有挑战性的视觉推理任务NLVR，并将之前的最佳结果提升了22%（从54%提升到76%）。最后，我们进行了详细的消融研究，证明了我们的创新模型组件和预训练策略对我们强大的结果的重要贡献；同时，我们还展示了不同编码器的一些注意力可视化。*

提示:

- 在视觉特征嵌入中，不一定需要使用边界框，任何类型的视觉-空间特征都可以使用。
- LXMERT模型输出的语言隐藏状态和视觉隐藏状态都会经过跨模态层，因此它们包含了两个模态的信息。要访问只依赖于自身的模态，请选择元组中第一个输入的视觉/语言隐藏状态。
- 双向跨模态编码器的注意力只在将语言模态用作输入、视觉模态用作上下文向量时返回注意力值。此外，虽然跨模态编码器包含各自模态和跨模态的自注意力，但只返回跨模态注意力，忽略了两个自注意力输出。

此模型由[eltoto1219](https://huggingface.co/eltoto1219)提供。原始代码可在[这里](https://github.com/airsplay/lxmert)找到。

## 文档资源

- [问答任务指南](../tasks/question_answering)

## LxmertConfig

[[autodoc]] LxmertConfig

## LxmertTokenizer

[[autodoc]] LxmertTokenizer

## LxmertTokenizerFast

[[autodoc]] LxmertTokenizerFast

## Lxmert特定的输出

[[autodoc]] models.lxmert.modeling_lxmert.LxmertModelOutput

[[autodoc]] models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput

[[autodoc]] models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput

[[autodoc]] models.lxmert.modeling_tf_lxmert.TFLxmertModelOutput

[[autodoc]] models.lxmert.modeling_tf_lxmert.TFLxmertForPreTrainingOutput

## LxmertModel

[[autodoc]] LxmertModel
    - forward

## LxmertForPreTraining

[[autodoc]] LxmertForPreTraining
    - forward

## LxmertForQuestionAnswering

[[autodoc]] LxmertForQuestionAnswering
    - forward

## TFLxmertModel

[[autodoc]] TFLxmertModel
    - call

## TFLxmertForPreTraining

[[autodoc]] TFLxmertForPreTraining
    - call