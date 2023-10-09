<!--版权 2020 The HuggingFace团队。版权所有。

根据Apache许可证第2版 ("许可证")授权;除非符合许可证的要求，否则您不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本

除非适用法律要求或书面同意，以书面形式分发的软件均是基于"AS IS"的基础上分发的，不附带任何明示或暗示的保证或条件。请参阅许可证，了解有关特定语言的权限和限制。

⚠️ 请注意，此文件使用Markdown格式，但包含了特定于我们doc-builder（类似于MDX）的语法，可能在您的Markdown查看器中无法正确渲染。

-->

# 漏斗变压器

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=funnel">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-funnel-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/funnel-transformer-small">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>


## 概述

Funnel Transformer模型是在论文[Funnel-Transformer：为高效语言处理过滤连续冗余](https://arxiv.org/abs/2006.03236)中提出的。它是一个双向变压器模型，类似于BERT，但在每个层块之后加上了池化操作，类似于传统的计算机视觉中的卷积神经网络（CNN）。

论文中的摘要如下：

*随着语言预训练的成功，开发更高效的可扩展体系结构以更低的成本利用丰富的未标记数据变得非常有吸引力。为了提高效率，我们检查了保持全长标记级表示中经常被忽视的冗余，特别是对于仅需要序列的单向量表示的任务。基于这个直觉，我们提出Funnel-Transformer，逐渐将隐藏状态序列压缩到更短的序列，从而减少计算成本。更重要的是，通过将长度减少的保存的FLOPs再投资于构建更深或更宽的模型，进一步提高模型性能。此外，为了执行常见的预训练目标所需的标记级预测，Funnel-Transformer能够通过解码器从已减少的隐藏序列中恢复每个标记的深度表示。经验证明，在具有可比或更少FLOPs的情况下，Funnel-Transformer在包括文本分类、语言理解和阅读理解在内的各种序列级预测任务上表现优于标准的Transformer。*

提示：

- 由于Funnel Transformer使用了池化操作，所以在每个层块之后，隐藏状态的序列长度会发生变化。这样一来，它们的长度会变为原始长度的一半，从而加快了下一个隐藏状态的计算速度。
  基础模型的最终序列长度因此是原始序列长度的四分之一。该模型可以直接用于只需要句子摘要的任务（如序列分类或多项选择）。对于其他任务，使用完整模型；这个完整模型具有解码器，可以将最终的隐藏状态上采样到与输入相同的序列长度。
- 对于分类等任务，这不是一个问题，但对于掩码语言建模或标记分类等任务，我们需要具有与原始输入序列长度相同的隐藏状态。在这些情况下，最终的隐藏状态会上采样到输入序列长度，并经过两个额外的层。这就是为什么每个检查点都有两个版本的原因。带有“-base”后缀的版本仅包含三个块，而没有该后缀的版本包含三个块和上采样头及其额外的层。
- 漏斗变压器检查点都有完整版本和基础版本。第一组应用于[`FunnelModel`]、[`FunnelForPreTraining`]、
  [`FunnelForMaskedLM`]、[`FunnelForTokenClassification`] 和
  [`FunnelForQuestionAnswering`]。第二组应用于
  [`FunnelBaseModel`]、[`FunnelForSequenceClassification`] 和
  [`FunnelForMultipleChoice`]。

该模型由[sgugger](https://huggingface.co/sgugger)贡献。原始代码可在[此处](https://github.com/laiguokun/Funnel-Transformer)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)


## FunnelConfig

[[autodoc]] FunnelConfig

## FunnelTokenizer

[[autodoc]] FunnelTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FunnelTokenizerFast

[[autodoc]] FunnelTokenizerFast

## 漏斗特定输出

[[autodoc]] models.funnel.modeling_funnel.FunnelForPreTrainingOutput

[[autodoc]] models.funnel.modeling_tf_funnel.TFFunnelForPreTrainingOutput

## FunnelBaseModel

[[autodoc]] FunnelBaseModel
    - forward

## FunnelModel

[[autodoc]] FunnelModel
    - forward

## FunnelModelForPreTraining

[[autodoc]] FunnelForPreTraining
    - forward

## FunnelForMaskedLM

[[autodoc]] FunnelForMaskedLM
    - forward

## FunnelForSequenceClassification

[[autodoc]] FunnelForSequenceClassification
    - forward

## FunnelForMultipleChoice

[[autodoc]] FunnelForMultipleChoice
    - forward

## FunnelForTokenClassification

[[autodoc]] FunnelForTokenClassification
    - forward

## FunnelForQuestionAnswering

[[autodoc]] FunnelForQuestionAnswering
    - forward

## TFFunnelBaseModel

[[autodoc]] TFFunnelBaseModel
    - call

## TFFunnelModel

[[autodoc]] TFFunnelModel
    - call

## TFFunnelModelForPreTraining

[[autodoc]] TFFunnelForPreTraining
    - call

## TFFunnelForMaskedLM

[[autodoc]] TFFunnelForMaskedLM
    - call

## TFFunnelForSequenceClassification

[[autodoc]] TFFunnelForSequenceClassification
    - call

## TFFunnelForMultipleChoice

[[autodoc]] TFFunnelForMultipleChoice
    - call

## TFFunnelForTokenClassification

[[autodoc]] TFFunnelForTokenClassification
    - call

## TFFunnelForQuestionAnswering

[[autodoc]] TFFunnelForQuestionAnswering
    - call