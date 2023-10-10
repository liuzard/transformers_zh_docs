<!--版权所有2020年HuggingFace团队保留。

根据Apache许可证第2.0版（“许可证”）许可;你除符合许可证的规定外，不得使用此文件。
你可以在以下链接获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，根据许可证分发的软件以“原样”分发，不附带任何明示或暗示的担保或条件。详细了解许可证的特殊语言。
注意，此文件使用Markdown格式，但包含特定于我们的doc-builder的语法（类似于MDX），可能在Markdown查看器中无法正确显示。

-->

# XLM

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=xlm">
<img alt="All_model_pages-xlm" src="https://img.shields.io/badge/All_model_pages-xlm-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/xlm-mlm-en-2048">
<img alt="Hugging Face-Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

[XLM模型](https://arxiv.org/abs/1901.07291)是由Guillaume Lample, Alexis Conneau提出的。它是使用以下目标之一进行预训练的转换器：

- 因果语言建模（CLM）目标（下一个token预测），
- 掩蔽语言建模（MLM）目标（类似于BERT），或
- 一种翻译语言建模（TLM）目标（BERT的MLM扩展，可用于多语言输入）

论文中的摘要如下所示：

*最近的研究表明，生成预训练在英语自然语言理解方面的效率。在本文中，我们将这种方法扩展到多种语言，并展示了跨语言预训练的有效性。我们提出了两种学习跨语言语言模型（XLM）的方法：一种是无监督学习，只依赖于单语数据，另一种是监督学习，使用新的跨语言语言模型目标利用并行数据。我们在跨语言分类、无监督和监督机器翻译方面取得了最新的成果。在XNLI上，我们的方法提高了4.9%的准确性。在无监督机器翻译方面，我们在WMT'16德英上达到了34.3 BLEU的效果，超过了以往的最佳效果。在监督机器翻译方面，我们在WMT'16罗马尼亚英语上获得了38.5 BLEU的最新成果，超过了以前的最佳方法4 BLEU。我们的代码和预训练模型将公开发布。*

提示：

- XLM有许多不同的检查点，它们使用不同的目标进行训练：CLM，MLM或TLM。确保为你的任务选择正确的目标（例如，MLM检查点不适合生成）。
- XLM有多语言检查点，可以使用特定的`lang`参数。请查看[multilingual]（../multilingual）页面以获取更多信息。
- 这是一个在多种语言上训练的转换器模型。此模型的训练有三种不同类型，该库提供了所有这些类型的检查点：

  * 因果语言建模（CLM）是传统的自回归训练（因此该模型也可以在前一节中）。为每个训练样本选择其中一种语言，并且模型输入是一句256个标记的句子，可能跨越其中一种语言的多个文档。
  * 掩蔽语言建模（MLM）类似于RoBERTa。为每个训练样本选择其中一种语言，并且模型输入是一句256个标记的句子，可能跨越其中一种语言的多个文档，并且进行标记token的动态掩码。
  * MLM和翻译语言建模（TLM）的组合。这包括连接两种不同语言的句子，并进行随机掩蔽。为了预测其中一个被掩蔽的token，模型可以使用语言1中的周围上下文和语言2给出的上下文。

本模型由[thomwolf](https://huggingface.co/thomwolf)贡献。原始代码可以在[这里](https://github.com/facebookresearch/XLM/)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## XLMConfig

[[autodoc]] XLMConfig

## XLMTokenizer

[[autodoc]] XLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLM特定输出

[[autodoc]] models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput

## XLMModel

[[autodoc]] XLMModel
    - forward

## XLMWithLMHeadModel

[[autodoc]] XLMWithLMHeadModel
    - forward

## XLMForSequenceClassification

[[autodoc]] XLMForSequenceClassification
    - forward

## XLMForMultipleChoice

[[autodoc]] XLMForMultipleChoice
    - forward

## XLMForTokenClassification

[[autodoc]] XLMForTokenClassification
    - forward

## XLMForQuestionAnsweringSimple

[[autodoc]] XLMForQuestionAnsweringSimple
    - forward

## XLMForQuestionAnswering

[[autodoc]] XLMForQuestionAnswering
    - forward

## TFXLMModel

[[autodoc]] TFXLMModel
    - call

## TFXLMWithLMHeadModel

[[autodoc]] TFXLMWithLMHeadModel
    - call

## TFXLMForSequenceClassification

[[autodoc]] TFXLMForSequenceClassification
    - call

## TFXLMForMultipleChoice

[[autodoc]] TFXLMForMultipleChoice
    - call

## TFXLMForTokenClassification

[[autodoc]] TFXLMForTokenClassification
    - call

## TFXLMForQuestionAnsweringSimple

[[autodoc]] TFXLMForQuestionAnsweringSimple
    - call