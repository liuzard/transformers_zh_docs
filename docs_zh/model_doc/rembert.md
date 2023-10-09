<!--版权所有2020年The HuggingFace团队保留。

根据Apache许可证第2版（“许可证”）获得许可；您除非遵守许可证，否则不得使用此文件。您可以在获取以下许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”“没有保修”或“条件”的基础上分发的，无论是明示或暗示。特定领域的特定语言渲染，其在您的Markdown查看器中可能无法正确渲染。

-->

# RemBERT

## 概览

RemBERT模型是由Hyung Won Chung、Thibault Févry、Henry Tsai和Melvin Johnson在[Rethinking Embedding Coupling in Pre-trained Language Models](https://arxiv.org/abs/2010.12821)一文中提出的。

文中的摘要如下：

*我们重新评估了在最先进的预训练语言模型中输入嵌入和输出嵌入之间共享权重的标准做法。我们表明，解耦的嵌入提供了增加建模灵活性的功能，使我们能够通过重新分配多语种模型的输入嵌入的参数来显著提高参数分配的效率。通过在Transformer层中重新分配输入嵌入参数，我们在微调期间在相同数量的参数下实现了显著更好的标准自然语言理解任务性能。我们还表明，在预训练后丢弃输出嵌入后，分配额外的能力给模型在微调阶段提供了好处。我们的分析表明，较大输出嵌入可以防止模型的最后层对预训练任务过度特殊化，并鼓励Transformer表示更加通用和可转移到其他任务和语言。利用这些发现，我们能够训练出在XTREME基准测试上表现强劲的模型，而不增加微调阶段的参数数量。*

提示：

对于微调，RemBERT可以被认为是mBERT的一个更大版本，具有嵌入层的ALBERT类似分解。在预训练期间，输入嵌入是不绑定的，与使用BERT的不同，这使得可以使用较小的输入嵌入（在微调期间保留），同时使用更大的输出嵌入（在微调期间丢弃）。分词器也类似于Albert而不是BERT。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问题回答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## RemBertConfig

[[autodoc]] RemBertConfig

## RemBertTokenizer

[[autodoc]] RemBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RemBertTokenizerFast

[[autodoc]] RemBertTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RemBertModel

[[autodoc]] RemBertModel
    - forward

## RemBertForCausalLM

[[autodoc]] RemBertForCausalLM
    - forward

## RemBertForMaskedLM

[[autodoc]] RemBertForMaskedLM
    - forward

## RemBertForSequenceClassification

[[autodoc]] RemBertForSequenceClassification
    - forward

## RemBertForMultipleChoice

[[autodoc]] RemBertForMultipleChoice
    - forward

## RemBertForTokenClassification

[[autodoc]] RemBertForTokenClassification
    - forward

## RemBertForQuestionAnswering

[[autodoc]] RemBertForQuestionAnswering
    - forward

## TFRemBertModel

[[autodoc]] TFRemBertModel
    - call

## TFRemBertForMaskedLM

[[autodoc]] TFRemBertForMaskedLM
    - call

## TFRemBertForCausalLM

[[autodoc]] TFRemBertForCausalLM
    - call

## TFRemBertForSequenceClassification

[[autodoc]] TFRemBertForSequenceClassification
    - call

## TFRemBertForMultipleChoice

[[autodoc]] TFRemBertForMultipleChoice
    - call

## TFRemBertForTokenClassification

[[autodoc]] TFRemBertForTokenClassification
    - call

## TFRemBertForQuestionAnswering

[[autodoc]] TFRemBertForQuestionAnswering
    - call