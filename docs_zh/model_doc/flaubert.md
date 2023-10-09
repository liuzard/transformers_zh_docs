<!--版权 © 2020 HuggingFace团队。版权所有。

根据Apache许可证版本2.0（“许可证”）获得许可; 除非符合许可证，否则你不得使用此文件。你可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样”分发的，

不提供任何明示或暗示的担保或条件。有关许可证的特定语言，请参阅

特定于许可证的限制。

⚠️请注意，此文件是用Markdown编写的，但包含我们的文档构建器（类似于MDX）的特定语法，可能在你的Markdown查看器中无法正常显示。

-->

# FlauBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=flaubert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-flaubert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/flaubert_small_cased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概览

FlauBERT模型是由Hang Le等人在论文[FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372)中提出的。它是一个使用掩码语言建模（MLM）目标（类似于BERT）进行预训练的transformer模型。

论文的摘要如下：

*语言模型已成为实现许多不同自然语言处理（NLP）任务最先进结果的关键步骤。利用当今可用的大量无标签文本，它们提供了一种有效的方式来预训练连续词表示，可以在下游任务中进行微调，以及它们在句子级的上下文化。在英语中，这已被广泛证明使用上下文化表示（Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018; Radford et al., 2018; Devlin et al.,
2019; Yang et al., 2019b）。在本文中，我们介绍并共享FlauBERT，这是一个在非常大和异构的法语语料库上学习的模型。使用新的法国国家科学研究中心（CNRS）Jean Zay超级计算机，训练了不同大小的模型。我们将法语语言模型应用于各种NLP任务（文本分类、释义、自然语言推理、解析、词义消歧）并显示它们大部分时间胜过其他预训练方法。FlauBERT的不同版本以及用于下游任务的统一评估协议，称为FLUE（French Language Understanding Evaluation），已与研究界共享，用于进一步的可重现实验。*

该模型由[formiel](https://huggingface.co/formiel)贡献。原始代码可以在[此处](https://github.com/getalp/Flaubert)找到。

提示：
- 与RoBERTa类似，没有句子顺序预测（只在MLM目标上进行训练）。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## FlaubertConfig

[[autodoc]] FlaubertConfig

## FlaubertTokenizer

[[autodoc]] FlaubertTokenizer

## FlaubertModel

[[autodoc]] FlaubertModel
    - forward

## FlaubertWithLMHeadModel

[[autodoc]] FlaubertWithLMHeadModel
    - forward

## FlaubertForSequenceClassification

[[autodoc]] FlaubertForSequenceClassification
    - forward

## FlaubertForMultipleChoice

[[autodoc]] FlaubertForMultipleChoice
    - forward

## FlaubertForTokenClassification

[[autodoc]] FlaubertForTokenClassification
    - forward

## FlaubertForQuestionAnsweringSimple

[[autodoc]] FlaubertForQuestionAnsweringSimple
    - forward

## FlaubertForQuestionAnswering

[[autodoc]] FlaubertForQuestionAnswering
    - forward

## TFFlaubertModel

[[autodoc]] TFFlaubertModel
    - call

## TFFlaubertWithLMHeadModel

[[autodoc]] TFFlaubertWithLMHeadModel
    - call

## TFFlaubertForSequenceClassification

[[autodoc]] TFFlaubertForSequenceClassification
    - call

## TFFlaubertForMultipleChoice

[[autodoc]] TFFlaubertForMultipleChoice
    - call

## TFFlaubertForTokenClassification

[[autodoc]] TFFlaubertForTokenClassification
    - call

## TFFlaubertForQuestionAnsweringSimple

[[autodoc]] TFFlaubertForQuestionAnsweringSimple
    - call