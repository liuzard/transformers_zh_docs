<!--版权所有 © 2020 Hugging Face团队.

根据Apache License, Version 2.0 (许可证)，除非符合许可证，否则您不得使用此文件。您可以从以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非根据适用法律要求或书面同意，否则以“按现状”的形式发布的软件在未附带任何明示或暗示的担保或条件的情况下分发。有关特定语言的详情请参阅许可协议。

请注意，此文件是Markdown格式，但包含特定于我们的文档构建器（类似于MDX）的语法，可能无法在Markdown查看器中正确渲染。-->

# DeBERTa-v2

## 概述

DeBERTa模型是由[Pengcheng He](https://arxiv.org/abs/2006.03654)，[Xiaodong Liu](https://arxiv.org/abs/2006.03654)，[Jianfeng Gao](https://arxiv.org/abs/2006.03654)，[Weizhu Chen](https://arxiv.org/abs/2006.03654)在论文“DeBERTa：具有解耦关注的BERT解码增强模型”中提出的。它基于Google于2018年发布的BERT模型和Facebook于2019年发布的RoBERTa模型。

DeBERTa在RoBERTa的基础上使用了解耦关注和增强的掩码解码器训练，其中使用了RoBERTa训练数据的一半。

论文摘要如下：

*最近，预训练的神经语言模型在许多自然语言处理（NLP）任务的性能方面有了显著提高。在本文中，我们提出了一种新的模型架构DeBERTa（具有解耦关注的BERT解码增强模型），它使用两种新技术改进了BERT和RoBERTa模型。第一种技术是解耦关注机制，其中每个单词使用两个向量表示其内容和位置，并使用解耦矩阵来计算单词之间的关注权重。第二种技术是使用增强的掩码解码器来替换输出的softmax层，以预测模型预训练的掩码标记。我们表明，这两种技术显著提高了模型预训练的效率和下游任务的性能。与RoBERTa-Large相比，使用一半训练数据训练的DeBERTa模型在广泛的NLP任务上始终表现更好，在MNLI上提高了+0.9%（90.2% vs 91.1%），在SQuAD v2.0上提高了+2.3%（88.4% vs 90.7%）和在RACE上提高了+3.6%（83.2% vs 86.8%）。DeBERTa的代码和预训练模型将在https://github.com/microsoft/DeBERTa上公开。*

下面的信息可以直接在[原始实现存储库](https://github.com/microsoft/DeBERTa)上看到。DeBERTa v2是DeBERTa模型的第二个版本。它包括用于SuperGLUE单模型提交的15亿模型，达到了89.9，相比人类基准的89.8。您可以在作者的[博客](https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/)中找到有关此提交的更多详细信息。

v2的新功能：

- **词汇表** 在v2中，标记器的词汇表更改为使用从训练数据中构建的128K大小的新词汇表。标记器不再使用基于GPT2的标记器，而是使用基于[sentencepiece](https://github.com/google/sentencepiece)实现的标记器。
- **nGiE（ngram引入输入编码）** DeBERTa-v2模型在第一个Transformer层旁边使用了一个额外的卷积层，以更好地学习输入标记的局部依赖性。
- **在注意力层中，将位置投影矩阵与内容投影矩阵共享** 基于之前的实验，这可以节省参数而不影响性能。
- **应用桶编码相对位置** DeBERTa-v2模型使用对数桶编码相对位置，类似于T5。
- **900M模型和15亿模型** 提供了两种附加模型大小：900M和15亿，在下游任务的性能方面有着显著改进。

此模型由[DeBERTa](https://huggingface.co/DeBERTa)贡献。此模型的TF 2.0实现由[kamalkraj](https://huggingface.co/kamalkraj)贡献。原始代码在[此处](https://github.com/microsoft/DeBERTa)可找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## DebertaV2Config

[[autodoc]] DebertaV2Config

## DebertaV2Tokenizer

[[autodoc]] DebertaV2Tokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## DebertaV2TokenizerFast

[[autodoc]] DebertaV2TokenizerFast
    - build_inputs_with_special_tokens
    - create_token_type_ids_from_sequences

## DebertaV2Model

[[autodoc]] DebertaV2Model
    - forward

## DebertaV2PreTrainedModel

[[autodoc]] DebertaV2PreTrainedModel
    - forward

## DebertaV2ForMaskedLM

[[autodoc]] DebertaV2ForMaskedLM
    - forward

## DebertaV2ForSequenceClassification

[[autodoc]] DebertaV2ForSequenceClassification
    - forward

## DebertaV2ForTokenClassification

[[autodoc]] DebertaV2ForTokenClassification
    - forward

## DebertaV2ForQuestionAnswering

[[autodoc]] DebertaV2ForQuestionAnswering
    - forward

## DebertaV2ForMultipleChoice

[[autodoc]] DebertaV2ForMultipleChoice
    - forward

## TFDebertaV2Model

[[autodoc]] TFDebertaV2Model
    - call

## TFDebertaV2PreTrainedModel

[[autodoc]] TFDebertaV2PreTrainedModel
    - call

## TFDebertaV2ForMaskedLM

[[autodoc]] TFDebertaV2ForMaskedLM
    - call

## TFDebertaV2ForSequenceClassification

[[autodoc]] TFDebertaV2ForSequenceClassification
    - call

## TFDebertaV2ForTokenClassification

[[autodoc]] TFDebertaV2ForTokenClassification
    - call

## TFDebertaV2ForQuestionAnswering

[[autodoc]] TFDebertaV2ForQuestionAnswering
    - call

## TFDebertaV2ForMultipleChoice

[[autodoc]] TFDebertaV2ForMultipleChoice
    - call