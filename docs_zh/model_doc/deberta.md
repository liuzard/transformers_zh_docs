<!--
版权所有2020年The HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）授权；除非遵守许可证，否则您不得使用该文件。您可以在以下位置获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按“按原样”分发，不附带任何明示或默示的担保或条件。请参阅许可证以了解许可证下的特定语言和限制。

⚠️ 请注意，此文件是Markdown格式，但包含特定于我们的doc-builder（类似于MDX）的语法，这可能在您的Markdown查看器中无法正常呈现。

-->

# DeBERTa

## 概述

DeBERTa模型由Pengcheng He，Xiaodong Liu，Jianfeng Gao，Weizhu Chen在[DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)中提出。它基于Google于2018年发布的BERT模型和Facebook于2019年发布的RoBERTa模型。

它在RoBERTa的基础上使用了解耦的注意力和增强的掩码解码器训练，训练数据使用了RoBERTa的一半。

论文中的摘要如下：

*预训练神经语言模型在许多自然语言处理（NLP）任务的性能显著提高。在本文中，我们提出了一种新的模型架构DeBERTa（Decoding-enhanced BERT with disentangled attention），通过两种新技术改进了BERT和RoBERTa模型。第一种是解耦的注意力机制，其中使用两个向量来表示每个单词，分别对其内容和位置进行编码，并使用这些单词之间的分立矩阵计算注意力权重。其次，使用增强的掩码解码器代替输出softmax层对模型预训练的掩码标记进行预测。我们展示了这两种技术显著提高了模型预训练的效率和下游任务的性能。与RoBERTa-Large相比，在一半的训练数据上训练的DeBERTa模型在各种NLP任务上表现一致更好，MNLI的提升为+0.9%（90.2% vs. 91.1%），SQuAD v2.0的提升为+2.3%（88.4% vs. 90.7%），RACE的提升为+3.6%（83.2% vs. 86.8%）。DeBERTa的源代码和预训练模型将在https://github.com/microsoft/DeBERTa上公开提供。*

此模型由[DeBERTa](https://huggingface.co/DeBERTa)贡献。这个模型的TF 2.0实现由[kamalkraj](https://huggingface.co/kamalkraj)贡献。原始代码可以在[这里](https://github.com/microsoft/DeBERTa)找到。

## 资源

以下是官方Hugging Face和社区（标有🌎）资源的列表，可帮助您开始使用DeBERTa。如果您有兴趣提交资源以包含在此处，请随时提交拉取请求，我们会进行审查！资源理想情况下应该展示出一些新的东西，而不是重复现有的资源。

<PipelineTag pipeline="text-classification"/>

- 有关如何使用DeBERTa加速大型模型训练的博文：[Accelerate Large Model Training using DeepSpeed](https://huggingface.co/blog/accelerate-deepspeed) 。
- 有关如何使用DeBERTa进行[机器学习的超级客户服务](https://huggingface.co/blog/supercharge-customer-service-with-machine-learning) 的博文。
- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)支持`DebertaForSequenceClassification`。
- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)支持`TFDebertaForSequenceClassification`。
- [文本分类任务指南](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification" />

- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)支持`DebertaForTokenClassification`。
- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)支持`TFDebertaForTokenClassification`。
- 🤗 Hugging Face课程的[Token classification](https://huggingface.co/course/chapter7/2?fw=pt)章节。
- 🤗 Hugging Face课程的[Byte-Pair Encoding tokenization](https://huggingface.co/course/chapter6/5?fw=pt)章节。
- [标记分类任务指南](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)支持`DebertaForMaskedLM`。
- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)支持`TFDebertaForMaskedLM`。
- 🤗 Hugging Face课程的[Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt)章节。
- [遮罩语言建模任务指南](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)支持`DebertaForQuestionAnswering`。
- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)支持`TFDebertaForQuestionAnswering`。
- 🤗 Hugging Face课程的[Question answering](https://huggingface.co/course/chapter7/7?fw=pt)章节。
- [问答任务指南](../tasks/question_answering)

## DebertaConfig

[[autodoc]] DebertaConfig

## DebertaTokenizer

[[autodoc]] DebertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## DebertaTokenizerFast

[[autodoc]] DebertaTokenizerFast
    - build_inputs_with_special_tokens
    - create_token_type_ids_from_sequences

## DebertaModel

[[autodoc]] DebertaModel
    - forward

## DebertaPreTrainedModel

[[autodoc]] DebertaPreTrainedModel

## DebertaForMaskedLM

[[autodoc]] DebertaForMaskedLM
    - forward

## DebertaForSequenceClassification

[[autodoc]] DebertaForSequenceClassification
    - forward

## DebertaForTokenClassification

[[autodoc]] DebertaForTokenClassification
    - forward

## DebertaForQuestionAnswering

[[autodoc]] DebertaForQuestionAnswering
    - forward

## TFDebertaModel

[[autodoc]] TFDebertaModel
    - call

## TFDebertaPreTrainedModel

[[autodoc]] TFDebertaPreTrainedModel
    - call

## TFDebertaForMaskedLM

[[autodoc]] TFDebertaForMaskedLM
    - call

## TFDebertaForSequenceClassification

[[autodoc]] TFDebertaForSequenceClassification
    - call

## TFDebertaForTokenClassification

[[autodoc]] TFDebertaForTokenClassification
    - call

## TFDebertaForQuestionAnswering

[[autodoc]] TFDebertaForQuestionAnswering
    - call
-->