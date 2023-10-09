<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（"许可证"）授权;除非符合许可证规定否则禁止使用本文件。

你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，依据许可证分发的软件是基于"按原样" BASIS，不提供任何形式的担保或条件，明示或暗示。
有关许可证下的特定语言的明示或暗示的任何形式担保和条件，请参阅许可证。

⚠️ 注意，此文件采用Markdown格式，但包含我们doc-builder（类似于MDX）的特定语法，可能无法在Markdown阅读器中正常呈现。

-->

# XLM-RoBERTa

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=xlm-roberta">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-xlm--roberta-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/xlm-roberta-base">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概览

XLM-RoBERTa模型是由Alexis Conneau，Kartikay Khandelwal，Naman Goyal，Vishrav Chaudhary，Guillaume
Wenzek，Francisco Guzmán，Edouard Grave，Myle Ott，Luke Zettlemoyer和Veselin Stoyanov在论文[Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)中提出的。它基于Facebook于2019年发布的RoBERTa模型。它是一个大型的多语言语言模型，训练了2.5TB的经过筛选的CommonCrawl数据。

论文摘要如下：

*本文表明，大规模预训练多语言语言模型可以显著提高广泛的跨语言转移任务的性能。我们在一百种语言上训练了一种基于Transformer的屏蔽语言模型，使用了2 TB以上的经过筛选的CommonCrawl数据。我们的模型名为XLM-R，在各种跨语言基准测试中，包括XNLI平均准确率提高了13.8％，MLQA平均F1分数提高了12.3％，NER平均F1分数提高了2.1％。 XLM-R在资源匮乏的语言上表现出色，相对于之前的XLM模型，Swahili的XNLI准确率提高了11.8％，乌尔都语提高了9.2％。我们还详细评估了实现这些增益所需的关键因素，包括（1）正向转移和容量稀释之间的权衡和（2）大规模高资源和低资源语言的性能。最后，我们首次展示了在不牺牲每种语言的性能的情况下进行多语言建模的可能性；在GLUE和XNLI基准测试中，XLM-R与强大的单语模型具有很强的竞争力。我们将公开提供XLM-R的代码、数据和模型。*

提示：

- XLM-RoBERTa是在100种不同语言上训练的多语言模型。与一些XLM多语言模型不同，它不需要`lang`张量来判断使用的是哪种语言，并且应该能够从输入id确定正确的语言。
- 使用了RoBERTa在XLM方法上的技巧，但不使用翻译语言建模目标。它仅对来自一种语言的句子进行屏蔽语言建模。
- 此实现与RoBERTa相同。有关用法示例以及输入和输出的相关信息，请参阅[RoBERTa的文档](roberta)。

此模型由[stefan-it](https://huggingface.co/stefan-it)贡献。原始代码可以在[此处](https://github.com/pytorch/fairseq/tree/master/examples/xlmr)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）资源列表，以帮助你入门XLM-RoBERTa。如果你有兴趣提交资源以包含在此处，请随时打开Pull Request，我们将进行审查！该资源应该展示出一些新东西，而不是重复现有资源。

<PipelineTag pipeline="text-classification"/>

- 有关如何在AWS上使用Habana Gaudi [对XLM RoBERTa进行多类分类微调的博文](https://www.philschmid.de/habana-distributed-training)
- [`XLMRobertaForSequenceClassification`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)支持。
- [`TFXLMRobertaForSequenceClassification`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)支持。
- [`FlaxXLMRobertaForSequenceClassification`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb)支持。
- [文本分类](https://huggingface.co/docs/transformers/tasks/sequence_classification)章节的🤗 Hugging Face任务指南。
- [文本分类任务指南](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [`XLMRobertaForTokenClassification`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)支持。
- [`TFXLMRobertaForTokenClassification`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)支持。
- [`FlaxXLMRobertaForTokenClassification`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification)支持。
- [标记分类](https://huggingface.co/course/chapter7/2?fw=pt)章节的🤗 Hugging Face课程。
- [标记分类任务指南](../tasks/token_classification)

<PipelineTag pipeline="text-generation"/>

- [`XLMRobertaForCausalLM`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)支持。
- [因果语言模型](https://huggingface.co/docs/transformers/tasks/language_modeling)章节的🤗 Hugging Face任务指南。
- [因果语言模型任务指南](../tasks/language_modeling)

<PipelineTag pipeline="fill-mask"/>

- [`XLMRobertaForMaskedLM`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)支持。
- [`TFXLMRobertaForMaskedLM`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)支持。
- [`FlaxXLMRobertaForMaskedLM`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb)支持。
- [屏蔽语言模型](https://huggingface.co/course/chapter7/3?fw=pt)章节的🤗 Hugging Face课程。
- [屏蔽语言模型](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`XLMRobertaForQuestionAnswering`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)支持。
- [`TFXLMRobertaForQuestionAnswering`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)支持。
- [`FlaxXLMRobertaForQuestionAnswering`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering)支持。
- [问答](https://huggingface.co/course/chapter7/7?fw=pt)章节的🤗 Hugging Face课程。
- [问答任务指南](../tasks/question_answering)

**多选**

- [`XLMRobertaForMultipleChoice`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)支持。
- [`TFXLMRobertaForMultipleChoice`]由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)支持。
- [多选题任务指南](../tasks/multiple_choice)

🚀 部署

- 有关如何在AWS Lambda上[部署无服务器的XLM RoBERTa](https://www.philschmid.de/multilingual-serverless-xlm-roberta-with-huggingface)的博文。

## XLMRobertaConfig

[[autodoc]] XLMRobertaConfig

## XLMRobertaTokenizer

[[autodoc]] XLMRobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLMRobertaTokenizerFast

[[autodoc]] XLMRobertaTokenizerFast

## XLMRobertaModel

[[autodoc]] XLMRobertaModel
    - forward

## XLMRobertaForCausalLM

[[autodoc]] XLMRobertaForCausalLM
    - forward

## XLMRobertaForMaskedLM

[[autodoc]] XLMRobertaForMaskedLM
    - forward

## XLMRobertaForSequenceClassification

[[autodoc]] XLMRobertaForSequenceClassification
    - forward

## XLMRobertaForMultipleChoice

[[autodoc]] XLMRobertaForMultipleChoice
    - forward

## XLMRobertaForTokenClassification

[[autodoc]] XLMRobertaForTokenClassification
    - forward

## XLMRobertaForQuestionAnswering

[[autodoc]] XLMRobertaForQuestionAnswering
    - forward

## TFXLMRobertaModel

[[autodoc]] TFXLMRobertaModel
    - call

## TFXLMRobertaForCausalLM

[[autodoc]] TFXLMRobertaForCausalLM
    - call

## TFXLMRobertaForMaskedLM

[[autodoc]] TFXLMRobertaForMaskedLM
    - call

## TFXLMRobertaForSequenceClassification

[[autodoc]] TFXLMRobertaForSequenceClassification
    - call

## TFXLMRobertaForMultipleChoice

[[autodoc]] TFXLMRobertaForMultipleChoice
    - call

## TFXLMRobertaForTokenClassification

[[autodoc]] TFXLMRobertaForTokenClassification
    - call

## TFXLMRobertaForQuestionAnswering

[[autodoc]] TFXLMRobertaForQuestionAnswering
    - call

## FlaxXLMRobertaModel

[[autodoc]] FlaxXLMRobertaModel
    - __call__

## FlaxXLMRobertaForCausalLM

[[autodoc]] FlaxXLMRobertaForCausalLM
    - __call__

## FlaxXLMRobertaForMaskedLM

[[autodoc]] FlaxXLMRobertaForMaskedLM
    - __call__

## FlaxXLMRobertaForSequenceClassification

[[autodoc]] FlaxXLMRobertaForSequenceClassification
    - __call__

## FlaxXLMRobertaForMultipleChoice

[[autodoc]] FlaxXLMRobertaForMultipleChoice
    - __call__

## FlaxXLMRobertaForTokenClassification

[[autodoc]] FlaxXLMRobertaForTokenClassification
    - __call__

## FlaxXLMRobertaForQuestionAnswering

[[autodoc]] FlaxXLMRobertaForQuestionAnswering
    - __call__