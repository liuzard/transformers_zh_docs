<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）的规定，除非符合许可证的要求，否则你不得使用此文件。
你可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样”分发的，
不附带任何形式的明示或默示担保。请查看许可证以了解许可证下特定语言的权限和限制。

⚠️请注意，此文件是Markdown格式的，但包含我们doc-builder的特定语法（类似MDX），可能无法在你的Markdown查看器中正确显示。-->

# mT5

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=mt5">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-mt5-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/mt5-small-finetuned-arxiv-cs-finetuned-arxiv-cs-full">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

mT5模型是由Linting Xue，Noah Constant，Adam Roberts，Mihir Kale，Rami Al-Rfou，Aditya Siddhant，Aditya Barua，Colin Raffel提出的[mT5：一种大规模多语言预训练文本到文本的转换器](https://arxiv.org/abs/2010.11934)。

论文中的摘要如下：

*最近，“文本到文本转换转换器”（T5）利用统一的文本到文本格式和规模，在各种英语自然语言处理任务上取得了最先进的结果。在本文中，我们介绍了一个名为mT5的T5多语言变体，该变体在覆盖了101种语言的新的基于Common Crawl的数据集上进行了预训练。我们详细介绍了mT5的设计和修改训练，并展示了它在许多多语言基准数据集上的最先进性能。我们还描述了一种简单的技术，以防止零射设置下的“意外翻译”，在此设置中，生成模型选择（部分）将其预测转化为错误的语言。本文中使用的所有代码和模型检查点都是公开可用的。*

注意：mT5只是在[mC4](https://huggingface.co/datasets/mc4)上进行了预训练，没有进行任何监督训练。
因此，该模型必须在使用下游任务之前进行微调，这与原始的T5模型不同。
由于mT5是无监督预训练的，使用任务前缀在单任务微调中没有真正的优势。如果你正在进行多任务微调，应该使用前缀。

Google发布了以下变体：

- [google/mt5-small](https://huggingface.co/google/mt5-small)

- [google/mt5-base](https://huggingface.co/google/mt5-base)

- [google/mt5-large](https://huggingface.co/google/mt5-large)

- [google/mt5-xl](https://huggingface.co/google/mt5-xl)

- [google/mt5-xxl](https://huggingface.co/google/mt5-xxl)。

该模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。原始代码可以在[这里](https://github.com/google-research/multilingual-t5)找到。

## 文档资源

- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## MT5Config

[[autodoc]] MT5Config

## MT5Tokenizer

[[autodoc]] MT5Tokenizer

有关详细信息，请参见[`T5Tokenizer`]。

## MT5TokenizerFast

[[autodoc]] MT5TokenizerFast

有关详细信息，请参见[`T5TokenizerFast`]。

## MT5Model

[[autodoc]] MT5Model

## MT5ForConditionalGeneration

[[autodoc]] MT5ForConditionalGeneration

## MT5EncoderModel

[[autodoc]] MT5EncoderModel

## MT5ForSequenceClassification

[[autodoc]] MT5ForSequenceClassification

## MT5ForQuestionAnswering

[[autodoc]] MT5ForQuestionAnswering

## TFMT5Model

[[autodoc]] TFMT5Model

## TFMT5ForConditionalGeneration

[[autodoc]] TFMT5ForConditionalGeneration

## TFMT5EncoderModel

[[autodoc]] TFMT5EncoderModel

## FlaxMT5Model

[[autodoc]] FlaxMT5Model

## FlaxMT5ForConditionalGeneration

[[autodoc]] FlaxMT5ForConditionalGeneration

## FlaxMT5EncoderModel

[[autodoc]] FlaxMT5EncoderModel