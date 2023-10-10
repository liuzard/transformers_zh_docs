<!--
版权所有2020年HuggingFace团队。保留所有权利。

根据Apache License, Version 2.0 （“许可证”）获得许可; 你除非符合该许可证，否则不得使用此文件。你可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本

除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”分发的，没有任何明示或暗示的保证或条件。有关许可证的详细信息，可以查阅许可证中的特定语言。

⚠️注意，该文件是Markdown格式的，但包含我们的文档生成器的特定语法（类似于MDX），在你的Markdown查看器中可能无法正确显示。

-->

# CTRL

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=ctrl">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-ctrl-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/tiny-ctrl">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

CTRL模型在Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong 和 Richard Socher提出的论文 [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) 中被提出。它是一个因果（单向）变压器，使用语言建模在一个非常大的140GB文本数据语料库上进行了预训练，其中第一个标记保留作为控制码（例如链接、书籍、维基百科等）。

来自论文的摘要如下：

*大规模的语言模型展示了有希望的文本生成能力，但用户不能轻易地控制生成文本的特定方面。我们发布了CTRL，一个具有163亿参数的条件变压器语言模型，训练了一些控制代码，这些代码控制着风格、内容和任务特定的行为。控制代码派生自与原始文本自然共现的结构，保留无监督学习的优势，同时更加明确地控制文本的生成。这些代码还允许CTRL预测在给定序列情况下训练数据的哪些部分最有可能。这为通过基于模型的源归因方法分析大量数据提供了一种潜在方法。*

提示：

- CTRL使用控制代码生成文本：它要求以特定的单词、句子或链接开始生成连贯的文本。有关更多信息，请参阅 [原始实现](https://github.com/salesforce/ctrl)。
- CTRL是一个带有绝对位置嵌入的模型，因此通常建议将输入在右侧而不是左侧进行填充。
- CTRL通过因果语言建模（CLM）目标进行训练，因此在序列中通过预测下一个token的特征进行训练。利用这个特性，CTRL可以生成语法连贯的文本，例如在*run_generation.py*示例脚本中可以观察到。
- PyTorch模型可以接受`past_key_values`作为输入，这是先前计算的键/值注意力对。TensorFlow模型接受`past`作为输入。使用`past_key_values`值可以防止模型在文本生成的上下文中重新计算预先计算的值。有关此参数的用法，请参阅 [`forward`](model_doc/ctrl#transformers.CTRLModel.forward) 方法。

此模型由[keskarnitishr](https://huggingface.co/keskarnitishr)贡献。原始代码可以在[此处](https://github.com/salesforce/ctrl)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [因果语言建模任务指南](../tasks/language_modeling)

## CTRLConfig

[[autodoc]] CTRLConfig

## CTRLTokenizer

[[autodoc]] CTRLTokenizer
    - save_vocabulary

## CTRLModel

[[autodoc]] CTRLModel
    - forward

## CTRLLMHeadModel

[[autodoc]] CTRLLMHeadModel
    - forward

## CTRLForSequenceClassification

[[autodoc]] CTRLForSequenceClassification
    - forward

## TFCTRLModel

[[autodoc]] TFCTRLModel
    - call

## TFCTRLLMHeadModel

[[autodoc]] TFCTRLLMHeadModel
    - call

## TFCTRLForSequenceClassification

[[autodoc]] TFCTRLForSequenceClassification
    - call
-->