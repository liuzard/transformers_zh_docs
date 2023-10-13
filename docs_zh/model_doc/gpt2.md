<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”）的条款，除非符合许可证的规定
你将无法使用此文件。

你可以获得许可证的副本，在下面的链接中获得该许可证

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件将按
“按原样”分布，没有任何形式的担保或条件，无论是明示还是暗示。有关许可下的特定语言的详细信息
你可能的限制。

⚠️请注意，此文件采用Markdown格式，但包含我们的doc-builder的特定语法（类似于MDX），可能无法正确显示在你的Markdown查看器中。

-->

# 开放人工智能 GPT2

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=gpt2">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-gpt2-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/gpt2">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

OpenAI GPT-2模型由Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei和Ilya Sutskever提出，详细内容请见[语言模型是无监督的多任务学习器](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)，来自[OpenAI](https://huggingface.co/openai)。它是一个因果（单向）变换器，使用了很大的语料库（约40GB的文本数据）进行了语言模型预训练。

论文摘要如下：

*GPT-2是一个基于变换器的大型语言模型，具有15亿个参数，使用包含800万个网页的数据集[1]进行了训练。GPT-2的训练目标很简单：根据文本中之前的所有单词来预测下一个单词。数据集的多样性使得这个简单的目标包含了许多任务在不同领域中的自然示范。 GPT-2是GPT的直接扩大，参数数量是GPT的10倍以上，使用了10倍以上的数据进行训练。*

提示：

- GPT-2是一个具有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧填充输入。
- GPT-2是通过因果语言建模（CLM）目标进行训练的，因此在预测序列中的下一个标记时具有强大的能力。利用这个特性，GPT-2能够生成句法连贯的文本，可以在*run_generation.py*示例脚本中观察到。
- 该模型可以接受*past_key_values*（对于PyTorch）或*past*（对于TF）作为输入，其中包含之前计算的键/值注意对。使用这个（*past_key_values*或*past*）值可以防止模型在文本生成的上下文中重新计算已预先计算的值。有关使用说明，请参阅PyTorch的[`GPT2Model.forward`]方法的*past_key_values*参数，或TF的[`TFGPT2Model.call`]方法的*past*参数。
- 启用*scale_attn_by_inverse_layer_idx*和*reorder_and_upcast_attn*标志将应用[Mistral](https://github.com/stanford-crfm/mistral/)的训练稳定性改进（仅适用于PyTorch）。

[Write With Transformer](https://transformer.huggingface.co/doc/gpt2-large)是由Hugging Face创建和托管的网页应用程序，展示了多个模型的生成能力。GPT-2是其中之一，有五个不同的大小可用：small，medium，large，xl以及small模型的蒸馏版本：*distilgpt-2*。

此模型由[thomwolf](https://huggingface.co/thomwolf)贡献。原始代码可以在[这里](https://openai.com/blog/better-language-models/)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）资源列表，以帮助你开始使用GPT2。如果你希望提交资源以包含在此处，请随时提交Pull Request，我们将进行审核！资源应该是新颖的，而不是重复的现有资源。

<PipelineTag pipeline="text-generation"/>

- 一篇博客介绍如何[使用Hugging Face对非英语GPT-2模型进行微调](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface)。
- 一篇介绍[如何生成文本：使用Transformers的不同解码方法进行语言生成](https://huggingface.co/blog/how-to-generate)的博客，其中包括GPT-2。
- 一篇关于[从零开始训练CodeParrot 🦜](https://huggingface.co/blog/codeparrot)的大型GPT-2模型的博客。
- 一篇关于[如何使用TensorFlow和XLA快速生成文本](https://huggingface.co/blog/tf-xla-generate)的博客，其中包括GPT-2。
- 一篇关于[如何使用Megatron-LM训练语言模型](https://huggingface.co/blog/megatron-training)的博客，其中包括一个GPT-2模型。
- 一篇有关[如何使用GPT2微调生成你最喜爱的艺术家风格的歌词的笔记本](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb)。 🌎
- 一篇有关[如何使用GPT2微调生成你最喜爱的Twitter用户风格的推文的笔记本](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb)。 🌎
- [因果语言模型](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch)一章的🤗Hugging Face课程。
- [`GPT2LMHeadModel`]由此[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)、[文本生成示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)支持。
- [`TFGPT2LMHeadModel`]由此[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)支持。
- [`FlaxGPT2LMHeadModel`]由此[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb)支持。
- [文本分类任务指南](../tasks/sequence_classification)
- [token分类任务指南](../tasks/token_classification)
- [因果语言建模任务指南](../tasks/language_modeling)

## GPT2Config

[[autodoc]] GPT2Config

## GPT2Tokenizer

[[autodoc]] GPT2Tokenizer
    - save_vocabulary

## GPT2TokenizerFast

[[autodoc]] GPT2TokenizerFast

## GPT2特定的输出

[[autodoc]] models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput

[[autodoc]] models.gpt2.modeling_tf_gpt2.TFGPT2DoubleHeadsModelOutput

## GPT2模型

[[autodoc]] GPT2Model
    - forward

## GPT2LMHeadModel

[[autodoc]] GPT2LMHeadModel
    - forward

## GPT2DoubleHeadsModel

[[autodoc]] GPT2DoubleHeadsModel
    - forward

## GPT2ForQuestionAnswering

[[autodoc]] GPT2ForQuestionAnswering
    - forward

## GPT2ForSequenceClassification

[[autodoc]] GPT2ForSequenceClassification
    - forward

## GPT2ForTokenClassification

[[autodoc]] GPT2ForTokenClassification
    - forward

## TFGPT2Model

[[autodoc]] TFGPT2Model
    - call

## TFGPT2LMHeadModel

[[autodoc]] TFGPT2LMHeadModel
    - call

## TFGPT2DoubleHeadsModel

[[autodoc]] TFGPT2DoubleHeadsModel
    - call

## TFGPT2ForSequenceClassification

[[autodoc]] TFGPT2ForSequenceClassification
    - call

## TFSequenceClassifierOutputWithPast

[[autodoc]] modeling_tf_outputs.TFSequenceClassifierOutputWithPast

## TFGPT2Tokenizer

[[autodoc]] TFGPT2Tokenizer

## FlaxGPT2Model

[[autodoc]] FlaxGPT2Model
    - __call__

## FlaxGPT2LMHeadModel

[[autodoc]] FlaxGPT2LMHeadModel
    - __call__