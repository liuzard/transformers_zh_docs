<!--版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）进行许可；除非符合许可证，否则不得使用此文件。你可以在下面链接中获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”基础分发的，不附带任何形式的明示或暗示的保证或条件。有关许可证下的特定语言的详细信息，请参阅许可证。

⚠️请注意，此文件采用Markdown格式，但包含特定于我们doc-builder（类似于MDX）的语法，可能在你的Markdown查看器中无法正确渲染。

-->

# OPT

## 概述

OPT模型是由Meta AI在《开放预训练Transformer语言模型》（Open Pre-trained Transformer Language Models）一文中提出的。OPT是一系列开源的大型因果语言模型，性能类似于GPT3。

论文摘要如下：

*大型语言模型通常经过数十万个计算天数的训练，展示了在零样本和少样本学习上的显著能力。由于它们的计算成本，要在没有重大资金的情况下复制这些模型是困难的。对于那些通过API可用的模型，无法访问完整的模型权重，这使得它们难以研究。我们提供了开放的预训练Transformer（OPT），这是一系列仅解码器的预训练Transformer，参数从125M到175B不等，我们希望完整而负责任地与感兴趣的研究人员共享。我们展示了OPT-175B与GPT-3相当，同时只需要1/7的碳足迹来进行开发。我们还发布了记录我们所面临的基础设施挑战的日志，并提供了用于处理所有发布模型的代码。*

提示：
- OPT与[`BartDecoder`]具有相同的架构。
- 与GPT2不同，OPT在每个提示的开头添加了EOS标记`</s>`。

该模型由[Arthur Zucker](https://huggingface.co/ArthurZ)、[Younes Belkada](https://huggingface.co/ybelkada)和[Patrick Von Platen](https://huggingface.co/patrickvonplaten)贡献。
原始代码可以在[此处](https://github.com/facebookresearch/metaseq)找到。

## 资源

以下是官方Hugging Face和社区（由🌎标记）资源的列表，可帮助你开始使用OPT。如果你有兴趣提交资源以被包括在此处，请随时发起Pull Request，我们会进行审核。该资源应具有展示出创新性而不是重复现有资源的理念。

<PipelineTag pipeline="text-generation" />

- [使用PEFT、bitsandbytes和Transformers对OPT进行微调](https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing)的笔记本。🌎
- [使用OPT的解码策略的博客文章](https://huggingface.co/blog/introducing-csearch#62-example-two---opt)。
- 🤗 Hugging Face课程中的[因果语言建模](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch)章节。
- 通过此[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)支持[`OPTForCausalLM`]。
- 通过此[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)支持[`TFOPTForCausalLM`]。
- 通过此[因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling)支持[`FlaxOPTForCausalLM`]。

<PipelineTag pipeline="text-classification" />

- [文本分类任务指南](sequence_classification.md)
- 通过此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)支持[`OPTForSequenceClassification`]。

<PipelineTag pipeline="question-answering" />

- 通过此[问题回答示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)支持[`OPTForQuestionAnswering`]。
- 🤗 Hugging Face课程中的[问题回答](https://huggingface.co/course/chapter7/7?fw=pt)章节。

⚡️ 推论

- 有关OPT的[如何通过PyTorch运行🤗 Accelerate处理非常大的模型](https://huggingface.co/blog/accelerate-large-models)的博客文章。

## OPTConfig

[[autodoc]] OPTConfig

## OPTModel

[[autodoc]] OPTModel
    - forward

## OPTForCausalLM

[[autodoc]] OPTForCausalLM
    - forward

## TFOPTModel

[[autodoc]] TFOPTModel
    - call

## TFOPTForCausalLM

[[autodoc]] TFOPTForCausalLM
    - call

## OPTForSequenceClassification

[[autodoc]] OPTForSequenceClassification
    - forward

## OPTForQuestionAnswering

[[autodoc]] OPTForQuestionAnswering
    - forward

## FlaxOPTModel

[[autodoc]] FlaxOPTModel
    - __call__


## FlaxOPTForCausalLM

[[autodoc]] FlaxOPTForCausalLM
    - __call__