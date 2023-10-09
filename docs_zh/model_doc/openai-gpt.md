<!--
版权所有 2020 年 HuggingFace 团队。版权所有。

根据 Apache 许可证第 2 版（“许可证”），您不得不遵守以下内容使用此文件，除非符合许可证的要求。
您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，否则根据许可证分发的软件默认处于“按原样”状态，
不附带任何明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言管理权和限制。

⚠️ 请注意，此文件采用 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，
可能在您的 Markdown 查看器中无法正确呈现。

-->

# OpenAI GPT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=openai-gpt">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-openai--gpt-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/openai-gpt">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

OpenAI GPT 模型最初由 Alec Radford、Karthik Narasimhan、Tim Salimans 和 Ilya Sutskever 在
[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
一文中提出。它是一个使用语言建模预训练的因果（单向）Transformer 模型，
在一个具有长期依赖关系的大型语料库（多伦多图书语料库）上进行预训练。

论文中的摘要如下：

*自然语言理解包括多种不同的任务，如文本蕴含、问答、语义相似度评估和文档分类。虽然大量的未标记文本语料库丰富，
但用于学习这些特定任务的标记数据却很少，这使得对于有区分力的训练模型来说，其性能表现较差。我们通过对未标记文本的多样化语料库进行生成性预训练以及接下来对每个特定任务的判别性微调，证明在这些任务上可以获得很大的收益。与以往的方法相比，我们在微调过程中利用了任务感知的输入转换，以实现有效的迁移学习，同时对模型架构的修改要求最小。我们示范了我们的方法在自然语言理解的广泛基准测试中的有效性。我们的通用任务无关模型在 12 个研究任务中有 9 个取得了实质性改进，超过了以各任务为目标构建架构的有区分性训练模型的现有技术水平。*

提示：

- GPT 是一个具有绝对位置嵌入的模型，因此通常建议在右侧对输入进行填充，而不是左侧。
- GPT 使用因果（casual）语言建模（CLM）目标进行训练，因此能够有效地预测序列中的下一个标记。利用这个特性，GPT-2 可以生成句法连贯的文本，可以在 *run_generation.py* 示例脚本中观察到。

[Write With Transformer](https://transformer.huggingface.co/doc/gpt) 是由 Hugging Face 创建和托管的一个网页应用程序，展示了几种模型的生成能力，其中包括 GPT。

此模型由 [thomwolf](https://huggingface.co/thomwolf) 贡献。原始代码可在 [此处](https://github.com/openai/finetune-transformer-lm) 找到。

注意：

如果您想要重现 *OpenAI GPT* 论文中的原始分词过程，您需要安装 `ftfy` 和 `SpaCy`：

```bash
pip install spacy ftfy==4.4.3
python -m spacy download en
```

如果您没有安装 `ftfy` 和 `SpaCy`，[`OpenAIGPTTokenizer`] 将默认使用 BERT 的 `BasicTokenizer` 进行分词，然后使用字节对编码（对于大多数用途来说应该没问题，不用担心）。

## 资源

以下是官方 Hugging Face 和社区（用 🌎 表示）资源列表，可以帮助您入门 OpenAI GPT。如果您有兴趣提交资源以包含在这里，请随时打开拉取请求，我们会进行审核！该资源应该展示出一些新的东西，而不是重复现有的资源。

<PipelineTag pipeline="text-classification"/>

- [使用 SetFit 在文本分类任务中胜过 OpenAI GPT-3 的博文](https://www.philschmid.de/getting-started-setfit)。
- 参见：[文本分类任务指南](../tasks/sequence_classification)。

<PipelineTag pipeline="text-generation"/>

- 介绍如何[使用 Hugging Face 对非英语 GPT-2 模型进行微调的博客](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface)。
- [使用 Transformers 进行语言生成的不同解码方法](https://huggingface.co/blog/how-to-generate)与 GPT-2。
- 关于从头开始训练 [CodeParrot 🦜](https://huggingface.co/blog/codeparrot)（一个大型 GPT-2 模型）的博客。
- [使用 TensorFlow 和 XLA 加速文本生成的博客](https://huggingface.co/blog/tf-xla-generate)与 GPT-2。
- [如何使用 Megatron-LM 训练语言模型](https://huggingface.co/blog/megatron-training)与 GPT-2 模型。
- 介绍如何[对 GPT2 进行微调以生成您最喜爱的艺术家风格的歌词的笔记本](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb)。🌎
- 介绍如何[对 GPT2 进行微调以生成与您最喜欢的 Twitter 用户风格相似的推文的笔记本](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb)。🌎
- 🤗 Hugging Face 课程中关于[因果语言建模](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch)的章节。
- 此 [因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)、[文本生成示例脚本](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) 都支持 [`OpenAIGPTLMHeadModel`]。
- 此 [因果语言建模示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) 和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb) 都支持 [`TFOpenAIGPTLMHeadModel`]。
- 参见：[因果语言建模任务指南](../tasks/language_modeling)。

<PipelineTag pipeline="token-classification"/>

- 关于 [Byte-Pair 编码分词](https://huggingface.co/course/en/chapter6/5)的课程材料。

## OpenAIGPTConfig

[[autodoc]] OpenAIGPTConfig

## OpenAIGPTTokenizer

[[autodoc]] OpenAIGPTTokenizer
    - save_vocabulary

## OpenAIGPTTokenizerFast

[[autodoc]] OpenAIGPTTokenizerFast

## OpenAI 特定输出

[[autodoc]] models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput

[[autodoc]] models.openai.modeling_tf_openai.TFOpenAIGPTDoubleHeadsModelOutput

## OpenAIGPTModel

[[autodoc]] OpenAIGPTModel
    - forward

## OpenAIGPTLMHeadModel

[[autodoc]] OpenAIGPTLMHeadModel
    - forward

## OpenAIGPTDoubleHeadsModel

[[autodoc]] OpenAIGPTDoubleHeadsModel
    - forward

## OpenAIGPTForSequenceClassification

[[autodoc]] OpenAIGPTForSequenceClassification
    - forward

## TFOpenAIGPTModel

[[autodoc]] TFOpenAIGPTModel
    - call

## TFOpenAIGPTLMHeadModel

[[autodoc]] TFOpenAIGPTLMHeadModel
    - call

## TFOpenAIGPTDoubleHeadsModel

[[autodoc]] TFOpenAIGPTDoubleHeadsModel
    - call

## TFOpenAIGPTForSequenceClassification

[[autodoc]] TFOpenAIGPTForSequenceClassification
    - call
-->