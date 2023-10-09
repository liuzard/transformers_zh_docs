<!--
版权所有2023年拥抱面团小组。  

根据Apache许可证第2.0版（“许可证”）许可，除非符合许可证要求，否则其他情况下请勿使用此文件。你可以获取许可证的副本，请参见  
http://www.apache.org/licenses/LICENSE-2.0  

除非适用法律要求或书面同意，否则按“原样”分发的软件以“不提供任何保证或条件”，明示或默示地进行分发，本许可不受特定的语言条款约束。参见许可证中的规定以了解许可证下的特定语言许可和限制。

⚠️ 请注意，此文件以Markdown格式编写，但包含了我们的文档生成器（类似MDX）的特定语法，可能在你的Markdown查看器中无法正确呈现。

-->

# UMT5

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=umt5">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-mt5-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/mt5-small-finetuned-arxiv-cs-finetuned-arxiv-cs-full">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

UMT5模型是由Hyung Won Chung、Xavier Garcia、Adam Roberts、Yi Tay、Orhan Firat、Sharan Narang和Noah Constant在[UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining(论文链接)](https://openreview.net/forum?id=kXwdL1cWOAi)中提出的。

论文中的摘要如下：

*预训练的多语言大型语言模型通常使用启发式的温度采样方法来平衡不同语言。但是之前的工作没有系统评估不同预训练语言分布在模型规模方面的有效性。在本文中，我们提出了一种新的采样方法UniMax，它在明确限制每种语言语料库的重复次数的同时，提供了对头部语言的更均匀覆盖，并减少了对尾部语言的过拟合。我们进行了广泛的实验，测试了一系列在各种多语言基准测试中改变模型规模的采样策略。我们发现UniMax优于标准的温度采样，在模型规模增加时这种优势仍然存在。作为我们的贡献的一部分，我们发布了：(i) 一个改进和更新的mC4多语言语料库，包含107种语言的290万亿个字符，以及(ii)一系列使用UniMax采样训练的预训练umT5模型检查点。

提示：

- UMT5模型仅在[mC4](https://huggingface.co/datasets/mc4)上进行了预训练，没有进行任何监督训练。
因此，在将该模型用于下游任务之前，需要进行微调，这与原始的T5模型不同。
- 由于umT5是以无监督方式进行预训练的，使用任务前缀进行单任务微调实际上没有任何优势。如果进行多任务微调，应该使用前缀。

Google发布了以下变体：

- [google/umt5-small](https://huggingface.co/google/umt5-small)
- [google/umt5-base](https://huggingface.co/google/umt5-base)
- [google/umt5-xl](https://huggingface.co/google/umt5-xl)
- [google/umt5-xxl](https://huggingface.co/google/umt5-xxl)。

此模型由[agemagician](https://huggingface.co/agemagician)和[stefan-it](https://huggingface.co/stefan-it)贡献。原始代码可在[此处](https://github.com/google-research/t5x)找到。

有关更多提示、代码示例和笔记本，请参阅[T5的文档页面](t5)。

## 与mT5的区别？
`UmT5`基于mT5，每个层都计算了非共享的相对位置偏差。这意味着该模型设置了每层的`has_relative_bias`。
转换脚本也不同，因为该模型的保存格式是t5x的最新版本。

# 使用示例

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/umt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")

>>> inputs = tokenizer(
...     "A <extra_id_0> walks into a bar and orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>.",
...     return_tensors="pt",
... )
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs))
['<pad><extra_id_0>nyone who<extra_id_1> drink<extra_id_2> a<extra_id_3> alcohol<extra_id_4> A<extra_id_5> A. This<extra_id_6> I<extra_id_7><extra_id_52><extra_id_53></s>']
```

## UMT5Config

[[autodoc]] UMT5Config

## UMT5Model

[[autodoc]] UMT5Model
    - forward

## UMT5ForConditionalGeneration

[[autodoc]] UMT5ForConditionalGeneration
    - forward

## UMT5EncoderModel

[[autodoc]] UMT5EncoderModel
    - forward

## UMT5ForSequenceClassification

[[autodoc]] UMT5ForSequenceClassification
    - forward

## UMT5ForQuestionAnswering

[[autodoc]] UMT5ForQuestionAnswering
    - forward
