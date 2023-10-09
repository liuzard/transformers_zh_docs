版权 © 2021 The HuggingFace Team。保留所有权利。

根据Apache License, Version 2.0许可证（"许可证"），仅在符合许可证的前提下，才能使用此文件。
你可以在以下链接获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件是基于"按原样"的基础分发的，没有任何明示或暗示的担保或条件，包括但不限于具体目的的适销性和合格性。
有关许可证的特定语言，请在许可证下查看。

⚠️ 请注意，此文件以Markdown格式编写，但包含我们文档构建器（类似于MDX）的特定语法，可能在Markdown查看器中无法正确显示。

# GPT Neo

## 概述

GPTNeo模型由Sid Black、Stella Biderman、Leo Gao、Phil Wang和Connor Leahy在[EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo)存储库中发布。它是一个在[Pile](https://pile.eleuther.ai/)数据集上训练的类似GPT2的因果语言模型。

该结构与GPT2类似，但GPT Neo在每个其他层中使用窗口大小为256个标记的局部注意力。

此模型由[valhalla](https://huggingface.co/valhalla)贡献。

### 生成文本

可以使用GPT Neo模型的`generate()`方法生成文本。

```python
>>> from transformers import GPTNeoForCausalLM, GPT2Tokenizer

>>> model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
>>> tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

>>> prompt = (
...     "惊人的发现中，科学家发现了一群生活在安第斯山脉一个偏远、以前未被探索过的山谷中的独角兽。更令研究人员感到惊讶的是，这些独角兽会说一口流利的英语。"
... )

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [因果语言建模任务指南](../tasks/language_modeling)

## GPTNeoConfig

[[autodoc]] GPTNeoConfig

## GPTNeoModel

[[autodoc]] GPTNeoModel
    - forward

## GPTNeoForCausalLM

[[autodoc]] GPTNeoForCausalLM
    - forward

## GPTNeoForQuestionAnswering

[[autodoc]] GPTNeoForQuestionAnswering
    - forward

## GPTNeoForSequenceClassification

[[autodoc]] GPTNeoForSequenceClassification
    - forward

## GPTNeoForTokenClassification

[[autodoc]] GPTNeoForTokenClassification
    - forward

## FlaxGPTNeoModel

[[autodoc]] FlaxGPTNeoModel
    - __call__

## FlaxGPTNeoForCausalLM

[[autodoc]] FlaxGPTNeoForCausalLM
    - __call__