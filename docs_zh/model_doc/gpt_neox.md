<!--版权 2022 年 The HuggingFace 团队。保留所有权利。

根据 Apache 许可证第 2.0 版（以下简称“许可证”），除非符合许可证规定，否则不得使用此文件。你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，不附带任何明示或默示的保证或条件。有关特定语言的详细信息，请参阅许可证。

⚠️ 请注意，此文件是 Markdown 格式，但包含了我们 doc-builder 的特定语法（类似于 MDX），可能无法在你的 Markdown 查看器中正确显示。

-->

# GPT-NeoX

## 概述

我们介绍 GPT-NeoX-20B，这是一个在 Pile 上训练的具有 200 亿参数的自回归语言模型，其权重将通过一种宽松许可证免费提供给公众。据我们所知，这是公开提供权重的最大稠密自回归模型。在这项工作中，我们描述了 GPT-NeoX-20B 的架构和训练，并在一系列语言理解、数学和知识任务上评估其性能。我们发现，与相同大小的 GPT-3 和 FairSeq 模型相比，GPT-NeoX-20B 是一个特别强大的少样本理解器，并且在五样本评估时性能提升更多。我们在 [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox) 上开源了训练和评估代码，以及模型权重。

该模型的开发由 Sid Black、Stella Biderman 和 Eric Hallahan 领导，模型是在 [CoreWeave](https://www.coreweave.com/) 的慷慨支持下训练的。

GPT-NeoX-20B 使用 fp16 进行训练，因此建议按以下方式初始化模型：

```python
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half().cuda()
```

GPT-NeoX-20B 还具有不同的分词器，与 GPT-J-6B 和 GPT-Neo 使用的分词器不同。新的分词器为空格字符分配了额外的标记，使得该模型在代码生成等特定任务中更加合适。

### 生成

可以使用 `generate()` 方法来使用 GPT Neo 模型生成文本。

``` python
>>> from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

>>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
>>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

>>> prompt = "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI."

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

- [因果语言模型任务指南](../tasks/language_modeling)

## GPTNeoXConfig

[[autodoc]] GPTNeoXConfig

## GPTNeoXTokenizerFast

[[autodoc]] GPTNeoXTokenizerFast

## GPTNeoXModel

[[autodoc]] GPTNeoXModel
    - forward

## GPTNeoXForCausalLM

[[autodoc]] GPTNeoXForCausalLM
    - forward

## GPTNeoXForQuestionAnswering

[[autodoc]] GPTNeoXForQuestionAnswering
    - forward

## GPTNeoXForSequenceClassification

[[autodoc]] GPTNeoXForSequenceClassification
    - forward

## GPTNeoXForTokenClassification

[[autodoc]] GPTNeoXForTokenClassification
    - forward