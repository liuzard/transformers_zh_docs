<!--版权2022年HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”），您除非符合许可证的规定，在未获得版权人的许可的情况下不得使用此文件。
您可以在以下位置获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发且不附带任何明示或暗示的保证或条件。请参阅许可证以获取有关许可证下特定语言的权限和限制的详细信息。

⚠️请注意，此文件以Markdown格式编写，但包含了我们文档构建器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确呈现。

-->

# MVP

## 概述

[MVP:多任务有监督预训练用于自然语言生成](https://arxiv.org/abs/2206.12131)是由Tianyi Tang，Junyi Li，Wayne Xin Zhao和Ji-Rong Wen在提出的。

根据摘要，

- MVP遵循标准的Transformer编码器-解码器架构。
- MVP通过有标签的数据集进行有监督的预训练。
- MVP还具有任务特定的软提示，以激发模型在执行特定任务方面的能力。
- MVP是专为自然语言生成而设计的，并可适应多种生成任务，包括但不限于摘要、数据到文本生成、开放式对话系统、故事生成、问答、问题生成、面向任务的对话系统、常识生成、释义生成、文本样式转换和文本简化。我们的模型还可以适应自然语言理解任务，如序列分类和（抽取式）问题回答。

提示：
- 我们在[这里](https://huggingface.co/models?filter=mvp)发布了一系列模型，包括MVP、带有任务特定提示的MVP和多任务预训练变体。
- 如果您想使用没有提示的模型（标准Transformer），可以通过`MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp')`进行加载。
- 如果您想使用带有任务特定提示的模型，例如摘要，可以通过`MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp-summarization')`进行加载。
- 我们的模型支持轻量级的前置调整（Prefix-tuning），具体方法请参见[这里](https://arxiv.org/abs/2101.00190)中的`set_lightweight_tuning()`方法。

此模型由[Tianyi Tang](https://huggingface.co/StevenTang)贡献。详细信息和说明可在[此处](https://github.com/RUCAIBox/MVP)找到。

## 示例
对于摘要，以下是使用MVP和带有摘要特定提示的MVP的示例。

```python
>>> from transformers import MvpTokenizer, MvpForConditionalGeneration

>>> tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")
>>> model_with_prompt = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp-summarization")

>>> inputs = tokenizer(
...     "Summarize: You may want to stick it to your boss and leave your job, but don't do it if these are your reasons.",
...     return_tensors="pt",
... )
>>> generated_ids = model.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
["为什么不要辞掉你的工作"]

>>> generated_ids = model_with_prompt.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
["如果这些是你的理由，就不要这样做"]
```

对于数据到文本生成，以下是使用MVP和多任务预训练变体的示例。
```python
>>> from transformers import MvpTokenizerFast, MvpForConditionalGeneration

>>> tokenizer = MvpTokenizerFast.from_pretrained("RUCAIBox/mvp")
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")
>>> model_with_mtl = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")

>>> inputs = tokenizer(
...     "Describe the following data: Iron Man | instance of | Superhero [SEP] Stan Lee | creator | Iron Man",
...     return_tensors="pt",
... )
>>> generated_ids = model.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['斯坦·李创作了虚构的超级英雄钢铁侠']

>>> generated_ids = model_with_mtl.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['钢铁侠是由漫威演员出版的美国漫画中出现的一个虚构的超级英雄。']
```

对于轻量级调整（即仅固定模型并调整提示），您可以加载具有随机初始化提示或具有任务特定提示的MVP。我们的代码还支持使用BART进行前缀调整，遵循[原始论文](https://arxiv.org/abs/2101.00190)的方法。

```python
>>> from transformers import MvpForConditionalGeneration

>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp", use_prompt=True)
>>> # 可训练参数的数量（全调整）
>>> sum(p.numel() for p in model.parameters() if p.requires_grad)
468116832

>>> # 带有随机初始化提示的轻量级调整
>>> model.set_lightweight_tuning()
>>> # 可训练参数的数量（轻量级调整）
>>> sum(p.numel() for p in model.parameters() if p.requires_grad)
61823328

>>> # 带有任务特定提示的轻量级调整
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")
>>> model.set_lightweight_tuning()
>>> # 原始的轻量级前缀调整
>>> model = MvpForConditionalGeneration.from_pretrained("facebook/bart-large", use_prompt=True)
>>> model.set_lightweight_tuning()
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## MvpConfig

[[autodoc]] MvpConfig

## MvpTokenizer

[[autodoc]] MvpTokenizer

## MvpTokenizerFast

[[autodoc]] MvpTokenizerFast

## MvpModel

[[autodoc]] MvpModel
    - forward

## MvpForConditionalGeneration

[[autodoc]] MvpForConditionalGeneration
    - forward

## MvpForSequenceClassification

[[autodoc]] MvpForSequenceClassification
    - forward

## MvpForQuestionAnswering

[[autodoc]] MvpForQuestionAnswering
    - forward

## MvpForCausalLM

[[autodoc]] MvpForCausalLM
    - forward