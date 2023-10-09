# GPT-NeoX-Japanese

## 概述

我们介绍了 GPT-NeoX 日语版，这是一个基于 [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox) 训练的用于日语的自回归语言模型。
日语是一种独特的语言，具有丰富的词汇和平假名、片假名和汉字等不同的写作方式。
为了解决日语这种独特的语言结构，我们使用了 [特殊的子词分词器](https://github.com/tanreinama/Japanese-BPEEncoder_V2)。我们非常感谢 *tanreinama* 对这个非常有帮助的分词器进行了开源。
根据 Google 在 [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) 上的研究建议，我们已经从 transformer blocks 中去除了偏置参数，以获得更好的模型性能。请详细参阅[本文](https://medium.com/ml-abeja/training-a-better-gpt-2-93b157662ae4)。

该模型的开发由 [Shinya Otani](https://github.com/SO0529)、[Takayoshi Makabe](https://github.com/spider-man-tm)、[Anuj Arora](https://github.com/Anuj040) 和 [Kyo Hattori](https://github.com/go5paopao) 来自 [ABEJA, Inc.](https://www.abejainc.com/) 领导。有关此模型构建活动的更多信息，请参阅[此处 (ja)](https://tech-blog.abeja.asia/entry/abeja-gpt-project-202207)。

### 生成

使用 GPT NeoX 日语模型可以使用 `generate()` 方法生成文本。

```python
>>> from transformers import GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseTokenizer

>>> model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")

>>> prompt = "人とAIが協調するためには、"

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

>>> print(gen_text)
人とAIが協調するためには、AIと人が共存し、AIを正しく理解する必要があります。
```

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)

## GPTNeoXJapaneseConfig

[[autodoc]] GPTNeoXJapaneseConfig

## GPTNeoXJapaneseTokenizer

[[autodoc]] GPTNeoXJapaneseTokenizer

## GPTNeoXJapaneseModel

[[autodoc]] GPTNeoXJapaneseModel
    - forward

## GPTNeoXJapaneseForCausalLM

[[autodoc]] GPTNeoXJapaneseForCausalLM
    - forward