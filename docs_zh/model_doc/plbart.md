<!--
版权所有©2022 HuggingFace团队。保留所有权利。

根据 Apache 许可证，版本 2.0（“许可证”）下，您不得使用此文件，除非符合许可证的要求。
您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证的规定进行的软件分发将按"原样"分发，不附带任何明示或暗示的担保或条件。
详细了解许可证中的特定语言和限制，请参阅许可证。

⚠️ 请注意，此文件是使用 Markdown 编写的，但包含我们文档构建器的特定语法（类似于 MDX），可能无法正确在您的 Markdown 查看器中显示。

-->

# PLBart

**免责声明：**如果您发现任何奇怪的问题，请提交[GitHub问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)，并指派给
[@gchhablani](https://www.github.com/gchhablani)。

## PLBart概述

PLBART 模型由 Wasi Uddin Ahmad、Saikat Chakraborty、Baishakhi Ray和Kai-Wei Chang 在论文 [Unified Pre-training for Program Understanding and Generation](https://arxiv.org/abs/2103.06333) 中提出。
它是一个类似于 BART 的模型，可用于执行代码摘要、代码生成和代码翻译任务。预训练模型 `plbart-base` 在 Java、Python 和英语上使用了多语言去噪任务进行训练。

根据摘要内容：

*代码摘要和生成可以实现程序语言 (PL) 和自然语言 (NL) 之间的转化，而代码翻译则支持从一种 PL 迁移到另一种 PL 的遗留代码。本文介绍了 PLBART，一个能够执行广泛的程序和语言理解和生成任务的序列到序列模型。
PLBART 在大量的 Java 和 Python 函数以及相关 NL 文本之间通过去噪自编码预训练。在英语语言的代码摘要、代码生成和七种编程语言的代码翻译的实验中，我们发现 PLBART 能够超越或匹敌最先进的模型。
此外，对于包括程序修复、克隆检测和易受攻击代码检测在内的判别任务的实验表明，PLBART 在程序理解方面非常有效。
此外，分析表明，PLBART 学习了程序语法、风格（例如标识符命名规范）、逻辑流程（例如 `else` 块中的 `if` 块等价于 `else if` 块）对程序语义至关重要，
并且即使在有限的注释下也表现出色。*

此模型由[gchhablani](https://huggingface.co/gchhablani)贡献。作者的代码可以在[这里](https://github.com/wasiahmad/PLBART)找到。

### PLBart的训练

PLBART 是一个多语言编码器-解码器（序列到序列）模型，主要用于代码到文本、文本到代码、代码到代码的任务。由于该模型是多语言的，它期望以不同的格式提供序列。
特殊的语言 ID 令牌在源文本和目标文本中均被添加。源文本的格式是 `X [eos, src_lang_code]`，其中 `X` 是源文本。
目标文本的格式是 `[tgt_lang_code] X [eos]`。`bos` 从未被使用。

然而，在微调中，如果只使用单一语言，则在某些情况下不提供语言令牌。有关此内容的更多详细信息，请参考[论文](https://arxiv.org/abs/2103.06333)。

在需要语言代码的情况下，传递文本作为第一个参数或使用关键字参数 `text`，[`~PLBartTokenizer.__call__`] 将对源文本格式进行编码，
如果使用关键字参数 `text_target`，则对目标文本格式进行编码。

- 监督训练

```python
>>> from transformers import PLBartForConditionalGeneration, PLBartTokenizer

>>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base", src_lang="en_XX", tgt_lang="python")
>>> example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
>>> expected_translation_english = "Returns the maximum value of a b c."
>>> inputs = tokenizer(example_python_phrase, text_target=expected_translation_english, return_tensors="pt")
>>> model(**inputs)
```

- 生成

  在生成目标文本时，将 `decoder_start_token_id` 设置为目标语言 ID。以下示例演示了如何使用 `uclanlp/plbart-python-en_XX` 模型将 Python 翻译为英语。

```python
>>> from transformers import PLBartForConditionalGeneration, PLBartTokenizer

>>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX")
>>> example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
>>> inputs = tokenizer(example_python_phrase, return_tensors="pt")
>>> model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-python-en_XX")
>>> translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"])
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Returns the maximum value of a b c."
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [因果语言建模任务指南](../tasks/language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## PLBartConfig

[[autodoc]] PLBartConfig

## PLBartTokenizer

[[autodoc]] PLBartTokenizer
    - build_inputs_with_special_tokens

## PLBartModel

[[autodoc]] PLBartModel
    - forward

## PLBartForConditionalGeneration

[[autodoc]] PLBartForConditionalGeneration
    - forward

## PLBartForSequenceClassification

[[autodoc]] PLBartForSequenceClassification
    - forward

## PLBartForCausalLM

[[autodoc]] PLBartForCausalLM
    - forward