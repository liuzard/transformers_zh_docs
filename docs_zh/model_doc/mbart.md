<!--版权2020 HuggingFace团队。版权所有。

根据Apache许可证第2.0版（"许可证"）许可；除非符合许可证的规定，否则你不得使用此文件。
你可以在以下网址获取许可证副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证授权的软件是基于“按原样”（as is）的基础分发的，无论是明示的还是暗示的。请参阅许可证以获取特定语言的权限和限制。
⚠️注意，这个文件是Markdown格式，但包含了我们的doc-builder的特殊语法（类似于MDX），这个语法在你的Markdown查看器中可能无法正确显示。

-->

# MBart和MBart-50

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=mbart">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-mbart-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/mbart-large-50-one-to-many-mmt">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**免责声明：**如果你发现任何奇怪的问题，请提交[Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)并指派给@patrickvonplaten

## MBart概述

MBart模型是由Yinhan Liu、Jiatao Gu、Naman Goyal、Xian Li、Sergey Edunov Marjan Ghazvininejad、Mike Lewis和Luke Zettlemoyer在[多语言去噪预训练神经机器翻译](https://arxiv.org/abs/2001.08210)一文中提出的。

根据摘要，MBart是一个在大规模单语语料库中针对序列到序列的去噪自编码器进行预训练的模型，使用的是BART目标。MBart是第一种通过去噪多语言全文预训练完成完整序列到序列模型的方法，而之前的方法只关注编码器、解码器或文本部分的重建。

这个模型是由[valhalla](https://huggingface.co/valhalla)贡献的。作者的代码可以在[这里](https://github.com/pytorch/fairseq/tree/master/examples/mbart)找到。

### MBart的训练

MBart是一个多语言编码器-解码器（序列到序列）模型，主要用于翻译任务。由于该模型是多语言的，它期望序列以不同的格式提供。特殊的语言ID标记被添加到源文本和目标文本中。源文本的格式是`X [eos, src_lang_code]`，其中`X`是源文本，目标文本的格式是`[tgt_lang_code] X [eos]`。`bos`从不被使用。

常规的[`~MBartTokenizer.__call__`]会对作为第一个参数传递或使用`text`关键字传递的源文本格式进行编码，并使用`text_label`关键字参数传递目标文本格式。

- 监督训练

```python
>>> from transformers import MBartForConditionalGeneration, MBartTokenizer

>>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
>>> example_english_phrase = "UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"

>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")

>>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
>>> # 正向传递
>>> model(**inputs)
```

- 生成

  在生成目标文本时，将`decoder_start_token_id`设置为目标语言ID。下面的例子展示了如何使用*facebook/mbart-large-en-ro*模型将英文翻译成罗马尼亚语。

```python
>>> from transformers import MBartForConditionalGeneration, MBartTokenizer

>>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX")
>>> article = "UN Chief Says There Is No Military Solution in Syria"
>>> inputs = tokenizer(article, return_tensors="pt")
>>> translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Şeful ONU declară că nu există o soluţie militară în Siria"
```

## MBart-50概述

MBart-50是由Yuqing Tang、Chau Tran、Xian Li、Peng-Jen Chen、Naman Goyal、Vishrav Chaudhary、Jiatao Gu和Angela Fan在[具有可扩展多语言预训练和微调的多语言翻译](https://arxiv.org/abs/2008.00401)一文中提出的。MBart-50是使用原始的*mbart-large-cc25*检查点创建的，通过扩展其嵌入层以随机初始化的向量形式增加了额外的25个语言标记，然后在50种语言上进行了预训练。

根据摘要

*多语言翻译模型可以通过多语言微调创建。与单向微调不同，预训练模型同时在多个方向上进行微调。它证明了预训练模型可以扩展到包含额外语言而不丧失性能。多语言微调相比最强基线（无论是从头开始的多语言模型还是双语微调）平均提高了1个BLEU，并且相比从头开始的双语基线平均提高了9.3个BLEU。*

### MBart-50的训练

MBart-50的文本格式与MBart略有不同。对于MBart-50，语言ID标记用作源文本和目标文本的前缀，即文本格式为`[lang_code] X [eos]`，其中`lang_code`是源文本的源语言ID，是目标文本的目标语言ID，而`X`分别是源文本或目标文本。

MBart-50有自己的tokenizer [`MBart50Tokenizer`]。

- 监督训练

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")

src_text = " UN Chief Says There Is No Military Solution in Syria"
tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"

model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")

model(**model_inputs)  # forward pass
```

- 生成

  若要使用mBART-50多语言翻译模型进行生成，需要将`eos_token_id`作为`decoder_start_token_id`，并将目标语言ID强制设为第一个生成的token。为了将目标语言ID强制设为第一个生成的token，可以将*forced_bos_token_id*参数传递给*generate*方法。下面的例子展示了如何使用*facebook/mbart-50-large-many-to-many*检查点将印地语翻译成法语和阿拉伯语翻译成英语。

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# translate Hindi to French
tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire en Syria."

# translate Arabic to English
tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "The Secretary-General of the United Nations says there is no military solution in Syria."
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言模型任务指南](../tasks/language_modeling)
- [掩码语言模型任务指南](../tasks/masked_language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## MBartConfig

[[autodoc]] MBartConfig

## MBartTokenizer

[[autodoc]] MBartTokenizer
    - build_inputs_with_special_tokens

## MBartTokenizerFast

[[autodoc]] MBartTokenizerFast

## MBart50Tokenizer

[[autodoc]] MBart50Tokenizer

## MBart50TokenizerFast

[[autodoc]] MBart50TokenizerFast

## MBartModel

[[autodoc]] MBartModel

## MBartForConditionalGeneration

[[autodoc]] MBartForConditionalGeneration

## MBartForQuestionAnswering

[[autodoc]] MBartForQuestionAnswering

## MBartForSequenceClassification

[[autodoc]] MBartForSequenceClassification

## MBartForCausalLM

[[autodoc]] MBartForCausalLM
    - forward

## TFMBartModel

[[autodoc]] TFMBartModel
    - call

## TFMBartForConditionalGeneration

[[autodoc]] TFMBartForConditionalGeneration
    - call

## FlaxMBartModel

[[autodoc]] FlaxMBartModel
    - __call__
    - encode
    - decode

## FlaxMBartForConditionalGeneration

[[autodoc]] FlaxMBartForConditionalGeneration
    - __call__
    - encode
    - decode

## FlaxMBartForSequenceClassification

[[autodoc]] FlaxMBartForSequenceClassification
    - __call__
    - encode
    - decode

## FlaxMBartForQuestionAnswering

[[autodoc]] FlaxMBartForQuestionAnswering
    - __call__
    - encode
    - decode