<!--版权所有2020年The HuggingFace团队。保留所有权利。

根据Apache许可，版本2.0（“许可证”）使用本文件，您不得违反此许可使用该文件。您可以在下面网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以了解许可证下的特定语言的权限和限制。

⚠️请注意，此文件是Markdown格式，但包含我们的文档构建器的特定语法（类似于MDX），可能无法在您的Markdown查看器中正确渲染。

-->

# NLLB

**声明：** tokenizer的默认行为最近已经修复（并因此更改）！

以前的版本在目标和源定但化的令牌序列末尾添加了`[self.eos_token_id, self.cur_lang_code]`。正如NLLB论文（第48页，6.1.1. 模型架构）所述：

*请注意，与之前多项工作（Arivazhagan等，2019年; Johnson等，2017年）相反，我们将源序列前置源语言而不是目标语言。这主要是因为我们优化模型在任意200种语言对上的零负样本性能具有优先权，对监督性能进行了一定的牺牲。*

先前的行为：

```python
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
>>> tokenizer("你今天过得怎么样？").input_ids
[13374, 1398, 4260, 4039, 248130, 2, 256047]

>>> # 2: '</s>'
>>> # 256047 : 'eng_Latn'
```

新的行为

```python
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
>>> tokenizer("你今天过得怎么样？").input_ids
[256047, 13374, 1398, 4260, 4039, 248130, 2]
```

可以通过以下方式启用旧的行为：
```python
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
```

详细信息，请随时查看链接的[PR](https://github.com/huggingface/transformers/pull/22313)和[Issue](https://github.com/huggingface/transformers/issues/19943)。

## NLLB概述

NLLB模型在Marta R. Costa-jussà、James Cross、Onur Çelebi、Maha Elbayad、Kenneth Heafield、Kevin Heffernan、Elahe Kalbassi、Janice Lam、Daniel Licht、Jean Maillard、Anna Sun、Skyler Wang、Guillaume Wenzek、Al Youngblood、Bapi Akula、Loic Barrault、Gabriel Mejia Gonzalez、Prangthip Hansanti、John Hoffman、Semarley Jarrett、Kaushik Ram Sadagopan、Dirk Rowe、Shannon Spruit、Chau Tran、Pierre Andrews、Necip Fazil Ayan、Shruti Bhosale、Sergey Edunov、AngelaFan、Cynthia Gao、Vedanuj Goswami、Francisco Guzmán、Philipp Koehn、Alexandre Mourachko、Christophe Ropers、Safiyyah Saleem、Holger Schwenk和Jeff Wang的论文《不让任何语言被遗忘：规模化的以人为本机器翻译》中提出。

该论文的摘要如下：

*受到在全球范围内消除语言障碍的目标驱动，机器翻译已经巩固其作为人工智能研究的重点。然而，这些努力已经围绕着很小的一部分语言聚集在一起，却抛弃了绝大部分主要是低资源语言的语言。想要打破200种语言的壁垒并确保安全、高质量的结果，同时保持伦理考虑在内，需要付出什么样的努力？在《不让任何语言被遗忘》中，我们通过首先通过与母语使用者进行初步访谈来为低资源语言翻译支持的需求提供背景。然后，我们创建了数据集和模型，旨在缩小低资源语言与高资源语言之间的性能差距。更具体地说，我们根据以稀疏的专家混合为基础的条件计算模型，在所有获取的针对低资源语言的数据上进行了训练，这些数据是通过专为低资源语言量身定制的新颖和有效的数据挖掘技术获得的。我们提出了多种架构和训练改进方法来抵消在进行数千个任务的训练时的过拟合问题。至关重要的是，我们使用人工翻译的基准测试——Flores-200对超过40,000种不同的翻译方向进行了性能评估，并结合了涵盖Flores-200所有语言的新颖毒性基准测试来评估翻译安全性。我们的模型在BLEU相对于先前的最新技术水平上实现了44%的提升，为实现通用翻译系统奠定了重要的基础。*

该实现包含发布的稠密模型。

**稀疏模型NLLB-MoE（专家混合）现已提供！更多详细信息[nllb-moe](nllb-moe)**

此模型由[Lysandre](https://huggingface.co/lysandre)贡献。作者的代码可以在[此处](https://github.com/facebookresearch/fairseq/tree/nllb)找到。

## 使用NLLB生成

在生成目标文本时，将`forced_bos_token_id`设置为目标语言ID。下面的示例演示了如何使用*facebook/nllb-200-distilled-600M*模型将英语翻译成法语。

请注意，我们使用的是法语的BCP-47代码`fra_Latn`。请参阅[此处](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)以获取Flores 200数据集中所有BCP-47代码的列表。

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

>>> article = "联合国秘书长表示叙利亚没有军事解决办法"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
Le chef de l'ONU dit qu'il n'y a pas de solution militaire en Syrie
```

### 从英语以外的任何其他语言生成

英语（`eng_Latn`）被设定为默认的翻译语言。要指定从其他语言翻译，请在tokenizer初始化的`src_lang`关键字参数中指定BCP-47代码。

以下示例演示了从罗马尼亚语翻译成德语的情况：

```py
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained(
...     "facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="ron_Latn"
... )
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True)

>>> article = "Şeful ONU spune că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
UN-Chef sagt, es gibt keine militärische Lösung in Syrien
```

## 文档资源

- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## NllbTokenizer

[[autodoc]] NllbTokenizer
    - build_inputs_with_special_tokens

## NllbTokenizerFast

[[autodoc]] NllbTokenizerFast