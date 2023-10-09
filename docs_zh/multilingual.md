<!--版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证的规定，否则不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0
   
获得许可证的副本
除非适用法律要求或书面同意，否则根据许可证分发的软件是按"原样"分发
基础，无论明示或暗示，包括但不限于对适销性、特定目的的适用性和非侵权性的保证。有关许可证的详细信息请参见

许可下的特定语言和限制。

⚠️请注意，这个文件是Markdown格式的，但含有我们的doc-builder的特定语法（类似于MDX），在您的Markdown查看器中可能无法正确渲染。

-->

# 用于推理的多语言模型

[[open-in-colab]]

在🤗 Transformers中有几个多语言模型，它们与单语言模型的推理使用方式不同。当然，并不是*所有*的多语言模型的用法都不同。有些模型，例如[bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased)，可以像单语言模型一样使用。本指南将展示如何使用在推理中用法不同的多语言模型。

## XLM

XLM有十个不同的检查点，其中只有一个是单语言的。其余九个模型检查点可以分为两类：使用语言嵌入和不使用语言嵌入的检查点。

### 带有语言嵌入的XLM

以下XLM模型在推理中使用语言嵌入来指定所使用的语言：

- `xlm-mlm-ende-1024` （语言掩蔽模型，英文-德文）
- `xlm-mlm-enfr-1024` （语言掩蔽模型，英文-法文）
- `xlm-mlm-enro-1024` （语言掩蔽模型，英文-罗马尼亚文）
- `xlm-mlm-xnli15-1024` （语言掩码模型，XNLI语言）
- `xlm-mlm-tlm-xnli15-1024` （语言掩蔽模型+翻译，XNLI语言）
- `xlm-clm-enfr-1024` （因果语言建模，英文-法文）
- `xlm-clm-ende-1024` （因果语言建模，英文-德文）

语言嵌入被表示为一个与传递给模型的`input_ids`形状相同的张量。这些张量中的值取决于所使用的语言，并由分词器的`lang2id`和`id2lang`属性进行识别。

在此示例中，加载`xlm-clm-enfr-1024`检查点（因果语言建模，英法双语）：

```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")
```

分词器的`lang2id`属性显示了此模型的语言及其ID：

```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

接下来，创建一个示例输入：

```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # 批量大小为1
```

将语言ID设置为`"en"`，并使用其定义语言嵌入。语言嵌入是一个张量，填充为`0`，因为这是英语的语言ID。这个张量的大小应与`input_ids`相同。

```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # 将其重塑为大小为（batch_size，sequence_length）
>>> langs = langs.view(1, -1)  # 现在的形状是[1，sequence_length]（批量大小为1）
```

现在，可以将`input_ids`和语言嵌入传递给模型：

```py
>>> outputs = model(input_ids, langs=langs)
```

[run_generation.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation/run_generation.py)脚本可以使用带有语言嵌入的`xlm-clm`检查点生成文本。

### 不带语言嵌入的XLM

以下XLM模型在推理中不需要语言嵌入：

- `xlm-mlm-17-1280` （语言掩蔽模型，17种语言）
- `xlm-mlm-100-1280` （语言掩蔽模型，100种语言）

这些模型用于通用的句子表示，与前面的XLM检查点不同。

## BERT

以下BERT模型可用于多语言任务：

- `bert-base-multilingual-uncased` （语言掩蔽模型+下一句预测，102种语言）
- `bert-base-multilingual-cased` （语言掩蔽模型+下一句预测，104种语言）

这些模型在推理中不需要语言嵌入。它们应根据上下文识别语言并进行推理。

## XLM-RoBERTa

以下XLM-RoBERTa模型可用于多语言任务：

- `xlm-roberta-base` （语言掩蔽模型，100种语言）
- `xlm-roberta-large` （语言掩蔽模型，100种语言）

XLM-RoBERTa在100种语言的新创建和清理的CommonCrawl数据上进行了2.5TB的训练。相比先前发布的多语言模型（如mBERT或XLM），它在分类、序列标注和问答等下游任务上提供了很大的改进。

## M2M100

以下M2M100模型可用于多语言翻译：

- `facebook/m2m100_418M` （翻译）
- `facebook/m2m100_1.2B` （翻译）

在此示例中，加载`facebook/m2m100_418M`检查点，将中文翻译为英文。您可以在分词器中设置源语言：

```py
>>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> chinese_text = "不要插手巫师的事务, 因为他们是微妙的, 很快就会发怒."

>>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
>>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
```

对文本进行分词：

```py
>>> encoded_zh = tokenizer(chinese_text, return_tensors="pt")
```

M2M100要求将目标语言ID作为第一个生成的标记，以将其翻译为目标语言。在`generate`方法中，将`forced_bos_token_id`设置为`en`以将其翻译为英语：

```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart

以下MBart模型可用于多语言翻译：

- `facebook/mbart-large-50-one-to-many-mmt` （一对多多语言机器翻译，50种语言）
- `facebook/mbart-large-50-many-to-many-mmt` （多对多多语言机器翻译，50种语言）
- `facebook/mbart-large-50-many-to-one-mmt` （多对一多语言机器翻译，50种语言）
- `facebook/mbart-large-50` （多语言翻译，50种语言）
- `facebook/mbart-large-cc25`

在此示例中，加载`facebook/mbart-large-50-many-to-many-mmt`检查点，将芬兰语翻译为英语。您可以在分词器中设置源语言：

```py
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

对文本进行分词：

```py
>>> encoded_en = tokenizer(en_text, return_tensors="pt")
```

MBart要求将目标语言ID作为第一个生成的标记，以将其翻译为目标语言。在`generate`方法中，将`forced_bos_token_id`设置为`en_XX`以将其翻译为英语：

```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

使用`facebook/mbart-large-50-many-to-one-mmt`检查点时，不需要强制设置目标语言ID作为第一个生成的标记，否则用法相同。