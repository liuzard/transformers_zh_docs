# LongT5

## 概述

LongT5模型是由Mandy Guo、Joshua Ainslie、David Uthus、Santiago Ontanon、Jianmo Ni、Yun-Hsuan Sung和Yinfei Yang在《LongT5: Efficient Text-To-Text Transformer for Long Sequences》(https://arxiv.org/abs/2112.07916)中提出的。它是一种在文本到文本去噪生成任务中进行预训练的编码器-解码器Transformer模型。LongT5模型是T5模型的扩展，它使得可以使用两种不同的高效注意机制：(1) 局部注意力或者(2) 短暂全局注意力。

论文中的摘要如下：

*最近的研究表明，要么(1) 增加输入长度，要么(2) 增加模型大小，都可以提高基于Transformer的神经模型的性能。在本文中，我们提出了一种新模型，称为LongT5，在该模型中我们探索了同时扩展输入长度和模型大小的效果。具体来说，我们将长输入的注意力思想（ETC）与摘要预训练（PEGASUS）的预训练策略结合到可扩展的T5架构中。结果是我们提出了一种新的注意力机制，我们称之为“短暂全局”(TGlobal)，它模拟了ETC的局部/全局注意力机制，但不需要额外的侧输入。我们能够在几个摘要任务上取得最先进的结果，并且在问答任务上胜过了原始的T5模型。*

提示：

- [`LongT5ForConditionalGeneration`] 是 [`T5ForConditionalGeneration`] 的扩展，使用了传统编码器的*自注意力*层，并用高效的*局部*注意力或*短暂全局*(*tglobal*)注意力来替换。
- 与T5模型不同，LongT5不使用任务前缀。此外，它使用了一个不同的预训练目标，借鉴了[`PegasusForConditionalGeneration`]的预训练方式。
- LongT5模型是为长序列 *序列到序列* 任务设计的，其中输入序列超过常用的512个标记。它能够处理长度为16384个标记的输入序列。
- 对于*局部注意力*，稀疏的滑动窗口局部注意力操作使得给定的标记只能注意到其左右 `r` 个标记（默认情况下 `r=127`）。*局部注意力*不会引入模型的新参数。该机制的复杂性与输入序列长度 `l` 成线性关系：`O(l*r)`。
- *短暂全局注意力*是*局部注意力*的扩展。它还允许每个输入标记与该层中的所有其他标记进行交互。这是通过将输入序列分成固定长度 `k` 的块来实现的（默认情况下 `k=16`）。然后，通过对块中的每个标记的嵌入进行求和和归一化，获得该块的一个全局标记。由于这一点，注意力允许每个标记既可以关注附近的标记（就像局部注意力一样），也可以关注每个全局标记（就像标准的全局注意力一样）（*短暂*表示全局标记在每个注意力操作中是动态构建的）。因此，*TGlobal*注意力引入了一些新的参数 -- 全局相对位置偏置和全局标记嵌入的层归一化。该机制的复杂性是 `O(l(r + l/k))`。
- 下面是一个示例，展示了如何在 [pubmed数据集](https://huggingface.co/datasets/scientific_papers) 上评估经过微调的LongT5模型。

```python
>>> import evaluate
>>> from datasets import load_dataset
>>> from transformers import AutoTokenizer, LongT5ForConditionalGeneration

>>> dataset = load_dataset("scientific_papers", "pubmed", split="validation")
>>> model = (
...     LongT5ForConditionalGeneration.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
...     .to("cuda")
...     .half()
... )
>>> tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")


>>> def generate_answers(batch):
...     inputs_dict = tokenizer(
...         batch["article"], max_length=16384, padding="max_length", truncation=True, return_tensors="pt"
...     )
...     input_ids = inputs_dict.input_ids.to("cuda")
...     attention_mask = inputs_dict.attention_mask.to("cuda")
...     output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=2)
...     batch["predicted_abstract"] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
...     return batch


>>> result = dataset.map(generate_answer, batched=True, batch_size=2)
>>> rouge = evaluate.load("rouge")
>>> rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"])
```

此模型由[stancld](https://huggingface.co/stancld)贡献。
原始代码可以在[此处](https://github.com/google-research/longt5)找到。

## 文档资源

- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## LongT5Config

[[autodoc]] LongT5Config

## LongT5Model

[[autodoc]] LongT5Model
    - forward

## LongT5ForConditionalGeneration

[[autodoc]] LongT5ForConditionalGeneration
    - forward

## LongT5EncoderModel

[[autodoc]] LongT5EncoderModel
    - forward

## FlaxLongT5Model

[[autodoc]] FlaxLongT5Model
    - __call__
    - encode
    - decode

## FlaxLongT5ForConditionalGeneration

[[autodoc]] FlaxLongT5ForConditionalGeneration
    - __call__
    - encode
    - decode