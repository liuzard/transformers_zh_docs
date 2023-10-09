# NLLB-MOE


## 概述

NLLB模型在[Marta R. Costa-jussà, James Cross, Onur Çelebi等人的论文《No Language Left Behind: Scaling Human-Centered Machine Translation》](https://arxiv.org/abs/2207.04672)中进行了介绍。该论文的摘要如下：

*在消除全球范围内语言障碍的目标驱动下，机器翻译已成为现今人工智能研究的主要关注领域。然而，这些努力大多集中在一小部分语言上，而忽略了大多数低资源语言。为了突破200种语言的障碍，并确保安全、高质量的结果，同时考虑伦理因素，我们在《No Language Left Behind》中通过与母语者进行探索性访谈来定位低资源语言翻译支持的需求。然后，我们创建了面向缩小低资源语言和高资源语言性能差距的数据集和模型。具体而言，我们开发了一种基于稀疏门控混合专家的条件计算模型，该模型经过为低资源语言量身定制的创新和高效数据挖掘技术获取的数据进行训练。我们提出了多种架构和训练改进措施来抵制过拟合，同时在成千上万个任务上进行训练。关键是，我们使用人工翻译的基准数据集Flores-200评估了超过40,000个不同的翻译方向的性能，并结合了一个覆盖Flores-200中所有语言的新型毒性基准数据集，评估翻译的安全性。我们的模型相对于先前的最先进模型BLEU提升了44%，为实现通用翻译系统奠定了重要基础。*

提示：

- M2M100ForConditionalGeneration模型是NLLB和NLLB-MoE的基础模型
- NLLB-MoE与NLLB模型非常相似，但其前馈层基于SwitchTransformers的实现。
- 分词器与NLLB模型相同。

该模型由[Arthur Zucker](https://huggingface.co/ArtZucker)贡献。
原始代码可在[这里](https://github.com/facebookresearch/fairseq)找到。

## 与SwitchTransformers的实现差异

最大的差异在于标记的路由方式。NLLB-MoE使用了“top-2-gate”，即每个输入只选择两个最高预测概率的专家，并忽略其他专家。而在`SwitchTransformers`中，只计算了最高概率的top-1，这意味着标记被转发的概率较低。此外，如果一个标记没有被路由到任何专家，`SwitchTransformers`仍然会添加其未经修改的隐藏状态（类似残差连接），而在`NLLB`的top-2路由机制中对它们进行了屏蔽。

## 使用NLLB-MoE生成文本

可用的检查点需要约350GB的存储空间。如果您的计算机内存不足，请确保使用`accelerate`。

在生成目标文本时，将`forced_bos_token_id`设置为目标语言的id。以下示例展示了如何使用*facebook/nllb-200-distilled-600M*模型将英语翻译为法语。

注意，我们在法语（`fra_Latn`）中使用了法语的BCP-47代码。有关Flores 200数据集中所有BCP-47的列表，请参见[这里](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)。

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b")

>>> article = "Previously, Ring's CEO, Jamie Siminoff, remarked the company started when his doorbell wasn't audible from his shop in his garage."
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=50
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Auparavant, le PDG de Ring, Jamie Siminoff, a fait remarquer que la société avait commencé lorsque sa sonnette n'était pas audible depuis son magasin dans son garage."
```

### 从除英语以外的任何语言生成

英语（`eng_Latn`）设置为默认的翻译语言。为了指定从其他语言翻译，您应该在分词器初始化的`src_lang`关键字参数中指定BCP-47代码。

以下示例演示了从罗马尼亚语翻译为德语的情况：

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b", src_lang="ron_Latn")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b")

>>> article = "Şeful ONU spune că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
```

## 文档资源

- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)


## NllbMoeConfig

[[autodoc]] NllbMoeConfig

## NllbMoeTop2Router

[[autodoc]] NllbMoeTop2Router
    - route_tokens
    - forward

## NllbMoeSparseMLP

[[autodoc]] NllbMoeSparseMLP
    - forward

## NllbMoeModel

[[autodoc]] NllbMoeModel
    - forward

## NllbMoeForConditionalGeneration

[[autodoc]] NllbMoeForConditionalGeneration
    - forward