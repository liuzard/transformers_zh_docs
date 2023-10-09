<!--
版权2020年HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证，否则你不得使用此文件。
你可以在以下地址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的，不附带任何明示或暗示的担保或条件。
请详见许可证中有关特定语言的规定和限制。

⚠️请注意，此文件使用Markdown格式，但包含我们文档构建器的特定语法（类似于MDX），在你的Markdown查看器中可能无法正确显示。

-->

# HerBERT

## 概述

HerBERT模型是由Piotr Rybak，Robert Mroczkowski，Janusz Tracz，和Ireneusz Gawlik提出的[波兰语理解综合基准测试KLEJ](https://www.aclweb.org/anthology/2020.acl-main.111.pdf)中的一个基于BERT的语言模型。该模型是在波兰语语料库上使用只有MLM目标且采用动态掩码整个单词的方法训练的。

论文中的摘要如下：

*近年来，一系列基于Transformer的模型为通用自然语言理解（NLU）任务带来了重大改进。这样快速的研究进展离不开通用NLU基准测试，这些基准测试允许对提出的方法进行公正比较。然而，这样的基准测试仅适用于少数语言。为了解决这个问题，我们引入了一个面向波兰语理解的综合多任务基准测试，并附带一个在线排行榜。它由一系列多样化的任务组成，这些任务是从现有的用于命名实体识别、问答、文本蕴含等任务的数据集中采用的。我们还为电子商务领域引入了一个名为Allegro Reviews（AR）的新情感分析任务。为了确保提供一种公共的评估方案并推广能够推广到不同NLU任务的模型，该基准测试包含了来自不同领域和应用的数据集。此外，我们还发布了HerBERT，这是一个专门针对波兰语训练的基于Transformer的模型，它具有最佳的平均性能，并在九项任务中的其中三项中取得了最佳结果。最后，我们提供了一项广泛的评估，包括几种标准基线和最近提出的多语种基于Transformer的模型。*

使用示例：

```python
>>> from transformers import HerbertTokenizer, RobertaModel

>>> tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
>>> model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")

>>> encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors="pt")
>>> outputs = model(encoded_input)

>>> # HerBERT也可以使用AutoTokenizer和AutoModel进行加载：
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
>>> model = AutoModel.from_pretrained("allegro/herbert-klej-cased-v1")
```

本模型由[rmroczkowski](https://huggingface.co/rmroczkowski)提供。原始代码可在[此处](https://github.com/allegro/HerBERT)找到。

## HerbertTokenizer

[[autodoc]] HerbertTokenizer

## HerbertTokenizerFast

[[autodoc]] HerbertTokenizerFast
