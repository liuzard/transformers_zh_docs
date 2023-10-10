<!--版权所有2021 HuggingFace团队。

根据Apache许可证，版本2.0（“许可证”）获得许可；除非遵守许可证，否则禁止使用此文件。你可以在


http://www.apache.org/licenses/LICENSE-2.0


获取许可证副本。

除非适用法律要求或书面同意，根据许可证分发的软件是基于“AS IS” BASIS分发的，无论是明示还是暗示的，都不包括保证或条件。请参阅许可证以获取许可证下的特定语言的权限和限制的详细信息。

⚠️请注意，此文件采用Markdown格式，但包含我们的doc-builder的特定语法（类似于MDX），可能在你的Markdown查看器中无法正确渲染。-->

# CANINE

## 概述

CANINE模型最初在Jonathan H. Clark，Dan Garrette，Iulia Turc，John Wieting的研究论文[CANINE：为语言表示进行去分词预训练高效编码器](https://arxiv.org/abs/2103.06874)中提出。这是第一批在训练Transformer时不使用显式分词步骤（例如字节对编码（BPE），WordPiece或SentencePiece）的论文之一。相反，该模型直接在Unicode字符级别上进行训练。在字符级别进行训练必然会带来更长的序列长度，CANINE通过高效的降采样策略解决了这个问题，然后应用了深层Transformer编码器。

论文摘要如下：

*流水线NLP系统已经被端到端神经建模所取代，然而几乎所有常用的模型仍然需要一个显式的分词步骤。最近的基于数据派生子词词典的分词方法比手动设计的分词器更加灵活，但这些技术并不适用于所有语言，并且任何固定词汇表的使用都可能限制模型的适应能力。在本文中，我们提出了CANINE，一种直接在字符序列上操作的神经编码器，不需要显式的分词或词汇表，并且操作也可以直接在子词上使用作为软的归纳偏置。为了有效和高效地使用其更细粒度的输入，CANINE将降采样（减少输入序列长度）与深层变换器堆叠（编码上下文）相结合。尽管模型参数少28％，但CANINE在挑战性的多语言基准测试TyDi QA上的F1分数比可比的mBERT模型高2.8％。*

提示：

- CANINE在内部使用不少于3个Transformer编码器：2个"浅层"编码器（仅包含单个层）和1个"深层"编码器（一个常规的BERT编码器）。首先，使用一个"浅层"编码器对字符嵌入进行上下文化处理，使用本地注意力。接下来，在降采样之后，应用一个"深层"编码器。最后，在上采样之后，使用一个"浅层"编码器来创建最终的字符嵌入。有关上采样和下采样的详细信息，请参阅论文。
- CANINE默认使用2048个字符的最大序列长度。可以使用[`CanineTokenizer`]来为模型准备文本。
- 分类可以通过在特殊的[CLS]token的最终隐藏状态上放置一个线性层来完成（具有预定义的Unicode代码点）。然而，对于标记分类任务，被下采样的token序列需要上采样，以匹配原始字符序列的长度（为2048）。有关详细信息，请参阅论文。
- 模型：

  - [google/canine-c](https://huggingface.co/google/canine-c)：使用自回归字符损失进行预训练，12层，768隐藏，12头，121M参数（大小约为500 MB）。
  - [google/canine-s](https://huggingface.co/google/canine-s)：使用子词损失进行预训练，12层，768隐藏，12头，121M参数（大小约为500 MB）。

此模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可以在[这里](https://github.com/google-research/language/tree/master/language/canine)找到。


### 示例

CANINE适用于原始字符，因此可以在没有标记器的情况下使用：

```python
>>> from transformers import CanineModel
>>> import torch

>>> model = CanineModel.from_pretrained("google/canine-c")  # 使用自回归字符损失预训练的模型

>>> text = "hello world"
>>> # 使用Python的内置ord()函数将每个字符转换为其Unicode代码点ID
>>> input_ids = torch.tensor([[ord(char) for char in text]])

>>> outputs = model(input_ids)  # 正向传递
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

对于批量推理和训练，建议使用标记器（以便将所有序列填充/截断为相同长度）：

```python
>>> from transformers import CanineTokenizer, CanineModel

>>> model = CanineModel.from_pretrained("google/canine-c")
>>> tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

>>> inputs = ["人生就像一盒巧克力。", "你永远不知道你会得到什么。"]
>>> encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

>>> outputs = model(**encoding)  # 正向传递
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [多项选择任务指南](../tasks/multiple_choice)

## CANINE特定的输出

[[autodoc]] models.canine.modeling_canine.CanineModelOutputWithPooling

## CanineConfig

[[autodoc]] CanineConfig

## CanineTokenizer

[[autodoc]] CanineTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences

## CanineModel

[[autodoc]] CanineModel
    - forward

## CanineForSequenceClassification

[[autodoc]] CanineForSequenceClassification
    - forward

## CanineForMultipleChoice

[[autodoc]] CanineForMultipleChoice
    - forward

## CanineForTokenClassification

[[autodoc]] CanineForTokenClassification
    - forward

## CanineForQuestionAnswering

[[autodoc]] CanineForQuestionAnswering
    - forward