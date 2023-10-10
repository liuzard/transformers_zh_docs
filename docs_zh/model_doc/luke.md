<!--版权所有 2021 年 The HuggingFace 团队。保留所有权利。

根据 Apache 许可证，第 2.0 版（“许可证”），除非遵从该许可证，否则不得使用此文件。
你可以在下面获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据此许可证分发的软件是基于“按原样”分发的，
没有任何明示或暗示的担保或条件。有关许可证下的特定语言的详细信息，请参阅许可证。

⚠️ 请注意，此文件以 Markdown 格式编写，但包含特定语法，适用于我们的 doc-builder（类似于 MDX），
可能无法在你的 Markdown 查看器中正确显示。

-->

# LUKE

## 概述

LUKE 模型在 Yamada 等人提出的文章 [LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://arxiv.org/abs/2010.01057) 中介绍。
该模型基于 RoBERTa，添加了实体嵌入和实体感知的自注意机制，这有助于在涉及实体推理的各种下游任务中提高性能，如命名实体识别、抽取式和填空式问答、实体类型和关系分类。

论文中的摘要如下所示：

*实体表示在涉及实体的自然语言任务中非常有用。在本文中，我们提出了一种基于双向 Transformer 的单词和实体的新的预训练上下文表示方法。所提出的模型将给定文本中的单词和实体视为独立的标记，并输出有关它们的上下文表示。我们的模型使用基于 BERT 的掩蔽语言模型的新的预训练任务进行训练。该任务涉及在从维基百科获取的大型实体注释语料库中预测随机掩蔽的单词和实体。我们还提出了一种实体感知的自注意机制，该机制是 Transformer 自注意机制的扩展，并在计算注意力分数时考虑标记的类型（单词或实体）。所提出的模型在各种涉及实体的任务上取得了令人印象深刻的实证性能。特别是，在五个众所周知的数据集上，它获得了最先进的结果：Open Entity（实体类型）、TACRED（关系分类）、CoNLL-2003（命名实体识别）、ReCoRD（填空式问答）和 SQuAD 1.1（抽取式问答）。

提示：

- 这个实现与 [`RobertaModel`] 相同，只是增加了实体嵌入以及实体感知的自注意机制来改善涉及实体推理的任务性能。
- LUKE 将实体视为输入标记；因此，它需要额外的输入参数 `entity_ids`、`entity_attention_mask`、`entity_token_type_ids` 和 `entity_position_ids`。你可以使用
  [`LukeTokenizer`] 获得这些参数。
- [`LukeTokenizer`] 接受额外的输入参数 `entities` 和 `entity_spans`（输入文本中实体的基于字符的起始和结束位置）。`entities` 通常包含 [MASK] 类型的实体或维基百科实体。输入这些实体时的简要说明如下:

  - *输入 [MASK] 类型实体以计算实体表示*：[MASK] 类型的实体用于在预训练期间掩盖待预测的实体。当 LUKE 接收到 [MASK] 类型实体时，它会尽量通过从输入文本中收集关于该实体的信息来预测原始实体。因此，[MASK] 类型实体可用于涉及实体信息的下游任务，如实体类型识别、关系分类和命名实体识别。
  - *输入维基百科实体以计算知识增强型token表示*：LUKE 在预训练期间学习了维基百科实体的丰富信息（或知识），并将该信息存储在其实体嵌入中。通过使用维基百科实体作为输入标记，LUKE 输出通过包含在这些实体嵌入中存储的信息而丰富的token表示。这对于需要现实世界知识的任务（如问答）特别有效。

- 前面的用例有三个头模型：

  - [`LukeForEntityClassification`] 用于对输入文本中的单个实体进行分类的任务，例如实体类型识别，例如 [Open Entity 数据集](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html)。该模型在输出实体表示之上放置了一个线性头。
  - [`LukeForEntityPairClassification`] 用于对两个实体之间的关系进行分类的任务，例如关系分类，例如 [TACRED 数据集](https://nlp.stanford.edu/projects/tacred/)。该模型在给定实体对的输出表示的基础上放置了一个线性头。
  - [`LukeForEntitySpanClassification`] 用于对实体跨度序列进行分类的任务，例如命名实体识别（NER）。该模型在输出实体表示之上放置了一个线性头。你可以通过将文本中的所有可能实体跨度输入模型来使用此模型进行 NER。

  [`LukeTokenizer`] 有一个 `task` 参数，你可以通过指定 `task="entity_classification"`、`task="entity_pair_classification"` 或
  `task="entity_span_classification"` 来轻松地为这些头模型创建输入。请参阅每个头模型的示例代码。

  你可以在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LUKE)找到一个演示笔记本，介绍了如何对 [`LukeForEntityPairClassification`] 进行微调，用于关系分类。

  还有 3 个可用的笔记本，展示了如何使用 HuggingFace 的 LUKE 实现按照论文中报告的结果进行复现。它们可以在[此处](https://github.com/studio-ousia/luke/tree/master/notebooks)找到。

示例：

```python
>>> from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification

>>> model = LukeModel.from_pretrained("studio-ousia/luke-base")
>>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
# 示例 1：计算与实体提及 "Beyoncé" 相对应的上下文化实体表示

>>> text = "Beyoncé lives in Los Angeles."
>>> entity_spans = [(0, 7)]  # 与 "Beyoncé" 对应的基于字符的实体跨度
>>> inputs = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
>>> outputs = model(**inputs)
>>> word_last_hidden_state = outputs.last_hidden_state
>>> entity_last_hidden_state = outputs.entity_last_hidden_state
# 示例 2：输入维基百科实体以获取丰富的上下文化表示

>>> entities = [
...     "Beyoncé",
...     "Los Angeles",
... ]  # 与实体提及 "Beyoncé" 和 "Los Angeles" 对应的维基百科实体标题
>>> entity_spans = [(0, 7), (17, 28)]  # 与 "Beyoncé" 和 "Los Angeles" 对应的基于字符的实体跨度
>>> inputs = tokenizer(text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
>>> outputs = model(**inputs)
>>> word_last_hidden_state = outputs.last_hidden_state
>>> entity_last_hidden_state = outputs.entity_last_hidden_state
# 示例 3：使用 LukeForEntityPairClassification 头模型对两个实体之间的关系进行分类

>>> model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
>>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
>>> entity_spans = [(0, 7), (17, 28)]  # 与 "Beyoncé" 和 "Los Angeles" 对应的基于字符的实体跨度
>>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> predicted_class_idx = int(logits[0].argmax())
>>> print("预测的类别:", model.config.id2label[predicted_class_idx])
```

此模型由 [ikuyamada](https://huggingface.co/ikuyamada) 和 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可在[此处](https://github.com/studio-ousia/luke)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩蔽语言模型任务指南](../tasks/masked_language_modeling)
- [多选任务指南](../tasks/multiple_choice)

## LukeConfig

[[autodoc]] LukeConfig

## LukeTokenizer

[[autodoc]] LukeTokenizer
    - __call__
    - save_vocabulary

## LukeModel

[[autodoc]] LukeModel
    - forward

## LukeForMaskedLM

[[autodoc]] LukeForMaskedLM
    - forward

## LukeForEntityClassification

[[autodoc]] LukeForEntityClassification
    - forward

## LukeForEntityPairClassification

[[autodoc]] LukeForEntityPairClassification
    - forward

## LukeForEntitySpanClassification

[[autodoc]] LukeForEntitySpanClassification
    - forward

## LukeForSequenceClassification

[[autodoc]] LukeForSequenceClassification
    - forward

## LukeForMultipleChoice

[[autodoc]] LukeForMultipleChoice
    - forward

## LukeForTokenClassification

[[autodoc]] LukeForTokenClassification
    - forward

## LukeForQuestionAnswering

[[autodoc]] LukeForQuestionAnswering
    - forward