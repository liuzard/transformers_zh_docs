<!--版权所有 2022 The HuggingFace团队。保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”）许可；除非遵守许可证，否则不得使用此文件。
你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据本许可证分发的软件是基于“按原样” 分发的，不附带任何担保或条件，无论是明示的还是默示的。
有关许可的特定语言和限制，请参阅许可证。

⚠️ 请注意，此文件使用 Markdown 编写，但包含我们的文档构建器的特定语法（类似于 MDX），可能在你的 Markdown 查看器中无法正常呈现。-->

# TAPEX

<Tip warning={true}>

此模型仅处于维护模式，因此我们不会接受任何更改其代码的新 PR。

如果在运行此模型时遇到任何问题，请重新安装支持该模型的最后一个版本：v4.30.0。
你可以通过运行以下命令来执行此操作：`pip install -U transformers==4.30.0`。

</Tip>

## 概述

TAPEX 模型是由 Qian Liu、Bei Chen、Jiaqi Guo、Morteza Ziyadi、Zeqi Lin、Weizhu Chen 和 Jian-Guang Lou 在《TAPEX: Table Pre-training via Learning a Neural SQL Executor》一文中提出的。TAPEX 在合成的 SQL 查询问题上预训练了一个 BART 模型，然后可以通过微调来回答与表格数据相关的自然语言问题，以及进行表格事实检查。

TAPEX 已在多个数据集上进行了微调：
- [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253)（由 Microsoft 提供的顺序问答数据集）
- [WTQ](https://github.com/ppasupat/WikiTableQuestions)（由 Stanford University 提供的 Wiki 表格问答数据集）
- [WikiSQL](https://github.com/salesforce/WikiSQL)（由 Salesforce 提供的数据集）
- [TabFact](https://tabfact.github.io/)（由 USCB NLP Lab 提供的数据集）。

该论文的摘要如下：

*最近，通过利用大规模非结构化文本数据进行语言模型预训练取得了巨大成功。然而，由于缺乏大规模高质量的表格数据，将预训练应用于结构化的表格数据仍然是一项挑战。在本文中，我们提出了 TAPEX，通过学习一个基于神经网络的 SQL 执行器，来展示可以通过合成 SQL 查询来进行表格预训练。我们通过引导语言模型在多样化、大规模和高质量的合成语料库上模仿 SQL 执行器，以解决数据稀缺的挑战。我们在四个基准数据集上评估了 TAPEX。实验结果表明，TAPEX 在所有数据集上优于先前的表格预训练方法，达到了新的最佳结果。这包括将弱监督的 WikiSQL 符号对应准确率提高到 89.5%（+2.3%），将 WikiTableQuestions 符号对应准确率提高到 57.5%（+4.8%），将 SQA 符号对应准确率提高到 74.5%（+3.5%），以及将 TabFact 准确率提高到 84.2%（+3.2%）。据我们所知，这是第一项利用合成可执行程序进行表格预训练，并在各种下游任务上取得新的最佳结果的工作。*

提示：

- TAPEX 是一个生成型（seq2seq）模型。可以直接将 TAPEX 的权重插入到一个 BART 模型中。
- TAPEX 在模型中心有检查点，这些检查点仅经过预训练，或在 WTQ、SQA、WikiSQL 和 TabFact 上进行了微调。
- 将句子 + 表格以 `sentence + " " + linearized table` 的形式提供给模型。线性化表格的格式如下：
  `col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...`。
- TAPEX 有自己的 tokenizer，可以轻松地为模型准备所有数据。可以将 Pandas 数据帧和字符串传递给 tokenizer，
  它会自动创建 `input_ids` 和 `attention_mask`（如下面的使用示例所示）。

## 使用方法：推断

下面，我们演示了如何在表格问答中使用 TAPEX。正如可以看到的，可以直接将 TAPEX 的权重插入到 BART 模型中。
我们使用 [Auto API](auto)，它将根据模型中心的配置文件自动实例化适当的 tokenizer（[`TapexTokenizer`]）和模型（[`BartForConditionalGeneration`]）。

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large-finetuned-wtq")

>>> # 准备表格 + 问题
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> question = "利昂纳多·迪卡普里奥演了多少部电影？"

>>> encoding = tokenizer(table, question, return_tensors="pt")

>>> # 让模型自动生成答案
>>> outputs = model.generate(**encoding)

>>> # 解码为文本
>>> predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
>>> print(predicted_answer)
53
```

请注意，[`TapexTokenizer`] 还支持批量推断。因此，可以提供一批不同的表格/问题、一批单一表格和多个问题，或一批单一查询和多个表格。
让我们来举个例子：

```python
>>> # 准备表格 + 问题
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> questions = [
...     "利昂纳多·迪卡普里奥演了多少部电影？",
...     "哪个演员有 69 部电影？",
...     "有 87 部电影的演员的名字是什么？",
... ]
>>> encoding = tokenizer(table, questions, padding=True, return_tensors="pt")

>>> # 让模型自动生成答案
>>> outputs = model.generate(**encoding)

>>> # 解码为文本
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['53', '乔治·克鲁尼', '布拉德·皮特']
```

如果要进行表格验证（即确定给定句子是否由表格内容支持或反驳），可以实例化一个 [`BartForSequenceClassification`] 模型。
TAPEX 在模型中心上具有针对 TabFact 进行微调的检查点，这是一个重要的表格事实检查基准（准确率为 84%）。
以下代码示例再次利用了[Auto API](auto)。

```python
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-tabfact")
>>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/tapex-large-finetuned-tabfact")

>>> # 准备表格 + 句子
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> sentence = "乔治·克鲁尼拍过 30 部电影"

>>> encoding = tokenizer(table, sentence, return_tensors="pt")

>>> # 前向传播
>>> outputs = model(**encoding)

>>> # 打印预测结果
>>> predicted_class_idx = outputs.logits[0].argmax(dim=0).item()
>>> print(model.config.id2label[predicted_class_idx])
Refused
```


## TapexTokenizer

[[autodoc]] TapexTokenizer
    - __call__
    - save_vocabulary