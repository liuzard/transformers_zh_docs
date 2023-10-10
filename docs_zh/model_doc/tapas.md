版权所有 © 2020 HuggingFace 团队。

根据 Apache 许可证第 2.0 版 ("许可证")，你不能在未遵守许可证的情况下使用此文件。
你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

根据适用法律或书面同意，软件根据许可证被分发，基于 "现状" 分发，没有任何形式的保证或条件。详细信息请参阅许可证，以及许可证的限制。

⚠️ 注意，此文件是 Markdown 格式，但包含了用于我们文档构建器 (类似于 MDX) 的特定语法，可能无法在你的 Markdown 查看器中正确呈现。

# TAPAS

## 概述

TAPAS 模型是由 Jonathan Herzig、Paweł Krzysztof Nowak、Thomas Müller、Francesco Piccinno 和 Julian Martin Eisenschlos 在 [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://www.aclweb.org/anthology/2020.acl-main.398) 中提出的。它是一种基于 BERT 的模型，专门用于回答关于表格数据的问题。与 BERT 相比，TAPAS 使用了相对位置嵌入，且有 7 种标记类型来编码表格结构。TAPAS 在大量包含来自英文维基百科的数百万个表格和相应文本的数据集上进行掩蔽语言建模（MLM）目标的预训练。

在问答方面，TAPAS 有两个输出头部：一个单元格选择头部和一个聚合头部，用于在选定的单元格中（可选地）执行聚合操作（如计数或求和）。TAPAS 已经在几个数据集上进行了微调：
- [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253)（由微软提供的顺序问答数据集）
- [WTQ](https://github.com/ppasupat/WikiTableQuestions)（由斯坦福大学提供的维基表格问答数据集）
- [WikiSQL](https://github.com/salesforce/WikiSQL)（由 Salesforce 提供的维基SQL数据集）

TAPAS 在 SQA 和 WTQ 上达到了最先进的水平，在 WikiSQL 上的性能与最先进的方法相当，但其架构更简单。

论文中的摘要如下：

*通常，以表格为基础回答自然语言问题被视为一项语义解析任务。为减少完整逻辑形式的收集成本，一种流行的方法是专注于弱监督，即只提供底层数据而不提供逻辑形式。然而，从弱监督训练语义解析器存在困难，并且生成的逻辑形式仅在检索底层数据之前作为中间步骤使用。本文提出了 TAPAS，一种无需生成逻辑形式就能回答表格问题的方法。TAPAS 从弱监督训练，并通过选择表格单元格并可选地应用相应的聚合运算符到此选择上来预测底层数据。TAPAS 扩展了 BERT 的架构以对输入表格进行编码，并从从维基百科抓取的文本片段和表格的有效联合预训练中进行初始化，并进行端到端的训练。我们尝试了三个不同的语义解析数据集，并发现 TAPAS 在 SQA 上将最先进的准确率从 55.1 提高到 67.2，在 WIKISQL 和 WIKITQ 上的性能与最先进的方法相当，但具有更简单的模型架构。我们还发现，在我们的设置中，从 WIKISQL 到 WIKITQ 的转移学习（这在我们的设置中是微不足道的）产生了 48.7 的准确率，比最先进的结果高出 4.2 个点。*

此外，TAPAS 的作者还进行了进一步的预训练，以识别表格推理，通过创建了一组数百万个自动生成的训练示例的平衡数据集，在微调之前进行了中间步骤的学习。TAPAS 的作者将这种进一步的预训练称为中间预训练（因为 TAPAS 首先进行 MLM 预训练，然后在另一个数据集上进行预训练）。他们发现，中间预训练在 SQA 上进一步提高了性能，达到了最新的最先进结果，以及在 [TabFact](https://github.com/wenhuchen/Table-Fact-Checking) 上的最先进结果，TabFact 是一个拥有 16k 维基百科表格的大规模数据集，用于表格推理（二进制分类任务）。有关更多详细信息，请参阅他们的后续论文：[Understanding tables with intermediate pre-training](https://www.aclweb.org/anthology/2020.findings-emnlp.27/)，作者是 Julian Martin Eisenschlos、Syrine Krichene 和 Thomas Müller。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tapas_architecture.png"
alt="drawing" width="600"/> 

<small> TAPAS 架构。来自[原博客文章](https://ai.googleblog.com/2020/04/using-neural-networks-to-find-answers.html)。</small>

此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。这个模型的 Tensorflow 版本由 [kamalkraj](https://huggingface.co/kamalkraj) 贡献。原始代码可以在[这里](https://github.com/google-research/tapas)找到。

提示：

- TAPAS 模型默认使用相对位置嵌入（在表格的每个单元格中重新开始位置嵌入）。请注意，这是在原始 TAPAS 论文发表之后添加的。根据作者的说法，这通常会导致稍微更好的性能，并且允许你在不耗尽嵌入的情况下对较长的序列进行编码。这反映在 [`TapasConfig`] 的 `reset_position_index_per_cell` 参数中，默认设置为 `True`。可在 [hub](https://huggingface.co/models?search=tapas) 上提供的默认版本的模型中使用相对位置嵌入。如果要使用绝对位置嵌入，可以在调用 `from_pretrained()` 方法时传入一个额外的参数 `revision="no_reset"`。通常建议在右侧而不是左侧填充输入。
- TAPAS 基于 BERT，因此例如 `TAPAS-base` 对应于 `BERT-base` 架构。当然，`TAPAS-large` 的性能最好（论文中的结果是根据 `TAPAS-large` 得出的）。各种大小模型的结果显示在[原始的Github存储库](https://github.com/google-research/tapas>)上。
- TAPAS 有在 SQA 上进行微调的检查点，可以回答与表格相关的问题，也可以在对话设置中提问后续问题，例如 "他多大了？"。请注意，对于对话设置，TAPAS 的前向传递有所不同：在这种情况下，你必须逐一将每个表格-问题对输入模型，以便预测的 `labels` 可以覆盖模型对于先前问题的 `prev_labels` token类型 ID。有关更多信息，请参见"用法"部分。
- TAPAS 类似于 BERT，因此依赖于掩蔽语言建模 (MLM) 目标。因此，它在预测掩盖的token和自然语言理解方面效率很高，但对于文本生成来说并不是最佳选择。使用因果语言建模 (CLM) 目标训练的模型在这方面更好。请注意，可以在 EncoderDecoderModel 框架中使用 TAPAS 作为编码器，将其与自回归文本解码器（如 GPT-2）结合使用。

## 使用：微调

在这里，我们将解释如何在你自己的数据集上对 [`TapasForQuestionAnswering`] 进行微调。

**步骤 1：选择你使用 TAPAS 的 3 种方式之一 - 或进行实验**

基本上，可以通过三种不同的方式对 [`TapasForQuestionAnswering`] 进行微调，对应于对 Tapas 进行微调的不同数据集：

1. SQA：如果你想在对话设置中提问与表格相关的后续问题。例如，如果你先问 "第一位男演员的名字是什么？"，然后可以问一个后续问题，比如 "他多大了？"。在这里，问题不涉及任何聚合（所有问题都是单元格选择问题）。
2. WTQ：如果你不想在对话设置中提问问题，而是只是问与表格相关的问题，这些问题可能涉及聚合，例如计算行数、求和单元格值或求平均单元格值。然后你可以询问 "Cristiano Ronaldo 职业生涯进球总数是多少？"。这种情况也被称为**弱监督**，因为模型本身必须在只有问题的答案作为监督的情况下，学习适当的聚合操作符（SUM/COUNT/AVERAGE/NONE）。
3. WikiSQL-监督：这个数据集基于 WikiSQL，模型在训练过程中被给予了真实的聚合操作符。这也被称为**强监督**。在这种情况下，学习适当的聚合操作符要简单得多。

总结一下：

| **任务**                            | **示例数据集** | **描述**                                                                                         |
|-------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------|
| 对话式                    | SQA                 | 对话式，仅进行单元格选择的问题                                                           |
| 弱监督聚合    | WTQ                 | 问题可能涉及聚合，并且模型必须根据问题的答案作为监督来学习适当的聚合操作符 |
| 强监督聚合  | WikiSQL-监督  | 问题可能涉及聚合，并且模型必须根据黄金聚合操作符来学习适当的聚合操作符  |

<frameworkcontent>
<pt>
可以像下面展示的那样，使用预训练的基本模型和随机初始化的分类头部，从 hub 中初始化一个模型。

```py
>>> from transformers import TapasConfig, TapasForQuestionAnswering

>>> # 例如，使用默认的 SQA 配置的基础型号
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base")

>>> # 或者，使用 WTQ 配置的基础型号
>>> config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> # 或者，使用 WikiSQL 配置的基础型号
>>> config = TapasConfig("google-base-finetuned-wikisql-supervised")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

当然，你不一定要按照 TAPAS 的这三种方式之一进行操作。在初始化 [`TapasConfig`] 时，你也可以根据需要定义任何超参数，然后基于该配置创建 [`TapasForQuestionAnswering`]。例如，如果你的数据集既包含对话式问题又包含可能涉及聚合的问题，则可以按照以下方式进行：例如：

```py
>>> from transformers import TapasConfig, TapasForQuestionAnswering

>>> # 可以选择任意初始化分类头部的方式（请参阅 TapasConfig 的文档）
>>> config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True)
>>> # 使用自定义的分类头部初始化预训练的基础型号
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```
</pt>
<tf>
可以像下面展示的那样，使用预训练的基本模型和随机初始化的分类头部，从 hub 中初始化一个模型。请确保已安装 [tensorflow_probability](https://github.com/tensorflow/probability) 依赖包：

```py
>>> from transformers import TapasConfig, TFTapasForQuestionAnswering

>>> # 例如，使用默认的 SQA 配置的基础型号
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base")

>>> # 或者，使用 WTQ 配置的基础型号
>>> config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> # 或者，使用 WikiSQL 配置的基础型号
>>> config = TapasConfig("google-base-finetuned-wikisql-supervised")
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

当然，你不一定要按照 TAPAS 的这三种方式之一进行操作。在初始化 [`TapasConfig`] 时，你也可以根据需要定义任何超参数，然后基于该配置创建 [`TFTapasForQuestionAnswering`]。例如，如果你的数据集既包含对话式问题又包含可能涉及聚合的问题，则可以按照以下方式进行：例如：

```py
>>> from transformers import TapasConfig, TFTapasForQuestionAnswering

>>> # 可以选择任意初始化分类头部的方式（请参阅 TapasConfig 的文档）
>>> config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True)
>>> # 使用自定义的分类头部初始化预训练的基础型号
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```
</tf>
</frameworkcontent>

你还可以从已经进行微调的检查点开始。在此说明需要注意的是，由于 L2 损失的一些问题，WTQ 上的已经微调的检查点有些脆弱。有关更多信息，请参阅[此处](https://github.com/google-research/tapas/issues/91#issuecomment-735719340)。

有关 HuggingFace hub 上可用的所有预训练和微调 TAPAS 检查点的列表，请参见[此处](https://huggingface.co/models?search=tapas)。

**步骤 2：将数据准备成 SQA 格式**

无论你选择了上述哪种方式，你都应该将数据准备成 [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253) 格式。这种格式是一个具有以下列的 TSV/CSV 文件：

- `id`：可选，表格-问题对的 ID，用于记录目的。
- `annotator`：可选，注释表格-问题对的人员的 ID，用于记录目的。
- `position`：指示问题是与表格相关的第一个、第二个、第三个等的整数。只在对话设置（SQA）的情况下需要。WTQ/WikiSQL-监督情况下不需要此列。
- `question`：字符串
- `table_file`：字符串，包含表格数据的 csv 文件的名称
- `answer_coordinates`：一个或多个元组的列表（每个元组都是答案的一个单元格坐标，即行列对）
- `answer_text`：一个或多个字符串的列表（每个字符串都是答案的一个单元格值）
- `aggregation_label`：聚合操作符的索引。在强监督聚合（WikiSQL-监督）的情况下需要
- `float_answer`：问题的浮点数答案，如果有的话（如果没有则为 np.nan）。在弱监督聚合（如 WTQ 和 WikiSQL）的情况下需要

**使用方法：推断**

```py
import torch
from transformers import TapasTokenizer, TapasForQuestionAnswering

model_name = "google/tapas-base"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

# Prepare the input
data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
queries = [
    "What is the name of the first actor?",
    "How many movies has George Clooney played in?",
    "What is the total number of movies?",
]
answer_coordinates = [[(0, 0)], [(2, 1)], [(0, 1), (1, 1), (2, 1)]]
answer_text = [["Brad Pitt"], ["69"], ["209"]]
table = pd.DataFrame.from_dict(data)
inputs = tokenizer(
    table=table,
    queries=queries,
    answer_coordinates=answer_coordinates,
    answer_text=answer_text,
    padding="max_length",
    return_tensors="pt",
)

# Perform the inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the predictions
predicted_answer_coordinates = outputs.predicted_answer_coordinates
predicted_answer_text = outputs.predicted_answer_text

print(predicted_answer_coordinates)
print(predicted_answer_text)
```

请根据你的需求修改输入数据和使用的模型，然后运行以上代码进行推断。

## TAPAS模型推理的使用方法

在这里，我们解释如何在推理过程中使用[TapasForQuestionAnswering]或[TFTapasForQuestionAnswering]（即在新数据上进行预测）。对于推理，模型只需要提供`input_ids`，`attention_mask`和`token_type_ids`（可以使用[TapasTokenizer]获取）即可获得logits。接下来，你可以使用方便的[`~models.tapas.tokenization_tapas.convert_logits_to_predictions`]方法将它们转换为预测的坐标和可选的聚合指数。

但是，请注意，推理的方式取决于设置是否是对话式的。在非对话式设置中，可以并行地对批处理中的所有表格-问题对进行推理。以下是一个例子:

```py
>>> from transformers import TapasTokenizer, TapasForQuestionAnswering
>>> import pandas as pd

>>> model_name = "google/tapas-base-finetuned-wtq"
>>> model = TapasForQuestionAnswering.from_pretrained(model_name)
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> outputs = model(**inputs)
>>> predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
...     inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
... )

>>> # let's print out the results:
>>> id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
>>> aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

>>> answers = []
>>> for coordinates in predicted_answer_coordinates:
...     if len(coordinates) == 1:
...         # only a single cell:
...         answers.append(table.iat[coordinates[0]])
...     else:
...         # multiple cells
...         cell_values = []
...         for coordinate in coordinates:
...             cell_values.append(table.iat[coordinate])
...         answers.append(", ".join(cell_values))

>>> display(table)
>>> print("")
>>> for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
...     print(query)
...     if predicted_agg == "NONE":
...         print("Predicted answer: " + answer)
...     else:
...         print("Predicted answer: " + predicted_agg + " > " + answer)
What is the name of the first actor?
Predicted answer: Brad Pitt
How many movies has George Clooney played in?
Predicted answer: COUNT > 69
What is the total number of movies?
Predicted answer: SUM > 87, 53, 69
```

对于对话式设置，则必须**依次**提供每个表格-问题对给模型，以便先前的表格-问题对的`prev_labels` token类型可以被预测的上一个表格-问题对的`labels`覆盖。再次说明，更多信息可以在[此笔记本](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb)（适用于PyTorch）和[此笔记本](https://github.com/kamalkraj/Tapas-Tutorial/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb)（适用于TensorFlow）中找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)

## TAPAS特定输出
[[autodoc]] models.tapas.modeling_tapas.TableQuestionAnsweringOutput

## TapasConfig
[[autodoc]] TapasConfig

## TapasTokenizer
[[autodoc]] TapasTokenizer
    - __call__
    - convert_logits_to_predictions
    - save_vocabulary

## TapasModel
[[autodoc]] TapasModel
    - forward
    
## TapasForMaskedLM
[[autodoc]] TapasForMaskedLM
    - forward

## TapasForSequenceClassification
[[autodoc]] TapasForSequenceClassification
    - forward
    
## TapasForQuestionAnswering
[[autodoc]] TapasForQuestionAnswering
    - forward

## TFTapasModel
[[autodoc]] TFTapasModel
    - call
    
## TFTapasForMaskedLM
[[autodoc]] TFTapasForMaskedLM
    - call

## TFTapasForSequenceClassification
[[autodoc]] TFTapasForSequenceClassification
    - call
    
## TFTapasForQuestionAnswering
[[autodoc]] TFTapasForQuestionAnswering
    - call