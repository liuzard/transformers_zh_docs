<!--版权 2020 The HuggingFace Team。版权所有。

根据Apache License, Version 2.0（"许可证"）许可；你不得使用此文件，除非符合
许可证的规定。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证基础上以"原样"分发，
没有明示或暗示的任何担保或条件。请参阅许可证以获取
特定语言的权限和限制。

⚠️注意，此文件是Markdown格式，但包含我们文档构建器的特定语法（类似于MDX），可能无法在Markdown查看器中正确呈现。-->

# 处理器

在Transformers库中，处理器可以有两个不同的含义：
- 用于多模态模型（如[Wav2Vec2](../model_doc/wav2vec2)（语音和文本）或[CLIP](../model_doc/clip)（文本和视觉））预处理输入的对象
- 在旧版本的库中用于预处理GLUE或SQUAD数据的不推荐使用的对象。

## 多模态处理器

任何多模态模型都需要一个用于编码或解码将多个模态（包括文本、视觉和音频）分组的数据的对象。这是由称为处理器的对象处理的，它们将文本模态的标记器（tokenizer）、图像处理器（vision）和特征提取器（audio）等多个处理对象组合在一起。

这些处理器继承了以下实现保存和加载功能的基类：

[[autodoc]] ProcessorMixin

## 不推荐使用的处理器

所有处理器都遵循[`~data.processors.utils.DataProcessor`]的架构。处理器返回一个[`~data.processors.utils.InputExample`]的列表。这些[`~data.processors.utils.InputExample`]可以转换为[`~data.processors.utils.InputFeatures`]以供模型使用。

[[autodoc]] data.processors.utils.DataProcessor

[[autodoc]] data.processors.utils.InputExample

[[autodoc]] data.processors.utils.InputFeatures

## GLUE

[通用语言理解评估（GLUE）](https://gluebenchmark.com/) 是一个基准测试，评估模型在各种现有自然语言理解任务上的性能。它与论文[GLUE: A multi-task benchmark and analysis platform for natural language understanding](https://openreview.net/pdf?id=rJ4km2R5t7)一起发布。

该库共托管了10个处理器，用于以下任务：MRPC、MNLI、MNLI（不匹配）、CoLA、SST2、STSB、QQP、QNLI、RTE和WNLI。

这些处理器是：

- [`~data.processors.utils.MrpcProcessor`]
- [`~data.processors.utils.MnliProcessor`]
- [`~data.processors.utils.MnliMismatchedProcessor`]
- [`~data.processors.utils.Sst2Processor`]
- [`~data.processors.utils.StsbProcessor`]
- [`~data.processors.utils.QqpProcessor`]
- [`~data.processors.utils.QnliProcessor`]
- [`~data.processors.utils.RteProcessor`]
- [`~data.processors.utils.WnliProcessor`]

此外，还可以使用以下方法从数据文件加载值并将其转换为[`~data.processors.utils.InputExample`]列表。

[[autodoc]] data.processors.glue.glue_convert_examples_to_features


## XNLI

[跨语言NLI语料库（XNLI）](https://www.nyu.edu/projects/bowman/xnli/) 是一个基准测试，评估跨语言文本表示的质量。XNLI是一个基于[*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/)的众包数据集，用于15种不同语言（包括高资源语言如英语和低资源语言如斯瓦希里）的文本对进行文本蕴涵注释。

该库托管了用于加载XNLI数据的处理器：

- [`~data.processors.utils.XnliProcessor`]

请注意，由于测试集上有金标签，评估是在测试集上进行的。

在[run_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/text-classification/run_xnli.py)脚本中给出了使用这些处理器的示例。

## SQuAD

[斯坦福问答数据集（SQuAD）](https://rajpurkar.github.io/SQuAD-explorer//) 是一个评估模型在问答任务上性能的基准测试。有两个版本可用，v1.1和v2.0。第一个版本(v1.1)与论文[SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)同时发布。第二个版本(v2.0)与论文[Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822)同时发布。

该库托管了两个版本的处理器：

### 处理器

这些处理器是：

- [`~data.processors.utils.SquadV1Processor`]
- [`~data.processors.utils.SquadV2Processor`]

它们都继承自抽象类[`~data.processors.utils.SquadProcessor`]

[[autodoc]] data.processors.squad.SquadProcessor
    - all

此外，还可以使用以下方法将SQuAD示例转换为[`~data.processors.utils.SquadFeatures`]，以用作模型的输入。

[[autodoc]] data.processors.squad.squad_convert_examples_to_features


这些处理器以及上述方法都可以与包含数据的文件和*tensorflow_datasets*软件包一起使用。以下是示例。

### 使用示例

以下示例演示了使用处理器以及使用数据文件进行转换的方法：

```python
# 加载 V2 处理器
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)

# 加载 V1 处理器
processor = SquadV1Processor()
examples = processor.get_dev_examples(squad_v1_data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

使用*tensorflow_datasets*与使用数据文件一样简单：

```python
# tensorflow_datasets 仅处理 Squad V1。
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

在[run_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py)脚本中给出了使用这些处理器的另一个示例。