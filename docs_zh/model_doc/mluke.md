<!--版权2021 The HuggingFace 团队。保留所有权利。

根据Apache许可证，版本2.0（"许可证"）许可；除非符合所述许可证，否则不得使用此文件。
您可以获得许可证的副本，许可证网址为

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于"原样"的基础，不附带任何形式的明示或暗示的保证。
有关特定语言的限制和限制的信息，请参阅许可证。

⚠️ 请注意，此文件是Markdown格式，但包含供我们的文档构建程序（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确显示。

-->

# mLUKE

## 概述

mLUKE模型是由Ryokan Ri，Ikuya Yamada和Yoshimasa Tsuruoka在[mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models](https://arxiv.org/abs/2110.08151)中提出的。它是[LUKE模型](https://arxiv.org/abs/2010.01057)的多语言扩展，基于XLM-RoBERTa进行训练。

它基于XLM-RoBERTa并添加了实体嵌入，有助于改善涉及对实体进行推理的各种下游任务的性能，如命名实体识别，抽取式问答，关系分类，完形填空式知识补充。

来自论文的摘要如下：

*最近的研究表明，多语言预训练语言模型可以通过来自维基百科实体的跨语言对齐信息有效地改进。然而，现有方法仅在预训练中利用实体信息，并不明确地在下游任务中使用实体。在这项研究中，我们探讨了利用实体表示进行下游跨语言任务的有效性。我们使用24种语言训练了一个带有实体表示的多语言语言模型，并展示了该模型在各种跨语言迁移任务中始终优于基于单词的预训练模型。我们还分析了该模型，关键见解是将实体表示纳入输入中使我们能够提取更多与语言无关的特征。我们还使用mLAMA数据集对模型进行了多语言完型填充提示任务的评估。我们表明，与仅使用单词表示相比，基于实体的提示更有可能引出正确的实际知识。*

可以直接将mLUKE的权重插入到LUKE模型中，如下所示：

```python
from transformers import LukeModel

model = LukeModel.from_pretrained("studio-ousia/mluke-base")
```

请注意，mLUKE有自己的标记器[`MLukeTokenizer`]。您可以按以下方式初始化它：

```python
from transformers import MLukeTokenizer

tokenizer = MLukeTokenizer.from_pretrained("studio-ousia/mluke-base")
```

由于mLUKE的架构等同于LUKE，因此可以参考[LUKE的文档页面](luke)获取所有提示、代码示例和笔记本。

此模型由[ryo0634](https://huggingface.co/ryo0634)贡献。原始代码可以在[这里](https://github.com/studio-ousia/luke)找到。

## MLukeTokenizer

[[autodoc]] MLukeTokenizer
    - __call__
    - save_vocabulary