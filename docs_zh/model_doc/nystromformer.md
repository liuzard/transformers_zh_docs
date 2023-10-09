<!--版权所有2022 HuggingFace团队。保留所有权利。

根据Apache License, Version 2.0（“许可证”）许可;除非您遵守许可证，否则您不得使用此文件。您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据本许可证分发的软件按“按原样”基础分发，不附带任何明示或暗示的担保或条件。有关许可证下各种语言的特定权限和限制的详细信息，请参阅许可证。

⚠️ 请注意，此文件采用Markdown格式，但包含我们文档生成器（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正确显示。

-->

# Nyströmformer

## 概述

Nyströmformer模型是由Yunyang Xiong、Zhanpeng Zeng、Rudrasis Chakraborty、Mingxing Tan、Glenn Fung、Yin Li和Vikas Singh在《Nyströmformer：一种基于Nyström的自注意力近似算法》(https://arxiv.org/abs/2102.03902)中提出的。

论文摘要如下：

*Transformer已经成为广泛应用于自然语言处理任务的强大工具。驱动Transformer出色性能的关键组成部分是自注意机制，它编码了其他标记对每个特定标记的影响或依赖关系。尽管带来了益处，但自注意力在输入序列长度上的二次复杂度限制了它在更长序列上的应用，这是社区正在积极研究的课题。为了解决这个限制，我们提出了Nyströmformer - 一种在序列长度函数上具有良好可扩展性的模型。我们的思想基于将Nyström方法调整为以O(n)复杂度近似标准自注意力。Nyströmformer的可扩展性使其能够应用于具有数千标记的更长序列。我们在GLUE基准测试和IMDB评论的多个下游任务上进行了评估，并发现我们的Nyströmformer表现与标准自注意力相当，甚至在某些情况下稍微更好。在长距离竞技场(Long Range Arena，LRA)基准测试中进行更长序列任务时，Nyströmformer相对于其他高效自注意力方法表现良好。我们的代码在此https URL上可用。*

此模型由[novice03](https://huggingface.co/novice03)贡献。原始代码可在此处找到：[https://github.com/mlpen/Nystromformer](https://github.com/mlpen/Nystromformer)。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [遮蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## NystromformerConfig

[[autodoc]] NystromformerConfig

## NystromformerModel

[[autodoc]] NystromformerModel
    - forward

## NystromformerForMaskedLM

[[autodoc]] NystromformerForMaskedLM
    - forward

## NystromformerForSequenceClassification

[[autodoc]] NystromformerForSequenceClassification
    - forward

## NystromformerForMultipleChoice

[[autodoc]] NystromformerForMultipleChoice
    - forward

## NystromformerForTokenClassification

[[autodoc]] NystromformerForTokenClassification
    - forward

## NystromformerForQuestionAnswering

[[autodoc]] NystromformerForQuestionAnswering
    - forward