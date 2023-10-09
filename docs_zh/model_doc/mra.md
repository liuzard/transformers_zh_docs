<!--版权所有2023年The HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”）获得许可；除非符合许可证，否则你不得使用此文件。
你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言和限制。

⚠请注意，此文件以Markdown格式编写，但包含我们的文档构建器（类似于MDX）的特定语法，可能无法在你的Markdown查看器中正确呈现。

-->

# MRA

## 概述

MRA模型在[Zhanpeng Zeng, Sourav Pal, Jeffery Kline, Glenn M Fung, and Vikas Singh的“Multi Resolution Analysis (MRA) for Approximate Self-Attention”](https://arxiv.org/abs/2207.10284)中提出。

来自论文的摘要如下：

*Transformer已经成为自然语言处理和视觉领域许多任务的首选模型。对于训练和部署Transformer更高效的最近工作，已经确定了许多用于近似自注意力矩阵（Transformer架构的关键模块）的策略。有效的想法包括各种预定稀疏模式、低秩基展开以及它们的组合。在本文中，我们对经典的多分辨率分析（MRA）概念，比如小波，进行了重新检验，这些概念在该领域的潜在价值迄今为止还未充分发掘。我们展示了基于经验反馈和现代硬件和实现挑战指导的设计选择的简单近似的MRA自注意力方法，在大多数感兴趣的标准方面具有出色的性能。我们进行了一系列广泛的实验证明，这种多分辨率方案优于大多数高效的自注意力提议，并且对于短序列和长序列都是有利的。代码可在https://github.com/mlpen/mra-attention找到。*

此模型由[novice03](https://huggingface.co/novice03)贡献。
原始代码可在[此处](https://github.com/mlpen/mra-attention)找到。


## MraConfig

[[autodoc]] MraConfig


## MraModel

[[autodoc]] MraModel
    - forward


## MraForMaskedLM

[[autodoc]] MraForMaskedLM
    - forward


## MraForSequenceClassification

[[autodoc]] MraForSequenceClassification
    - forward

## MraForMultipleChoice

[[autodoc]] MraForMultipleChoice
    - forward


## MraForTokenClassification

[[autodoc]] MraForTokenClassification
    - forward


## MraForQuestionAnswering

[[autodoc]] MraForQuestionAnswering
    - forward