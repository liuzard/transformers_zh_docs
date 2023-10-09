<!--版权所有2020年The HuggingFace团队。保留所有权利。

根据Apache License, Version 2.0许可证授权，除非符合许可证，否则您不得使用此文件。
您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证发布的软件是基于“按原样”分发的，没有任何形式的担保或条件，不论是明示的还是暗示的。
有关特定语言的详细信息，请参阅许可证中的许可证。

⚠️请注意，此文件以Markdown格式编写，但包含与我们的文档构建器（类似于MDX）的特定语法，该语法在您的Markdown查看器中可能无法正确渲染。

-->

# BORT

<Tip warning={true}>

此模型仅处于维护模式，因此我们不会接受任何更改其代码的新请求。

如果您在运行此模型时遇到任何问题，请重新安装支持此模型的最后一个版本：v4.30.0。
您可以通过运行以下命令来执行此操作：`pip install -U transformers==4.30.0`。

</Tip>

## 概述

BORT模型是由Adrian de Wynter和Daniel J. Perry在[《Optimal Subarchitecture Extraction for BERT》](https://arxiv.org/abs/2010.10499)一文中提出的。它是BERT体系结构的一种最佳子体系结构参数子集，作者将其称为“Bort”。

论文中的摘要如下：

*我们通过应用最新的神经体系结构搜索算法，在Devlin等人（2018年）的BERT体系结构中提取了一种最佳的体系结构参数子集。这种最佳子集被我们称为“Bort”，它明显更小，其有效大小（不包括嵌入层）为原始BERT-large体系结构的5.5%，相当于净大小的16%。Bort也能够在288个GPU小时内进行预训练，这相当于预训练性能最好的BERT参数体系结构变体RoBERTa-large（Liu等人，2019年）所需时间的1.2%，以及相同硬件上训练BERT-large所需GPU小时的世界纪录的约33%。它在CPU上的速度也要快7.9倍，而且性能要优于体系结构的其他压缩变体和一些非压缩变体：在多个公开的自然语言理解（NLU）基准测试中，它相对于BERT-large的性能提高了0.3%到31%，绝对值。*

提示：

- BORT的模型体系结构基于BERT，因此可以参考[BERT的文档页面](bert)获取模型的API以及用法示例。
- BORT使用RoBERTa分词器而不是BERT分词器，因此可以参考[RoBERTa的文档页面](roberta)获取分词器的API以及用法示例。
- BORT需要一种特定的微调算法，称为[Agora](https://adewynter.github.io/notes/bort_algorithms_and_applications.html#fine-tuning-with-algebraic-topology)，很遗憾这个算法尚未开源。如果有人尝试实现该算法使BORT微调工作，那对社区将非常有用。

此模型由[stefan-it](https://huggingface.co/stefan-it)贡献。原始代码可在[此处](https://github.com/alexa/bort/)找到。