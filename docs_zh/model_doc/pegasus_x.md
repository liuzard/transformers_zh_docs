<!--
版权所有2022年的HuggingFace团队。

根据Apache 2.0许可证许可；在遵守许可证的条件下，你不得使用此文件。
你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础进行分发的，没有任何明示或暗示的担保或条件。请查看许可证以了解授予许可的特定语言和限制。

⚠️ 请注意，此文件以Markdown格式编写，但包含我们的doc-builder（类似于MDX）的特定语法，可能无法在Markdown查看器中正确呈现。

-->

# PEGASUS-X

## 概述

PEGASUS-X模型是由Jason Phang、Yao Zhao和Peter J. Liu在[Investigating Efficiently Extending Transformers for Long Input Summarization](https://arxiv.org/abs/2208.04347)中提出的。

PEGASUS-X（PEGASUS扩展版）通过额外的长输入预训练和在编码器中使用交错的块局部注意力和全局标记来扩展PEGASUS模型，以用于长输入摘要。

论文中摘要如下：

*尽管大型预训练Transformer模型已经证明在处理自然语言任务方面非常有效，但处理长序列输入仍然是一个重大挑战。其中一个任务就是长输入摘要，输入的长度超过了大多数预训练模型的最大输入上下文。通过一系列广泛的实验，我们研究了哪些模型架构变化和预训练范例能够最有效地使预训练Transformer适应长输入摘要。我们发现，使用全局编码器标记的交错块局部Transformer取得了性能和效率的良好平衡，并且在长序列上进行附加的预训练阶段可以明显提高下游摘要性能。基于我们的发现，我们介绍了PEGASUS-X，这是PEGASUS模型的扩展版本，通过额外的长输入预训练来处理最长16K标记的输入。PEGASUS-X在长输入摘要任务上取得了与更大模型相当的强大性能，同时添加了少量的附加参数，并且不需要进行模型并行训练。*

提示：

* PEGASUS-X使用与PEGASUS相同的分词器。

该模型由[zphang](<https://huggingface.co/zphang)贡献。原始代码可在[此处](https://github.com/google-research/pegasus)找到。

## 文档资源

- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## PegasusXConfig

[[autodoc]] PegasusXConfig


## PegasusXModel

[[autodoc]] PegasusXModel
    - forward


## PegasusXForConditionalGeneration

[[autodoc]] PegasusXForConditionalGeneration
    - forward
