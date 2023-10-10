<!--
版权所有2023 The HuggingFace团队。

根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证的规定，否则你不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，按“原样”方式分发的软件都不附带任何明示或暗示的担保或条件。请查阅许可证以了解许可证下的特定语言和限制。

⚠️ 请注意，此文件是Markdown格式，但包含了我们的文档生成器（类似于MDX）的特定语法，你的Markdown查看器可能无法正确渲染。

-->

# MEGA

## 概述

[《Mega: Moving Average Equipped Gated Attention》](https://arxiv.org/abs/2209.10655) 是由Xuezhe Ma、Chunting Zhou、Xiang Kong、Junxian He、Liangke Gui、Graham Neubig、Jonathan May和Luke Zettlemoyer提出的MEGA模型。MEGA提出了一种新的自注意力方法，在每个编码器层中除了标准点积注意力的单个头之外，还引入了一个多头指数移动平均，使注意机制具有更强的位置偏置。这使得MEGA在标准基准测试中可以与Transformer竞争，并且具有显著较少的参数。MEGA的计算效率使得它能够扩展到非常长的序列，使其成为长文本NLP任务的一个有吸引力的选择。

论文的摘要如下：

*Transformer注意力机制的设计选择，包括弱归纳偏差和二次计算复杂度，限制了它在建模长序列方面的应用。在本文中，我们引入了Mega，这是一种简单、理论上有依据的、带有（指数）移动平均的单头门控注意力机制，用于将位置感知的局部依赖的归纳偏差纳入位置不可知的注意机制。我们进一步提出了Mega的一个变体，它在仅产生微小质量损失的情况下，通过将整个序列高效地切分为具有固定长度的多个块，实现了线性时间和空间复杂度。在包括长序列建模竞技场、神经机器翻译、自回归语言建模以及图像和语音分类在内的广泛序列建模基准测试上的大量实验表明，Mega比其他序列模型取得了显著的改进，包括Transformer的变体和最近的状态空间模型。*

提示：

- MEGA可以在参数相对较少的情况下表现出色。请参阅MEGA论文的附录D，其中列举了在不同环境中表现良好的架构规格示例。如果将MEGA用作解码器，请务必设置`bidirectional=False`，以避免与默认的双向错误。
- Mega-chunk是MEGA的一个变体，将时间和空间复杂度从二次降低到线性。使用MegaConfig.use_chunking进行切块，并使用MegaConfig.chunk_size来控制块大小。

该模型由[mnaylor](https://huggingface.co/mnaylor)贡献。
原始代码可以在[此处](https://github.com/facebookresearch/mega)找到。

实现说明：

- MEGA的原始实现在对填充和因果自注意力的注意力掩码的期望上存在不一致性。此实现解决了这个问题。
- 原始实现不包括token类型嵌入；此实现通过MegaConfig.add_token_type_embeddings选项来支持这些功能。

## MegaConfig

[[autodoc]] MegaConfig

## MegaModel

[[autodoc]] MegaModel
    - forward

## MegaForCausalLM

[[autodoc]] MegaForCausalLM
    - forward

## MegaForMaskedLM

[[autodoc]] MegaForMaskedLM
    - forward

## MegaForSequenceClassification

[[autodoc]] MegaForSequenceClassification
    - forward

## MegaForMultipleChoice

[[autodoc]] MegaForMultipleChoice
    - forward

## MegaForTokenClassification

[[autodoc]] MegaForTokenClassification
    - forward

## MegaForQuestionAnswering

[[autodoc]] MegaForQuestionAnswering
    - forward
-->