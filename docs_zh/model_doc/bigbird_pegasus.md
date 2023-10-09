<!--版权所有2021年The HuggingFace团队。
根据Apache 2.0许可证（“许可证”）获得许可; 在符合许可证的情况下，你不能使用此文件。
你可以在以下链接中获得许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件都是基于“按原样”提供的，
不附带任何明示或暗示的保证或条件。有关特定语言的保证或条件，请参见许可证的规定。
请注意，该文件是以 Markdown 为基础的，但包含我们文档生成器的特定语法（类似于 MDX）
这些语法可能无法在你的 Markdown 查看器中正确呈现。
-->

# BigBirdPegasus

## 概述

BigBird模型在[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)一文中提出，
作者为Zaheer, Manzil和Guruganesh, Guru和Dubey, Kumar Avinava和Ainslie, Joshua和Alberti, Chris和Ontanon，
Santiago和Pham, Philip和Ravula, Anirudh和Wang, Qifan和Yang, Li等人。BigBird是基于稀疏注意力的变压器模型，
从BERT等基于Transformer的模型扩展到更长的序列。除了稀疏注意力外，BigBird还对输入序列应用全局注意力和随机注意力。
从理论上讲，已经证明了应用稀疏、全局和随机注意力可以近似全注意力，同时对于更长的序列在计算上更加高效。
由于处理更长上下文的能力，BigBird在各种长文本NLP任务（如问答和摘要）中表现出与BERT或RoBERTa相比的改进。

论文中的摘要如下：

*基于Transformers的模型，如BERT，是自然语言处理中最成功的深度学习模型之一。
不幸的是，它们的一个核心限制是对序列长度的二次依赖关系(主要是在内存方面)，这是由于它们的全注意力机制引起的。
为了解决这个问题，我们提出了BigBird，一种将这种二次依赖关系减小到线性的稀疏注意力机制。
我们展示了BigBird是序列函数的一个通用逼近器，并且是图灵完备的，从而保留了这种二次依赖、全注意力模型的这些属性。
在推导的过程中，我们的理论分析揭示了在稀疏注意力机制中有O（1）全局令牌（如CLS）的一些好处，
这些令牌作为稀疏注意力机制的一部分参与到整个序列中。
所提出的稀疏注意力可以处理长度多达之前使用相似硬件时的8倍的序列。
由于处理更长上下文的能力，BigBird在各种NLP任务(如问答和摘要)中显著提高了性能。
我们还提出了将其应用于基因组数据的新型应用。*

提示：

- 关于BigBird的注意力机制如何工作的详细解释，请参见[此博文](https://huggingface.co/blog/big-bird)。
- BigBird有两种实现方式：**original_full**和**block_sparse**。
  对于序列长度<1024，建议使用**original_full**，因为使用**block_sparse**注意力没有任何好处。
- 代码当前使用3个块和2个全局块的窗口大小。
- 序列长度必须是块大小的整数倍。
- 当前实现仅支持**ITC**。
- 当前实现不支持**num_random_blocks = 0**。
- BigBirdPegasus使用[PegasusTokenizer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pegasus/tokenization_pegasus.py)。
- BigBird是一个具有绝对位置嵌入的模型，所以通常建议在右侧而不是左侧填充输入。

原始代码可在[此处](https://github.com/google-research/bigbird)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## BigBirdPegasusConfig

[[autodoc]] BigBirdPegasusConfig
    - all

## BigBirdPegasusModel

[[autodoc]] BigBirdPegasusModel
    - forward

## BigBirdPegasusForConditionalGeneration

[[autodoc]] BigBirdPegasusForConditionalGeneration
    - forward

## BigBirdPegasusForSequenceClassification

[[autodoc]] BigBirdPegasusForSequenceClassification
    - forward

## BigBirdPegasusForQuestionAnswering

[[autodoc]] BigBirdPegasusForQuestionAnswering
    - forward

## BigBirdPegasusForCausalLM

[[autodoc]] BigBirdPegasusForCausalLM
    - forward