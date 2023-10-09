<!--版权 2021 The HuggingFace Team. 保留所有权利。

根据Apache许可证2.0版（“许可证”），除非符合许可证的规定，否则你不得使用此文件。
你可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以及根据许可证，以“现有基础”方式分发的软件，无论是明示或暗示的，都不附带任何形式的担保或条件。有关许可证下的特定语言的详细信息，请参阅许可证。

⚠️请注意，此文件是Markdown格式的，但包含特定的语法，用于我们的doc构建器（类似于MDX），这可能不会在你的Markdown查看器中正确显示。-->

# BigBird

## 概述

BigBird模型是由Zaheer，Manzil和Guruganesh，Guru和Dubey，Kumar Avinava和Ainslie，Joshua和Alberti，Chris和Ontanon，Santiago和Pham，Philip和Ravula，Anirudh和Wang，Qifan和Yang，Li等人在[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)提出的。BigBird是一种基于稀疏注意力的Transformer模型，它扩展了基于Transformer的模型，如BERT，以处理更长的序列。除了稀疏注意力外，BigBird还对输入序列应用全局注意力和随机注意力。从理论上讲，已经证明了应用稀疏、全局和随机注意力可以近似完全注意力，同时在处理更长序列时具有更高的计算效率。由于处理更长上下文的能力，BigBird在各种长文本NLP任务（如问答和摘要）中相对于BERT或RoBERTa表现出改进的性能。

论文中的摘要如下所示：

*基于Transformer的模型（如BERT）是NLP领域最成功的深度学习模型之一。然而，它们的一个核心限制是由于其完全注意机制使其对序列长度有二次依赖关系（主要是在内存方面）。为了解决这个问题，我们提出了一种稀疏注意机制BigBird将这个二次依赖关系减少为线性关系。我们展示了BigBird是序列函数的通用近似器，并且是图灵完备的，从而保留了二次依赖完全注意模型的这些性质。在这一过程中，我们的理论分析揭示了具有O(1)全局令牌（如CLS），这些令牌作为稀疏注意机制的一部分关注整个序列的好处。所提出的稀疏注意机制可以处理比以前相似硬件上可能的长度多8倍的序列。由于处理更长上下文的能力，BigBird在各种NLP任务（如问答和摘要）中显著提升了性能。我们还提出了基因组数据的新应用。*

提示：

- 关于BigBird注意力机制的详细解释，请参阅[此博客文章](https://huggingface.co/blog/big-bird)。
- BigBird有两种实现：**original_full**和**block_sparse**。对于长度小于1024的序列，建议使用**original_full**，因为使用**block_sparse**注意力没有好处。
- 代码目前使用3个块的窗口大小和2个全局块。
- 序列长度必须可被块大小整除。
- 当前实现仅支持**ITC**。
- 当前实现不支持**num_random_blocks = 0**。
- BigBird是一个具有绝对位置嵌入的模型，因此通常建议将输入填充在右侧而不是左侧。

该模型由[vasudevgupta](https://huggingface.co/vasudevgupta)贡献。原始代码可以在[此处](https://github.com/google-research/bigbird)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [遮蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多选任务指南](../tasks/multiple_choice)

## BigBirdConfig

[[autodoc]] BigBirdConfig

## BigBirdTokenizer

[[autodoc]] BigBirdTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## BigBirdTokenizerFast

[[autodoc]] BigBirdTokenizerFast

## BigBird特定的输出

[[autodoc]] models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput

## BigBirdModel

[[autodoc]] BigBirdModel
    - forward

## BigBirdForPreTraining

[[autodoc]] BigBirdForPreTraining
    - forward

## BigBirdForCausalLM

[[autodoc]] BigBirdForCausalLM
    - forward

## BigBirdForMaskedLM

[[autodoc]] BigBirdForMaskedLM
    - forward

## BigBirdForSequenceClassification

[[autodoc]] BigBirdForSequenceClassification
    - forward

## BigBirdForMultipleChoice

[[autodoc]] BigBirdForMultipleChoice
    - forward

## BigBirdForTokenClassification

[[autodoc]] BigBirdForTokenClassification
    - forward

## BigBirdForQuestionAnswering

[[autodoc]] BigBirdForQuestionAnswering
    - forward

## FlaxBigBirdModel

[[autodoc]] FlaxBigBirdModel
    - __call__

## FlaxBigBirdForPreTraining

[[autodoc]] FlaxBigBirdForPreTraining
    - __call__

## FlaxBigBirdForCausalLM

[[autodoc]] FlaxBigBirdForCausalLM
    - __call__

## FlaxBigBirdForMaskedLM

[[autodoc]] FlaxBigBirdForMaskedLM
    - __call__

## FlaxBigBirdForSequenceClassification

[[autodoc]] FlaxBigBirdForSequenceClassification
    - __call__

## FlaxBigBirdForMultipleChoice

[[autodoc]] FlaxBigBirdForMultipleChoice
    - __call__

## FlaxBigBirdForTokenClassification

[[autodoc]] FlaxBigBirdForTokenClassification
    - __call__

## FlaxBigBirdForQuestionAnswering

[[autodoc]] FlaxBigBirdForQuestionAnswering
    - __call__