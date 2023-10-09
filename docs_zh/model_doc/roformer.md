<!--版权所有2021年HuggingFace团队。保留所有权利。

根据Apache许可证，版本2.0（“许可证”），除非符合许可证的规定，否则您不得使用此文件。
您可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”提供的，没有任何明示或暗示的保证或条件。
请注意，此文件是Markdown格式，但包含我们的文档构建器（类似于MDX）的特定语法，可能无法在Markdown查看器中正确呈现。

-->

# RoFormer

## 概述

RoFormer模型是由Jianlin Su、Yu Lu、Shengfeng Pan、Bo Wen和Yunfeng Liu在论文[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864v1.pdf)中提出的。

论文摘要如下：

*Transformer架构中的位置编码为序列中不同位置的元素之间的依赖建模提供了监督。我们研究了在基于Transformer的语言模型中编码位置信息的各种方法，并提出了一种名为RoPE（Rotary Position Embedding）的新实现。所提出的RoPE利用旋转矩阵对绝对位置信息进行编码，并在自注意力公式中自然地融入了明确的相对位置依赖性。值得注意的是，RoPE具有诸如能够扩展到任意序列长度、随着相对距离的增加而衰减的标记间依赖、以及能够装备线性自注意力与相对位置编码的能力等宝贵特性。因此，配备了旋转位置嵌入（RoPE）的增强Transformer模型，即RoFormer，在具有长文本的任务中实现了出色的性能。我们公开了关于中文数据的理论分析以及一些初步实验结果。即将更新英文基准测试的进行中实验。*

提示：

- RoFormer是一种类似BERT的自编码模型，采用了旋转位置嵌入。旋转位置嵌入在具有长文本的分类任务中显示出改进的性能。

该模型由[junnyu](https://huggingface.co/junnyu)贡献。原始代码可在[此处](https://github.com/ZhuiyiTechnology/roformer)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [遮盖语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## RoFormerConfig

[[autodoc]] RoFormerConfig

## RoFormerTokenizer

[[autodoc]] RoFormerTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RoFormerTokenizerFast

[[autodoc]] RoFormerTokenizerFast
    - build_inputs_with_special_tokens

## RoFormerModel

[[autodoc]] RoFormerModel
    - forward

## RoFormerForCausalLM

[[autodoc]] RoFormerForCausalLM
    - forward

## RoFormerForMaskedLM

[[autodoc]] RoFormerForMaskedLM
    - forward

## RoFormerForSequenceClassification

[[autodoc]] RoFormerForSequenceClassification
    - forward

## RoFormerForMultipleChoice

[[autodoc]] RoFormerForMultipleChoice
    - forward

## RoFormerForTokenClassification

[[autodoc]] RoFormerForTokenClassification
    - forward

## RoFormerForQuestionAnswering

[[autodoc]] RoFormerForQuestionAnswering
    - forward

## TFRoFormerModel

[[autodoc]] TFRoFormerModel
    - call

## TFRoFormerForMaskedLM

[[autodoc]] TFRoFormerForMaskedLM
    - call

## TFRoFormerForCausalLM

[[autodoc]] TFRoFormerForCausalLM
    - call

## TFRoFormerForSequenceClassification

[[autodoc]] TFRoFormerForSequenceClassification
    - call

## TFRoFormerForMultipleChoice

[[autodoc]] TFRoFormerForMultipleChoice
    - call

## TFRoFormerForTokenClassification

[[autodoc]] TFRoFormerForTokenClassification
    - call

## TFRoFormerForQuestionAnswering

[[autodoc]] TFRoFormerForQuestionAnswering
    - call

## FlaxRoFormerModel

[[autodoc]] FlaxRoFormerModel
    - __call__

## FlaxRoFormerForMaskedLM

[[autodoc]] FlaxRoFormerForMaskedLM
    - __call__

## FlaxRoFormerForSequenceClassification

[[autodoc]] FlaxRoFormerForSequenceClassification
    - __call__

## FlaxRoFormerForMultipleChoice

[[autodoc]] FlaxRoFormerForMultipleChoice
    - __call__

## FlaxRoFormerForTokenClassification

[[autodoc]] FlaxRoFormerForTokenClassification
    - __call__

## FlaxRoFormerForQuestionAnswering

[[autodoc]] FlaxRoFormerForQuestionAnswering
    - __call__