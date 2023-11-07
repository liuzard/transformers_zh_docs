<!--版权所有2020 The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）使用本文件，除非符合以下许可证的规定，否则不得使用此文件：
http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是按"原样"分发的，没有任何明示或暗示的保证或条件。
请参阅许可证以了解许可下特定的语言和限制。-->

# ALBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=albert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-albert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/albert-base-v2">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

ALBERT模型由Zhenzhong Lan、Mingda Chen、Sebastian Goodman、Kevin Gimpel、Piyush Sharma和Radu Soricut在[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)中提出。它提出了两种参数减少技术，以降低BERT的内存消耗和提高训练速度：

- 将嵌入矩阵拆分为两个较小的矩阵。
- 使用在组之间分割的重复层。

论文摘要如下：

*当预训练自然语言表示模型时，增加模型大小通常会改善下游任务的性能。然而，由于GPU/TPU内存限制、训练时间过长和意外的模型退化，进一步增加模型规模变得更加困难。为了解决这些问题，我们提出了两种参数减少技术，以降低BERT的内存消耗和提高训练速度。全面的经验证据表明，我们提出的方法使模型相对于原始的BERT更好地扩展。我们还使用了一种自监督损失，重点关注建模句间连贯性，并表明它在具有多句输入的下游任务中始终对其有所帮助。因此，我们的最佳模型在GLUE、RACE和SQuAD基准上建立了新的最佳结果，并且参数较BERT-large少。*

提示：

- ALBERT是一个带有绝对位置嵌入的模型，因此通常建议在右侧填充输入，而不是左侧。
- ALBERT使用重复层，导致内存占用较小，但计算成本与具有相同数量隐藏层的BERT-like架构相似，因为它必须遍历相同数量的（重复的）层。
- 嵌入大小E与隐藏大小H不同，这是有理由的，因为嵌入是上下文无关的（一个嵌入向量表示一个标记），而隐藏状态是上下文相关的（一个隐藏状态表示一个标记序列），因此逻辑上应该有H >> E。此外，嵌入矩阵是大型的，因为它是V x E（V是词汇大小）。如果E < H，则它有更少的参数。
- 层被分组分割的，共享参数（以节省内存）。
下一个句子预测被一个句子排序预测取代：在输入中，我们有两个连续的句子A和B，我们可以按照A后跟B或B后跟A的顺序输入。模型必须预测它们是否被交换。

此模型由[lysandre](https://huggingface.co/lysandre)贡献。此模型的jax版本由[kamalkraj](https://huggingface.co/kamalkraj)贡献。原始代码可以在[这里](https://github.com/google-research/ALBERT)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## AlbertConfig

[[autodoc]] AlbertConfig

## AlbertTokenizer

[[autodoc]] AlbertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## AlbertTokenizerFast

[[autodoc]] AlbertTokenizerFast

## Albert特定的输出

[[autodoc]] models.albert.modeling_albert.AlbertForPreTrainingOutput

[[autodoc]] models.albert.modeling_tf_albert.TFAlbertForPreTrainingOutput

## AlbertModel

[[autodoc]] AlbertModel
    - forward

## AlbertForPreTraining

[[autodoc]] AlbertForPreTraining
    - forward

## AlbertForMaskedLM

[[autodoc]] AlbertForMaskedLM
    - forward

## AlbertForSequenceClassification

[[autodoc]] AlbertForSequenceClassification
    - forward

## AlbertForMultipleChoice

[[autodoc]] AlbertForMultipleChoice

## AlbertForTokenClassification

[[autodoc]] AlbertForTokenClassification
    - forward

## AlbertForQuestionAnswering

[[autodoc]] AlbertForQuestionAnswering
    - forward

## TFAlbertModel

[[autodoc]] TFAlbertModel
    - call

## TFAlbertForPreTraining

[[autodoc]] TFAlbertForPreTraining
    - call

## TFAlbertForMaskedLM

[[autodoc]] TFAlbertForMaskedLM
    - call

## TFAlbertForSequenceClassification

[[autodoc]] TFAlbertForSequenceClassification
    - call

## TFAlbertForMultipleChoice

[[autodoc]] TFAlbertForMultipleChoice
    - call

## TFAlbertForTokenClassification

[[autodoc]] TFAlbertForTokenClassification
    - call

## TFAlbertForQuestionAnswering

[[autodoc]] TFAlbertForQuestionAnswering
    - call

## FlaxAlbertModel

[[autodoc]] FlaxAlbertModel
    - __call__

## FlaxAlbertForPreTraining

[[autodoc]] FlaxAlbertForPreTraining
    - __call__

## FlaxAlbertForMaskedLM

[[autodoc]] FlaxAlbertForMaskedLM
    - __call__

## FlaxAlbertForSequenceClassification

[[autodoc]] FlaxAlbertForSequenceClassification
    - __call__

## FlaxAlbertForMultipleChoice

[[autodoc]] FlaxAlbertForMultipleChoice
    - __call__

## FlaxAlbertForTokenClassification

[[autodoc]] FlaxAlbertForTokenClassification
    - __call__

## FlaxAlbertForQuestionAnswering

[[autodoc]] FlaxAlbertForQuestionAnswering
    - __call__