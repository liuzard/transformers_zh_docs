<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证的规定，
否则您不得使用此文件。您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”的基础分发的，
没有任何明示或暗示的保证或条件。有关许可证下的特定语言的权限和限制，请参阅许可证。

⚠️ 请注意，此文件采用Markdown格式，但包含我们的doc-builder的特定语法（类似于MDX），
这可能在您的Markdown查看器中无法正确呈现。
-->

# ELECTRA

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=electra">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-electra-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/electra_large_discriminator_squad2_512">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

ELECTRA模型是在论文 [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than
Generators](https://openreview.net/pdf?id=r1xMH1BtvB) 中提出的。ELECTRA是一种新的预训练方法，它训练两个transformer模型：生成器和鉴别器。生成器的作用是替换序列中的标记，因此它被训练为一个掩码语言模型。鉴别器是我们感兴趣的模型，它试图识别生成器在序列中替换的哪些标记。

论文中的摘要如下：

*掩码语言建模（MLM）预训练方法（如BERT）通过用[MASK]替换一些标记来破坏输入，然后训练模型以重构原始标记。尽管当转移到下游NLP任务时它们产生了良好的结果，但它们通常需要大量计算才能发挥作用。作为替代方案，我们提出了一种更加样本高效的预训练任务，称为替代标记检测。我们的方法不是掩码输入，而是通过用从小型生成器网络中采样的可行替代品替换一些标记来破坏输入。然后，我们训练一个鉴别模型，该模型预测破坏输入中的每个标记是由生成器样本替换的还是没有替换。详细实验证明，与MLM相比，这个新的预训练任务更加高效，因为任务是定义在所有输入标记上，而不仅仅是被遮蔽掉的小子集上。因此，我们的方法学习的上下文表示在模型大小，数据和计算资源相同的情况下显着优于BERT学习的表示。这种增益在小模型中特别明显；例如，我们在一个GPU上训练了4天的模型，在GLUE自然语言理解基准测试中的性能优于使用30倍计算资源训练的GPT模型。我们的方法在大规模情况下表现良好，当使用相同数量的计算资源时，它的性能与RoBERTa和XLNet相当，但它们的计算资源使用量只有它们的1/4。*

提示：

- ELECTRA是预训练方法，因此对底层模型BERT几乎没有进行任何更改。唯一的变化是嵌入大小和隐藏大小的分离：嵌入大小通常较小，而隐藏大小较大。使用额外的投影层（线性层）将嵌入从嵌入大小投影到隐藏大小。在嵌入大小与隐藏大小相同时，不使用投影层。
- ELECTRA是使用另一个（小型）掩码语言模型进行预训练的transformer模型。输入由该语言模型破坏，该语言模型接受随机遮蔽的输入文本，并输出一个文本，其中ELECTRA必须预测哪个标记是原始标记，哪个标记已被替换。与GAN训练类似，小型语言模型进行了几步训练（但目标是原始文本，而不是像传统GAN设置中的欺骗ELECTRA模型）。然后，进行了几步训练ELECTRA模型。
- 使用[Google Research的实现](https://github.com/google-research/electra)保存的ELECTRA检查点同时包含生成器和鉴别器。转换脚本要求用户命名要导出到正确架构的模型。转换为HuggingFace格式后，这些检查点可以加载到所有可用的ELECTRA模型中。这意味着鉴别器可加载到[`ElectraForMaskedLM`]模型中，生成器可加载到[`ElectraForPreTraining`]模型中（分类头将被随机初始化，因为它在生成器中不存在）。

该模型由[lysandre](https://huggingface.co/lysandre)贡献。原始代码可在[此处](https://github.com/google-research/electra)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../任务/masked_language_modeling)
- [多项选择任务指南](../任务/multiple_choice)

## ElectraConfig

[[autodoc]] ElectraConfig

## ElectraTokenizer

[[autodoc]] ElectraTokenizer

## ElectraTokenizerFast

[[autodoc]] ElectraTokenizerFast

## Electra特定输出

[[autodoc]] models.electra.modeling_electra.ElectraForPreTrainingOutput

[[autodoc]] models.electra.modeling_tf_electra.TFElectraForPreTrainingOutput

## ElectraModel

[[autodoc]] ElectraModel
    - forward

## ElectraForPreTraining

[[autodoc]] ElectraForPreTraining
    - forward

## ElectraForCausalLM

[[autodoc]] ElectraForCausalLM
    - forward

## ElectraForMaskedLM

[[autodoc]] ElectraForMaskedLM
    - forward

## ElectraForSequenceClassification

[[autodoc]] ElectraForSequenceClassification
    - forward

## ElectraForMultipleChoice

[[autodoc]] ElectraForMultipleChoice
    - forward

## ElectraForTokenClassification

[[autodoc]] ElectraForTokenClassification
    - forward

## ElectraForQuestionAnswering

[[autodoc]] ElectraForQuestionAnswering
    - forward

## TFElectraModel

[[autodoc]] TFElectraModel
    - call

## TFElectraForPreTraining

[[autodoc]] TFElectraForPreTraining
    - call

## TFElectraForMaskedLM

[[autodoc]] TFElectraForMaskedLM
    - call

## TFElectraForSequenceClassification

[[autodoc]] TFElectraForSequenceClassification
    - call

## TFElectraForMultipleChoice

[[autodoc]] TFElectraForMultipleChoice
    - call

## TFElectraForTokenClassification

[[autodoc]] TFElectraForTokenClassification
    - call

## TFElectraForQuestionAnswering

[[autodoc]] TFElectraForQuestionAnswering
    - call

## FlaxElectraModel

[[autodoc]] FlaxElectraModel
    - __call__

## FlaxElectraForPreTraining

[[autodoc]] FlaxElectraForPreTraining
    - __call__

## FlaxElectraForCausalLM

[[autodoc]] FlaxElectraForCausalLM
    - __call__

## FlaxElectraForMaskedLM

[[autodoc]] FlaxElectraForMaskedLM
    - __call__

## FlaxElectraForSequenceClassification

[[autodoc]] FlaxElectraForSequenceClassification
    - __call__

## FlaxElectraForMultipleChoice

[[autodoc]] FlaxElectraForMultipleChoice
    - __call__

## FlaxElectraForTokenClassification

[[autodoc]] FlaxElectraForTokenClassification
    - __call__

## FlaxElectraForQuestionAnswering

[[autodoc]] FlaxElectraForQuestionAnswering
    - __call__