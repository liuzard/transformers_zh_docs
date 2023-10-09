<! --版权2020年的拥抱面团团队。版权所有。

根据Apache许可证，版本2.0（“许可证”），您除非符合本许可证，否则不得使用此文件。您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按现状”分发的，
没有任何明示或暗示的保证或条件。有关特定语言的特定权限，请参阅许可证。

⚠️注意，此文件为markdown格式，但包含我们doc-builder的特定语法（类似于MDX），可能无法在您的markdown查看器中正确显示。-->

# MPNet

## 概述

MPNet模型是由Kaitao Song，Xu Tan，Tao Qin，Jianfeng Lu，Tie-Yan Liu在[MPNet：用于语言理解的掩蔽和置换预训练](https://arxiv.org/abs/2004.09297)中提出的。

MPNet采用了一种新颖的预训练方法，称为掩蔽和置换语言建模，以继承掩蔽语言建模和置换语言建模的优点，用于自然语言理解。

论文中的摘要如下：

* BERT采用了掩蔽语言建模（MLM）进行预训练，是最成功的预训练模型之一。由于BERT忽略了预测标记之间的依赖关系，XLNet引入了置换语言建模（PLM）进行预训练以解决此问题。然而，XLNet未利用句子的完整位置信息，因此在预训练和微调之间存在位置差异。在本文中，我们提出了MPNet，一种新颖的预训练方法，继承了BERT和XLNet的优点，并避免了它们的限制。MPNet通过置换语言建模（而不是BERT中的MLM）利用了预测标记之间的依赖关系，并将辅助位置信息作为输入，使模型看到完整的句子，从而减少了位置差异（而不是XLNet中的PLM）。我们在一个大规模数据集上对MPNet进行预训练（超过160GB的文本语料库），并在各种下游任务（GLUE，SQuAD等）上进行微调。实验结果表明，MPNet在性能上大大优于MLM和PLM，并在相同的模型设置下相比之前最先进的预训练方法（如BERT，XLNet，RoBERTa）在这些任务中取得了更好的结果。*

提示：

- MPNet没有`token_type_ids`，您不需要指示哪个标记属于哪个片段。只需使用分隔标记`tokenizer.sep_token`（或`[sep]`）分隔段落即可。

原始代码可在[此处](https://github.com/microsoft/MPNet)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问题回答任务指南](../tasks/question_answering)
- [掩蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## MPNetConfig

[[autodoc]] MPNetConfig

## MPNetTokenizer

[[autodoc]] MPNetTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## MPNetTokenizerFast

[[autodoc]] MPNetTokenizerFast

## MPNetModel

[[autodoc]] MPNetModel
    - forward

## MPNetForMaskedLM

[[autodoc]] MPNetForMaskedLM
    - forward

## MPNetForSequenceClassification

[[autodoc]] MPNetForSequenceClassification
    - forward

## MPNetForMultipleChoice

[[autodoc]] MPNetForMultipleChoice
    - forward

## MPNetForTokenClassification

[[autodoc]] MPNetForTokenClassification
    - forward

## MPNetForQuestionAnswering

[[autodoc]] MPNetForQuestionAnswering
    - forward

## TFMPNetModel

[[autodoc]] TFMPNetModel
    - call

## TFMPNetForMaskedLM

[[autodoc]] TFMPNetForMaskedLM
    - call

## TFMPNetForSequenceClassification

[[autodoc]] TFMPNetForSequenceClassification
    - call

## TFMPNetForMultipleChoice

[[autodoc]] TFMPNetForMultipleChoice
    - call

## TFMPNetForTokenClassification

[[autodoc]] TFMPNetForTokenClassification
    - call

## TFMPNetForQuestionAnswering

[[autodoc]] TFMPNetForQuestionAnswering
    - call