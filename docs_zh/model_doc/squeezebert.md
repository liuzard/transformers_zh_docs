<!--版权2020年The HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”），你不得使用此文件，除非符合许可证的规定。你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样” BASIS，不附带任何形式的明示或暗示的担保或条件。请参阅许可证以了解许可证下的具体权限和限制。

⚠️请注意，此文件保存在Markdown中，但包含我们的文档生成器的特定语法（类似于MDX），可能无法在你的Markdown查看器中正确渲染。

-->

# SqueezeBERT

## 概述

SqueezeBERT模型是由Forrest N. Iandola、Albert E. Shaw、Ravi Krishna、Kurt W. Keutzer在论文[SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316)中提出的。它是一个双向变换器，类似于BERT模型。BERT架构和SqueezeBERT架构之间的关键区别在于SqueezeBERT使用[分组卷积](https://blog.yani.io/filter-group-tutorial)而不是全连接层来进行Q、K、V和FFN层的操作。

论文中的摘要如下：

*人类每天阅读和编写数千亿条消息。此外，由于大规模数据集、大型计算系统和更好的神经网络模型的可用性，自然语言处理（NLP）技术在理解、校对和组织这些消息方面取得了重大进展。因此，将NLP应用于各种应用程序以帮助网络用户、社交网络和企业具有重大机会。特别是，我们认为智能手机和其他移动设备是大规模部署NLP模型的关键平台。然而，当今高精度NLP神经网络模型（例如BERT和RoBERTa）的计算成本非常高，BERT-base在Pixel 3智能手机上对一个文本片段进行分类需要1.7秒。在这项工作中，我们观察到，分组卷积等方法已经显著加速了计算机视觉网络，但这些技术中的许多技术尚未被NLP神经网络设计师采用。我们展示了如何使用分组卷积替换自注意力层中的若干操作，并在名为SqueezeBERT的新型网络架构中使用这种技术，在Pixel 3上运行的速度比BERT-base快4.3倍，同时在GLUE测试集上达到竞争力的准确性。SqueezeBERT代码将被发布。*

提示：

- SqueezeBERT是一个具有绝对位置嵌入的模型，因此通常建议在输入的右侧进行填充，而不是左侧。
- SqueezeBERT类似于BERT，因此依赖于遮蔽语言建模（MLM）目标。因此，它在预测遮蔽token和NLU方面效率很高，但对于文本生成来说并不是最佳选择。在这方面，使用因果语言建模（CLM）目标训练的模型更好。
- 在微调序列分类任务时，建议从*squeezebert/squeezebert-mnli-headless*检查点开始以获得最佳结果。

该模型由[forresti](https://huggingface.co/forresti)贡献。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [遮蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多选任务指南](../tasks/multiple_choice)

## SqueezeBertConfig

[[autodoc]] SqueezeBertConfig

## SqueezeBertTokenizer

[[autodoc]] SqueezeBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SqueezeBertTokenizerFast

[[autodoc]] SqueezeBertTokenizerFast

## SqueezeBertModel

[[autodoc]] SqueezeBertModel

## SqueezeBertForMaskedLM

[[autodoc]] SqueezeBertForMaskedLM

## SqueezeBertForSequenceClassification

[[autodoc]] SqueezeBertForSequenceClassification

## SqueezeBertForMultipleChoice

[[autodoc]] SqueezeBertForMultipleChoice

## SqueezeBertForTokenClassification

[[autodoc]] SqueezeBertForTokenClassification

## SqueezeBertForQuestionAnswering

[[autodoc]] SqueezeBertForQuestionAnswering