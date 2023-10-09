<!--版权所有2022 HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（"许可证"）进行许可；除非符合许可证，否则不得使用此文件。你可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本

除非适用法律要求或书面同意，否则根据许可证分发的软件系根据"原样"分发的软件，无论是明示或暗示的，不附带任何保证或条件。有关

特定语言的保证或条件的限制，请参见许可证。

⚠️请注意，此文件是Markdown格式的，但包含我们doc-builder的特定语法（类似于MDX），你的Markdown查看器可能无法正确呈现。

-->

# RoCBert

## 概述

RoCBert模型是由HuiSu、WeiweiShi、XiaoyuShen、XiaoZhou、TuoJi和JiaruiFang等人在论文[RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining](https://aclanthology.org/2022.acl-long.65.pdf)中提出的。它是一个针对各种形式的对抗攻击鲁棒的预训练中文语言模型。

论文的摘要如下：

*大规模预训练语言模型在自然语言处理任务中取得了先进的结果。然而，它们对对抗攻击特别是对于像中文这样的文字语言特别容易受攻击。本文提出了RoCBert：一种经过对比性学习目标预训练的中文Bert，具有抵御各种形式的对抗攻击（如诱骗攻击、同义词攻击、错别字攻击等）的能力。RoCBert以语义、音标和视觉特征作为输入信息。我们展示了所有这些特征对模型的鲁棒性都很重要，因为攻击可以在这三种形式下进行。在5个中文自然语言理解任务中，RoCBert在三种黑盒对抗算法下的表现优于强基准模型，并且在干净测试集上的性能没有损失。在人造攻击下，RoCBert在有害内容检测任务中表现最好。*

该模型由[weiweishi](https://huggingface.co/weiweishi)贡献。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [遮蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## RoCBert配置

[[autodoc]] RoCBertConfig
    - all


## RoCBert分词器

[[autodoc]] RoCBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## RoCBert模型

[[autodoc]] RoCBertModel
    - forward


## RoCBert用于预训练

[[autodoc]] RoCBertForPreTraining
    - forward


## RoCBert用于因果语言建模

[[autodoc]] RoCBertForCausalLM
    - forward


## RoCBert用于遮蔽语言建模

[[autodoc]] RoCBertForMaskedLM
    - forward


## RoCBert用于序列分类

[[autodoc]] transformers.RoCBertForSequenceClassification
    - forward

## RoCBert用于多项选择

[[autodoc]] transformers.RoCBertForMultipleChoice
    - forward


## RoCBert用于标记分类

[[autodoc]] transformers.RoCBertForTokenClassification
    - forward


## RoCBert用于问答

[[autodoc]] RoCBertForQuestionAnswering
    - forward