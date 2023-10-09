<!--版权所有 2022 年 The HuggingFace 团队保留所有权利。

根据 Apache 许可证 2.0 版（以下简称“许可证”），您可能不得使用本文件，除非其符合许可证的要求。
您可以在以下网址获取该许可证的副本:

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按实际情况提供的，并不提供任何明示或暗示的保证或条件。
有关特定语言版本的明示或暗示的保证或条件，请参阅许可证。

⚠️ 请注意，此文件采用 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，
可能无法在您的 Markdown 查看器中正确显示。

-->

# Nezha

## 概述

Nezha 模型由魏军秋等人在 [NEZHA: NLP 任务中的中文神经上下文表示](https://arxiv.org/abs/1909.00204) 中提出。

论文摘要如下：

*由于预训练语言模型通过在大规模语料库上进行预训练来捕捉文本中的深度上下文信息，因此它在各种自然语言理解（NLU）任务中取得了巨大的成功。
在本技术报告中，我们介绍了我们对中文语料库进行预训练语言模型 NEZHA（中文神经上下文表示）的实践，并对中文 NLU 任务进行了微调。
NEZHA 的当前版本基于 BERT，并具有一系列经过验证的改进，其中包括作为有效的位置编码方案的功能相对位置编码、整词蒙版策略、混合精度训练和 LAMB 优化器进行模型训练。
实验结果表明，NEZHA 在几个代表性的中文任务（包括实体命名识别（人民日报命名实体识别）、句子匹配（LCQMC）、中文情感分类（ChnSenti）和自然语言推理（XNLI））进行微调后，取得了最先进的性能。*

该模型由 [sijunhe](https://huggingface.co/sijunhe) 贡献。原始代码可在 [此处](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch) 找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## NezhaConfig

[[autodoc]] NezhaConfig

## NezhaModel

[[autodoc]] NezhaModel
    - forward

## NezhaForPreTraining

[[autodoc]] NezhaForPreTraining
    - forward

## NezhaForMaskedLM

[[autodoc]] NezhaForMaskedLM
    - forward

## NezhaForNextSentencePrediction

[[autodoc]] NezhaForNextSentencePrediction
    - forward

## NezhaForSequenceClassification

[[autodoc]] NezhaForSequenceClassification
    - forward

## NezhaForMultipleChoice

[[autodoc]] NezhaForMultipleChoice
    - forward

## NezhaForTokenClassification

[[autodoc]] NezhaForTokenClassification
    - forward

## NezhaForQuestionAnswering

[[autodoc]] NezhaForQuestionAnswering
    - forward