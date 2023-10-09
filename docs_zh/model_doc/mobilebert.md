<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）授权；除非符合许可证，否则不得使用此文件。
你可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证中规定的条件分发的软件是基于“按原样”提供的，不附带任何明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言和限制。

️请注意，此文件是Markdown格式，但包含特定于我们的doc-builder的语法（类似MDX），可能无法正确在你的Markdown查看器中呈现。-->

# MobileBERT

## 概述

MobileBERT模型是由Zhiqing Sun，Hongkun Yu，Xiaodan Song，Renjie Liu，Yiming Yang和Denny Zhou在文章[MobileBERT：一种用于资源受限设备的紧凑任务不可知BERT](https://arxiv.org/abs/2004.02984)中提出的。它是基于BERT模型的双向变换器，使用了多种方法进行压缩和加速。

该论文的摘要如下：

*最近通过使用具有数亿个参数的巨型预训练模型使自然语言处理（NLP）取得了巨大成功。然而，这些模型受到庞大的模型大小和高延迟的困扰，导致它们无法部署到资源受限的移动设备上。在本文中，我们提出了MobileBERT来压缩和加速流行的BERT模型。像原始的BERT一样，MobileBERT是任务不可知的，也就是说，它可以通过简单的微调应用于各种下游NLP任务。基本上，MobileBERT是BERT_LARGE的一个精简版本，同时配备了瓶颈结构和经过精心设计的自我注意力和前馈网络之间的平衡。为了训练MobileBERT，我们首先训练了一个特别设计的教师模型，即一个包含倒置瓶颈的BERT_LARGE模型。然后，我们从这个教师模型转移知识到MobileBERT。实证研究表明，MobileBERT比BERT_BASE小4.3倍，比BERT_BASE快5.5倍，同时在众所周知的基准上取得了有竞争力的结果。在GLUE的自然语言推理任务中，MobileBERT的得分为77.7（比BERT_BASE低0.6分），在Pixel 4手机上的延迟为62毫秒。在SQuAD v1.1/v2.0问题回答任务中，MobileBERT的开发F1得分为90.0/79.2（比BERT_BASE高1.5/2.1）。*

提示：

- MobileBERT模型具有绝对位置嵌入，因此通常建议在右侧对输入进行填充，而不是左侧。
- MobileBERT与BERT类似，因此依赖于遮罩语言建模（MLM）目标。因此，它在预测遮罩标记和NLU方面效率高，但对于文本生成而言并不理想。以因果语言建模（CLM）目标进行训练的模型在这方面更好。

此模型由[vshampor](https://huggingface.co/vshampor)贡献。原始代码可以在[这里](https://github.com/google-research/mobilebert)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问题回答任务指南](../tasks/question_answering)
- [遮罩语言模型任务指南](../tasks/masked_language_modeling)
- [多选任务指南](../tasks/multiple_choice)

## MobileBertConfig

[[autodoc]] MobileBertConfig

## MobileBertTokenizer

[[autodoc]] MobileBertTokenizer

## MobileBertTokenizerFast

[[autodoc]] MobileBertTokenizerFast

## MobileBert特定输出

[[autodoc]] models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput

[[autodoc]] models.mobilebert.modeling_tf_mobilebert.TFMobileBertForPreTrainingOutput

## MobileBertModel

[[autodoc]] MobileBertModel
    - forward

## MobileBertForPreTraining

[[autodoc]] MobileBertForPreTraining
    - forward

## MobileBertForMaskedLM

[[autodoc]] MobileBertForMaskedLM
    - forward

## MobileBertForNextSentencePrediction

[[autodoc]] MobileBertForNextSentencePrediction
    - forward

## MobileBertForSequenceClassification

[[autodoc]] MobileBertForSequenceClassification
    - forward

## MobileBertForMultipleChoice

[[autodoc]] MobileBertForMultipleChoice
    - forward

## MobileBertForTokenClassification

[[autodoc]] MobileBertForTokenClassification
    - forward

## MobileBertForQuestionAnswering

[[autodoc]] MobileBertForQuestionAnswering
    - forward

## TFMobileBertModel

[[autodoc]] TFMobileBertModel
    - call

## TFMobileBertForPreTraining

[[autodoc]] TFMobileBertForPreTraining
    - call

## TFMobileBertForMaskedLM

[[autodoc]] TFMobileBertForMaskedLM
    - call

## TFMobileBertForNextSentencePrediction

[[autodoc]] TFMobileBertForNextSentencePrediction
    - call

## TFMobileBertForSequenceClassification

[[autodoc]] TFMobileBertForSequenceClassification
    - call

## TFMobileBertForMultipleChoice

[[autodoc]] TFMobileBertForMultipleChoice
    - call

## TFMobileBertForTokenClassification

[[autodoc]] TFMobileBertForTokenClassification
    - call

## TFMobileBertForQuestionAnswering

[[autodoc]] TFMobileBertForQuestionAnswering
    - call