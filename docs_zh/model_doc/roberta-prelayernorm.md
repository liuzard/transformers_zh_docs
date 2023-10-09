<!--版权所有2022年HuggingFace团队，保留所有权利。

根据Apache许可证，版本2.0（“许可证”），您除非符合许可证，否则不得使用此文件。
您可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于
“按原样”基础分发，不附带任何明示或暗示的担保条件。请参阅许可证以
了解有关许可的特定语言和限制的详细信息。

⚠️ 请注意，此文件是Markdown格式的，但包含我们的doc-builder的特定语法（类似于MDX），在您的Markdown查看器中可能无法正确渲染。

-->

# RoBERTa-PreLayerNorm

## 概述

RoBERTa-PreLayerNorm模型是由Myle Ott、Sergey Edunov、Alexei Baevski、Angela Fan、Sam Gross、Nathan Ng、David Grangier、Michael Auli在[公正的序列建模快速、可扩展的工具包fairseq](https://arxiv.org/abs/1904.01038)中提出的。它与在[fairseq](https://fairseq.readthedocs.io/)中使用`--encoder-normalize-before`标志完全相同。

来自论文的摘要如下:

*fairseq是一个开源的序列建模工具包，允许研究人员和开发人员训练用于翻译、摘要、语言建模和其他文本生成任务的自定义模型。该工具包基于PyTorch，并支持在多个GPU和机器之间进行分布式训练。我们还支持在现代GPU上进行快速混合精度训练和推断。*

提示:

- 实施与[Roberta](roberta)相同，只是不是使用_Add_和_Norm_，而是使用_Norm_和_Add_。_Add_和_Norm_指的是[Attention Is All You Need](https://arxiv.org/abs/1706.03762)中描述的加法和层归一化。
- 这与在[fairseq](https://fairseq.readthedocs.io/)中使用`--encoder-normalize-before`标志完全相同。

此模型由[andreasmaden](https://huggingface.co/andreasmaden)贡献。
原始代码可以在[这里](https://github.com/princeton-nlp/DinkyTrain)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [遮蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## RobertaPreLayerNormConfig

[[autodoc]] RobertaPreLayerNormConfig

## RobertaPreLayerNormModel

[[autodoc]] RobertaPreLayerNormModel
    - forward

## RobertaPreLayerNormForCausalLM

[[autodoc]] RobertaPreLayerNormForCausalLM
    - forward

## RobertaPreLayerNormForMaskedLM

[[autodoc]] RobertaPreLayerNormForMaskedLM
    - forward

## RobertaPreLayerNormForSequenceClassification

[[autodoc]] RobertaPreLayerNormForSequenceClassification
    - forward

## RobertaPreLayerNormForMultipleChoice

[[autodoc]] RobertaPreLayerNormForMultipleChoice
    - forward

## RobertaPreLayerNormForTokenClassification

[[autodoc]] RobertaPreLayerNormForTokenClassification
    - forward

## RobertaPreLayerNormForQuestionAnswering

[[autodoc]] RobertaPreLayerNormForQuestionAnswering
    - forward

## TFRobertaPreLayerNormModel

[[autodoc]] TFRobertaPreLayerNormModel
    - call

## TFRobertaPreLayerNormForCausalLM

[[autodoc]] TFRobertaPreLayerNormForCausalLM
    - call

## TFRobertaPreLayerNormForMaskedLM

[[autodoc]] TFRobertaPreLayerNormForMaskedLM
    - call

## TFRobertaPreLayerNormForSequenceClassification

[[autodoc]] TFRobertaPreLayerNormForSequenceClassification
    - call

## TFRobertaPreLayerNormForMultipleChoice

[[autodoc]] TFRobertaPreLayerNormForMultipleChoice
    - call

## TFRobertaPreLayerNormForTokenClassification

[[autodoc]] TFRobertaPreLayerNormForTokenClassification
    - call

## TFRobertaPreLayerNormForQuestionAnswering

[[autodoc]] TFRobertaPreLayerNormForQuestionAnswering
    - call

## FlaxRobertaPreLayerNormModel

[[autodoc]] FlaxRobertaPreLayerNormModel
    - __call__

## FlaxRobertaPreLayerNormForCausalLM

[[autodoc]] FlaxRobertaPreLayerNormForCausalLM
    - __call__

## FlaxRobertaPreLayerNormForMaskedLM

[[autodoc]] FlaxRobertaPreLayerNormForMaskedLM
    - __call__

## FlaxRobertaPreLayerNormForSequenceClassification

[[autodoc]] FlaxRobertaPreLayerNormForSequenceClassification
    - __call__

## FlaxRobertaPreLayerNormForMultipleChoice

[[autodoc]] FlaxRobertaPreLayerNormForMultipleChoice
    - __call__

## FlaxRobertaPreLayerNormForTokenClassification

[[autodoc]] FlaxRobertaPreLayerNormForTokenClassification
    - __call__

## FlaxRobertaPreLayerNormForQuestionAnswering

[[autodoc]] FlaxRobertaPreLayerNormForQuestionAnswering
    - __call__