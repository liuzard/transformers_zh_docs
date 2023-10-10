<!--版权所有2021年The HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（"许可证"）使用本文件，除非你遵守许可证，否则不得使用此文件。
你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则按"原样"分发的软件根据许可证分发，
不附带任何明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言和限制条款。

⚠️请注意，此文件为Markdown格式，但包含特定于我们的doc-builder（类似于MDX）的语法，
在你的Markdown查看器中可能无法正常渲染。-->

# FNet

## 概述

FNet模型是由James Lee-Thorp、Joshua Ainslie、Ilya Eckstein和Santiago Ontanon于[FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)中提出的。该模型将BERT模型中的自注意层替换为傅里叶变换，仅返回变换的实部。该模型由于参数较少且内存使用效率更高，因此比BERT模型快得多。在GLUE基准测试中，该模型在BERT对应模型的准确率上达到了92-97%左右，并且训练速度比BERT模型快得多。论文摘要如下：

*我们证明Transformer编码器架构可以通过用“混合”输入token的简单线性变换替换自注意子层来加速，而准确度损失有限。这些线性混合器以及前馈层中的标准非线性性在几个文本分类任务中证明可以很好地模拟语义关系。最令人惊讶的是，我们发现将Transformer编码器中的自注意子层替换为标准的、非参数化傅里叶变换，在GLUE基准测试中实现了与BERT对应模型92-97%的准确度，但在标准512输入长度的GPU上训练速度提高了80%，在标准512输入长度的TPU上训练速度提高了70%。在更长的输入长度上，FNet模型的速度显著更快：与“高效”的Transformer模型相比，在长距离竞技场（Long Range Arena）基准测试中，FNet与最准确的模型的准确率相匹配，并且在GPU上的所有序列长度上快于最快的模型（在TPU上相对较短的长度上快）。最后，FNet具有轻量级的内存占用和特别高效的小型模型大小；在固定的速度和准确度预算下，小型的FNet模型优于Transformer对应模型。*

使用提示：

- 该模型在训练时没有使用attention mask，因为它基于傅里叶变换。该模型使用最大序列长度512进行训练，其中包括填充符号。因此，强烈建议在微调和推理中使用相同的最大序列长度。

该模型由[gchhablani](https://huggingface.co/gchhablani)贡献。可以在[这里](https://github.com/google-research/google-research/tree/master/f_net)找到原始代码。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## FNetConfig

[[autodoc]] FNetConfig

## FNetTokenizer

[[autodoc]] FNetTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FNetTokenizerFast

[[autodoc]] FNetTokenizerFast

## FNetModel

[[autodoc]] FNetModel
    - forward

## FNetForPreTraining

[[autodoc]] FNetForPreTraining
    - forward

## FNetForMaskedLM

[[autodoc]] FNetForMaskedLM
    - forward

## FNetForNextSentencePrediction

[[autodoc]] FNetForNextSentencePrediction
    - forward

## FNetForSequenceClassification

[[autodoc]] FNetForSequenceClassification
    - forward

## FNetForMultipleChoice

[[autodoc]] FNetForMultipleChoice
    - forward

## FNetForTokenClassification

[[autodoc]] FNetForTokenClassification
    - forward

## FNetForQuestionAnswering

[[autodoc]] FNetForQuestionAnswering
    - forward