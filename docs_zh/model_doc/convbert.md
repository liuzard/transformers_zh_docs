<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），您不得除了符合许可证的情况外使用此文件。
您可以获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件按“原样”的基础分发，没有任何明示或暗示的保证或条件。请参阅许可证以获取
许可证下的特定语言的权限和限制。

⚠️请注意，此文件是Markdown文件，但包含用于我们的文档构建器（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正确显示。

-->

# ConvBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=convbert">
<img alt="模型" src="https://img.shields.io/badge/所有模型页面-convbert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/conv-bert-base">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

ConvBERT模型是由Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan在[ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496)一文中提出的。

论文摘要如下：

*预训练语言模型（如BERT及其变体）最近在各种自然语言理解任务中取得了令人印象深刻的性能。然而，BERT严重依赖全局自注意力机制，因此内存占用和计算成本大。虽然它的所有注意力头都会从全局角度查询整个输入序列以生成注意力图，但我们观察到有些注意力头只需学习局部依赖关系，这意味着存在计算冗余。因此，我们提出了一种新颖的基于跨度的动态卷积来替换这些自注意力头，以直接模拟局部依赖关系。这些新颖的卷积头与其余自注意力头共同形成一个新的混合注意力块，在全局和局部上下文学习方面更加高效。我们将BERT与这种混合注意力设计相结合，构建了ConvBERT模型。实验证明，ConvBERT在各种下游任务中明显优于BERT及其变体，并且具有更低的训练成本和更少的模型参数。值得注意的是，ConvBERTbase模型的GLUE得分为86.4，比ELECTRAbase高0.7，同时训练成本不到1/4。代码和预训练模型将发布。*

对ConvBERT的训练技巧与BERT类似。

该模型由[abhishek](https://huggingface.co/abhishek)贡献。原始实现可以在这里找到:
https://github.com/yitu-opensource/ConvBert

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多选任务指南](../tasks/multiple_choice)

## ConvBertConfig

[[autodoc]] ConvBertConfig

## ConvBertTokenizer

[[autodoc]] ConvBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## ConvBertTokenizerFast

[[autodoc]] ConvBertTokenizerFast

## ConvBertModel

[[autodoc]] ConvBertModel
    - forward

## ConvBertForMaskedLM

[[autodoc]] ConvBertForMaskedLM
    - forward

## ConvBertForSequenceClassification

[[autodoc]] ConvBertForSequenceClassification
    - forward

## ConvBertForMultipleChoice

[[autodoc]] ConvBertForMultipleChoice
    - forward

## ConvBertForTokenClassification

[[autodoc]] ConvBertForTokenClassification
    - forward

## ConvBertForQuestionAnswering

[[autodoc]] ConvBertForQuestionAnswering
    - forward

## TFConvBertModel

[[autodoc]] TFConvBertModel
    - call

## TFConvBertForMaskedLM

[[autodoc]] TFConvBertForMaskedLM
    - call

## TFConvBertForSequenceClassification

[[autodoc]] TFConvBertForSequenceClassification
    - call

## TFConvBertForMultipleChoice

[[autodoc]] TFConvBertForMultipleChoice
    - call

## TFConvBertForTokenClassification

[[autodoc]] TFConvBertForTokenClassification
    - call

## TFConvBertForQuestionAnswering

[[autodoc]] TFConvBertForQuestionAnswering
    - call