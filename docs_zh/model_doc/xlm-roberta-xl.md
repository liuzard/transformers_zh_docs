<!--版权所有2022年HuggingFace团队。保留一切权利。

根据Apache许可证2.0版（“许可证”），除非符合许可证规定，否则你不得使用此文件。你可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按
"按原样"基础分发，不附带任何明示或暗示的担保或条件。详见许可证
明确了特定语言的性能和限制的详细信息。

⚠️请注意，该文件是Markdown格式的，但包含特定的语法，用于我们的doc-builder（类似于MDX），这可能不会
在你的Markdown查看器中正确显示。

-->

# XLM-RoBERTa-XL

## 概览

XLM-RoBERTa-XL模型是由Naman Goyal、Jingfei Du、Myle Ott、Giri Anantharaman、Alexis Conneau在《大规模跨语言掩码语言建模的转换器》中提出的[（Larger-Scale Transformers for Multilingual Masked Language Modeling）](https://arxiv.org/abs/2105.00572)。

论文中的摘要如下：

*最近的研究表明，跨语言语言模型预训练对于跨语言理解的有效性。在本研究中，我们展示了两个更大的多语种掩码语言模型，其参数为35亿个和107亿个。我们的两个新模型称为XLM-R XL和XLM-R XXL，其在XNLI上的平均准确率超过XLM-R分别1.8%和2.4%。与GLUE基准的RoBERTa-Large模型相比，在几个英语任务上都提高了0.3%的性能，同时处理了多出99个语言。这表明，容量更大的预训练模型可能在高资源语言上获得强大的性能，同时极大地提高低资源语言。我们将我们的代码和模型公开提供。*

提示：

- XLM-RoBERTa-XL是在100种不同语言上训练的多语种模型。与一些XLM多语种模型不同，它不需要‘lang’张量来理解使用的语言，应该能够根据输入id确定正确的语言。

该模型由[Soonhwan-Kwon](https://github.com/Soonhwan-Kwon)和[stefan-it](https://huggingface.co/stefan-it)贡献。原始代码可以在[此处](https://github.com/pytorch/fairseq/tree/master/examples/xlmr)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多选任务指南](../tasks/multiple_choice)

## XLMRobertaXLConfig

[[autodoc]] XLMRobertaXLConfig

## XLMRobertaXLModel

[[autodoc]] XLMRobertaXLModel
    - forward

## XLMRobertaXLForCausalLM

[[autodoc]] XLMRobertaXLForCausalLM
    - forward

## XLMRobertaXLForMaskedLM

[[autodoc]] XLMRobertaXLForMaskedLM
    - forward

## XLMRobertaXLForSequenceClassification

[[autodoc]] XLMRobertaXLForSequenceClassification
    - forward

## XLMRobertaXLForMultipleChoice

[[autodoc]] XLMRobertaXLForMultipleChoice
    - forward

## XLMRobertaXLForTokenClassification

[[autodoc]] XLMRobertaXLForTokenClassification
    - forward

## XLMRobertaXLForQuestionAnswering

[[autodoc]] XLMRobertaXLForQuestionAnswering
    - forward