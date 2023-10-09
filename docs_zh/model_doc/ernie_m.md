<!--版权2023由HuggingFace和百度团队所有。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证中的规定，否则不得使用此文件。
您可以获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，不附带任何形式的保证或条件。
有关特定语言下的许可证的限制，请参阅许可证。

⚠️请注意，此文件是Markdown格式的，但包含我们的文档生成器（类似于MDX）的特定语法，这可能在Markdown查看器中无法正确渲染。

-->

# ErnieM

## 概述

ErnieM模型是由Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang在《ERNIE-M:通过与单语语料库对齐跨语言语义增强多语言表示》（https://arxiv.org/abs/2012.15674）中提出的。

以下是论文摘要：

*最近的研究表明，预训练的跨语言模型在下游跨语言任务中取得了令人印象深刻的性能。这一改进得益于学习大量的单语和平行语料库。尽管普遍认为平行语料对于改善模型性能至关重要，但现有方法通常受到平行语料大小的限制，尤其是对于资源稀缺语言。在本文中，我们提出了一种新的训练方法ERNIE-M，它鼓励模型通过单语语料库对齐多种语言的表示，从而克服平行语料大小对模型性能的约束。我们的关键观点是将回译集成到预训练过程中。我们在单语语料库上生成伪平行句对，以便学习不同语言之间的语义对齐，从而增强跨语言模型的语义建模。实验证明ERNIE-M优于现有的跨语言模型，并在各种跨语言下游任务中取得了最新的 state-of-the-art 结果。*

提示：

1. Ernie-M是类似BERT的模型，因此是堆叠的Transformer Encoder。
2. 与BERT不同，作者使用了两种新技术来进行预训练：`Cross-attention Masked Language Modeling`和`Back-translation Masked Language Modeling`。目前，这两种LMHead目标在这里没有实现。
3. 它是一个多语言语言模型。
4. 预训练过程未使用Next Sentence Prediction。

该模型由[Susnato Dhar](https://huggingface.co/susnato)提供。原始代码可在[此处](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_m)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [多项选择任务指南](../tasks/multiple_choice)

## ErnieMConfig

[[autodoc]] ErnieMConfig


## ErnieMTokenizer

[[autodoc]] ErnieMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## ErnieMModel

[[autodoc]] ErnieMModel
    - forward

## ErnieMForSequenceClassification

[[autodoc]] ErnieMForSequenceClassification
    - forward


## ErnieMForMultipleChoice

[[autodoc]] ErnieMForMultipleChoice
    - forward


## ErnieMForTokenClassification

[[autodoc]] ErnieMForTokenClassification
    - forward


## ErnieMForQuestionAnswering

[[autodoc]] ErnieMForQuestionAnswering
    - forward

## ErnieMForInformationExtraction

[[autodoc]] ErnieMForInformationExtraction
    - forward