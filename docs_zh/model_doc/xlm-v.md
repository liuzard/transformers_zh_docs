<!--版权所有2023 HuggingFace团队。

根据Apache License， Version 2.0 （“许可证”）发布的；除非符合许可证的相关法律或书面同意要求，不得使用本文件。
你可以从以下链接获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

请注意，该文件采用Markdown格式，但包含着我们的doc-builder特定的语法（类似于MDX），可能在你的Markdown阅读器中无法正确渲染。

-->

# XLM-V

## 概述

XLM-V是一个多语言语言模型，其词汇量为一百万个标记，是在Common Crawl的2.5TB数据上训练的（与XLM-R相同）。
它是由Davis Liang、Hila Gonen、Yuning Mao、Rui Hou、Naman Goyal、Marjan Ghazvininejad、Luke Zettlemoyer和Madian Khabsa在[XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models](https://arxiv.org/abs/2301.10472)论文中首次提出的。

来自XLM-V论文摘要：

*大型多语言语言模型通常依赖于一个跨100多种语言共享的词汇表。
随着这些模型的参数数量和深度增加，词汇量基本不变。
这个词汇瓶颈限制了XLM-R等多语言模型的表示能力。
在本文中，我们介绍了一种通过减少语义重叠较小的语言之间的标记共享，并分配词汇容量来实现每个个体语言足够覆盖的方法，从而扩展到非常大的多语言词汇表。与XLM-R相比，使用我们的词汇表的标记化通常更有语义意义且更短。利用这个改进的词汇表，我们训练了XLM-V，一个具有一百万个标记词汇表的多语言语言模型。在我们测试的每一个任务上，XLM-V都优于XLM-R，包括自然语言推理（XNLI）、问题回答（MLQA、XQuAD、TyDiQA）和命名实体识别（WikiAnn），还有低资源任务（Americas NLI、MasakhaNER）。*

提示：

- XLM-V与XLM-RoBERTa模型架构兼容，只需将模型权重从[`fairseq`](https://github.com/facebookresearch/fairseq)库进行转换即可。
- 使用`XLMTokenizer`实现来加载词汇表并进行标记化。

`XLM-V`（基本大小）模型可以使用[`facebook/xlm-v-base`](https://huggingface.co/facebook/xlm-v-base)标识符来访问。

此模型由[stefan-it](https://huggingface.co/stefan-it)贡献，并包括对XLM-V在下游任务上的详细实验。
实验仓库可以在[这里](https://github.com/stefan-it/xlm-v-experiments)找到。