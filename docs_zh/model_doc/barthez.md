<!--版权所有2020年HuggingFace团队，保留所有权利。

根据Apache许可证第2版（"许可证"），除非符合许可证的规定，否则你不得使用此文件。你可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件以"按原样"的方式分发，不提供任何明示或暗示的保证或条件。有关许可证的特定语言和限制，请参阅许可证。

⚠️ 请注意，本文件是以Markdown格式编写的，但包含特定于我们doc-builder的语法（类似于MDX），可能无法在你的Markdown查看器中正确渲染。

-->

# BARThez

## 概述

BARThez模型是由Moussa Kamal Eddine、Antoine J.-P. Tixier和Michalis Vazirgiannis于2020年10月23日在[《BARThez: a Skilled Pretrained French Sequence-to-Sequence Model》](https://arxiv.org/abs/2010.12321)中提出的。

论文摘要：
*归纳传递学习是由自监督学习所实现的，在自然语言处理（NLP）领域引起了轰动，诸如BERT和BART等模型在无数自然语言理解任务上取得了新的最佳效果。虽然有一些值得注意的例外，但大多数可用的模型和研究都是针对英语进行的。在这项工作中，我们引入了BARThez，这是适用于法语的第一个BART模型（以我们所知）。BARThez在我们过去的研究中使用的非常大的单语法语话语料库上进行了预训练，我们对其进行了调整以适应BART的扰动方案。与已有的基于BERT的法语语言模型，如CamemBERT和FlauBERT不同，BARThez特别适用于生成任务，因为它不仅预训练了编码器，而且还预训练了解码器。除了FLUE基准的判别任务外，我们还在一个新的摘要数据集OrangeSum上评估了BARThez，我们在本文中发布了这个数据集。我们还继续对已经在BARThez语料库上进行了预训练的多语言BART进行预训练，并展示了由此产生的模型，我们称之为mBARTHez，与普通的BARThez相比，提供了显著的提升，并且与CamemBERT和FlauBERT不相上下或更胜一筹。*

此模型由[moussakam](https://huggingface.co/moussakam)贡献。作者的代码可以在[此处](https://github.com/moussaKam/BARThez)找到。

### 示例
- BARThez可以以与BART类似的方式对序列到序列任务进行微调，请参考：[examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md)。


## BarthezTokenizer

[[autodoc]] BarthezTokenizer

## BarthezTokenizerFast

[[autodoc]] BarthezTokenizerFast