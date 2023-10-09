<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache License，Version 2.0（“许可证”）许可。除非符合
许可下的规定，否则你不能使用此文件
许可证可以从以下网址获得：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，否则分发的软件将在
“按原样”基础上分发，没有任何担保或条件，
无论是明示的还是暗示的。有关许可的详细信息，请参阅许可证
特定语言覆盖下的限制。

⚠️ 请注意，此文件是Markdown格式的，但包含特定于我们doc-builder的语法（类似于MDX），可能在你的Markdown查看器中无法正确呈现。

-->

# Transformer XL

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=transfo-xl">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-transfo--xl-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/transfo-xl-wt103">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

Transformer-XL模型由Zihang Dai、Zhilin Yang、Yiming Yang、Jaime Carbonell、Quoc V.Le、Ruslan Salakhutdinov在《Transformer-XL：超越固定长度上下文的注意语言模型》中提出。它是具有相对定位（正弦）嵌入的因果（单向）变压器，可以复用以前计算的隐藏状态以便于更长的上下文（记忆）。该模型还使用自适应softmax输入和输出（绑定）。

论文中的摘要如下：

“变压器具有学习更长期依赖的潜力，但在语言建模的情况下受限于固定长度上下文。我们提出了一种新颖的神经体系结构Transformer-XL，它在不破坏时间连贯性的情况下，能够超越固定长度学习依赖关系。它由一个段级递归机制和一种新颖的位置编码方案组成。我们的方法不仅能够捕捉更长期的依赖，还能解决上下文碎片化问题。结果，Transformer-XL学习到的依赖性比RNN长80%，比常规Transformer长450%，在短序列和长序列上性能更好，并且在评估期间比常规Transformer快1800+倍。值得注意的是，我们将bpc/困惑度的最新结果改进为enwiki8的0.99，text8的1.08，WikiText-103的18.3，One Billion Word的21.8，Penn Treebank的54.5（无细调）。只在WikiText-103上进行培训时，Transformer-XL成功生成了具有数千个标记的合理连贯的新文本文章。”

提示：

- Transformer-XL使用相对正弦位置嵌入。可以在左侧或右侧进行填充。原始实现在SQuAD上训练时进行左侧填充，因此填充默认设置为左侧。
- Transformer-XL是少数没有序列长度限制的模型之一。
- 与常规GPT模型相同，但引入了两个连续片段的循环机制（类似于连续输入的常规RNN）。在此上下文中，片段是一串连续的标记（例如512），可以跨多个文档，然后按顺序输入模型。
- 基本上，将前一个片段的隐藏状态与当前输入连接起来以计算注意力分数。这使得模型能够关注先前片段中的信息以及当前片段中的信息。通过堆叠多个注意层，可以将感受野增加到多个先前段。
- 这将位置嵌入更改为位置相对嵌入（因为常规位置嵌入在给定位置上的当前输入和当前隐藏状态会产生相同的结果），并且需要对计算注意力分数的方式进行一些调整。

此模型由[thomwolf](https://huggingface.co/thomwolf)贡献。原始代码可以在[这里](https://github.com/kimiyoung/transformer-xl)找到。

<Tip warning={true}>

由于PyTorch中的一个错误，TransformerXL与*torch.nn.DataParallel*不兼容，请参阅[问题＃36035](https://github.com/pytorch/pytorch/issues/36035)

</Tip>

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [因果语言模型任务指南](../tasks/language_modeling)

## TransfoXLConfig

[[autodoc]] TransfoXLConfig

## TransfoXLTokenizer

[[autodoc]] TransfoXLTokenizer
    - save_vocabulary

## TransfoXL特定的输出

[[autodoc]] models.transfo_xl.modeling_transfo_xl.TransfoXLModelOutput

[[autodoc]] models.transfo_xl.modeling_transfo_xl.TransfoXLLMHeadModelOutput

[[autodoc]] models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLModelOutput

[[autodoc]] models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLLMHeadModelOutput

## TransfoXLModel

[[autodoc]] TransfoXLModel
    - forward

## TransfoXLLMHeadModel

[[autodoc]] TransfoXLLMHeadModel
    - forward

## TransfoXLForSequenceClassification

[[autodoc]] TransfoXLForSequenceClassification
    - forward

## TFTransfoXLModel

[[autodoc]] TFTransfoXLModel
    - call

## TFTransfoXLLMHeadModel

[[autodoc]] TFTransfoXLLMHeadModel
    - call

## TFTransfoXLForSequenceClassification

[[autodoc]] TFTransfoXLForSequenceClassification
    - call

## 内部层

[[autodoc]] AdaptiveEmbedding

[[autodoc]] TFAdaptiveEmbedding