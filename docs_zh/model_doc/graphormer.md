<!--版权 2022年The HuggingFace团队和Microsoft。版权所有。

根据MIT许可证；你除非遵守许可证，否则不得使用此文件。

除非适用法律要求或书面同意，在许可证下分发的软件都是基于"按原样"的基础分发的，没有任何明示或暗示的担保和条件。详见许可证中关于具体语言的限制和责任的条款。

注意，此文件是Markdown格式的，但包含特定的语法，用于我们的文档构建器（类似于MDX），可能不能正确显示在你的Markdown查看器中。

-->

# Graphormer

## 概述

Graphormer模型是由[Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234)一文中提出的，作者是Chengxuan Ying，Tianle Cai，Shengjie Luo，Shuxin Zheng，Guolin Ke，Di He，Yanming Shen和Tie-Yan Liu。它是一个图形Transformer模型，通过在预处理和整理期间生成嵌入和感兴趣的特征，并使用修改后的注意力，允许对图形而不是文本序列进行计算。

论文中的摘要如下：

*Transformer架构已经成为许多领域的主要选择，如自然语言处理和计算机视觉。然而，在流行的图级预测排行榜上，它与主流的GNN变种相比，并没有取得竞争性的性能。因此，Transformer模型如何适用于图表示学习仍然是一个谜。在本文中，我们通过提出Graphormer来解决这个谜题，它是建立在标准Transformer架构之上的，并且在广泛的图表示学习任务中获得了出色的结果，尤其是在最近的OGB大规模挑战上。我们运用Transformer在图形中的关键见解是将图的结构信息有效地编码到模型中。为此，我们提出了几种简单而有效的结构编码方法，以帮助Graphormer更好地模拟图结构化数据。此外，我们对Graphormer的表达能力进行了数学刻画，并展示了以我们的方式编码图结构信息时，许多流行的GNN变种可以被视为Graphormer的特殊情况。*

提示：

该模型在大型图形上（节点/边大于100个）的效果不佳，因为会导致内存溢出。
你可以减小批量大小、增加RAM内存或减小algos_graphormer.pyx中的`UNREACHABLE_NODE_DISTANCE`参数，但要超过700个节点/边会很困难。

该模型不使用分词器，而是在训练期间使用特殊的整理器。

该模型由[clefourrier](https://huggingface.co/clefourrier)贡献。原始代码可以在[此处](https://github.com/microsoft/Graphormer)找到。

## GraphormerConfig

[[autodoc]] GraphormerConfig


## GraphormerModel

[[autodoc]] GraphormerModel
    - forward


## GraphormerForGraphClassification

[[autodoc]] GraphormerForGraphClassification
    - forward