<!-- 版权所有2023年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）授权；您不得以任何方式使用此文件，除非您遵守许可证。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非根据适用法律或书面同意，本许可的软件将按“原样”分发，
不提供任何明示或暗示的担保或条件。请参阅许可证以获取
特定语言的权限和限制。

⚠️请注意，此文件采用Markdown格式，但包含特定于我们文档生成器（类似MDX）的语法，可能无法在Markdown查看器中正确显示。-->

# Informer

## 概述

Informer模型是由Haoyi Zhou、Shanghang Zhang、Jieqi Peng、Shuai Zhang、Jianxin Li、Hui Xiong和Wancai Zhang在文章[“Informer：非常高效的长序列时间序列预测转换器”]（https://arxiv.org/abs/2012.07436）中提出的。

该方法引入了概率注意机制来选择“活动”查询而不是“懒惰”查询，并提供了一个稀疏的Transformer，从而减轻了Vanilla Attention的二次计算和内存需求。

文章的摘要如下：

*许多现实世界的应用需要对长序列时间序列进行预测，例如电力消耗规划。长序列时间序列预测（LSTF）要求模型具有高预测能力，即能够有效捕捉输出和输入之间精确的长程依赖性耦合。最近的研究表明，Transformer具有增加预测能力的潜力。然而，Transformer存在几个严重问题，阻碍了其直接适用于LSTF，包括二次时间复杂性，高内存使用量以及编码器-解码器架构的固有限制。为了解决这些问题，我们设计了一种高效的基于Transformer的LSTF模型，称为Informer，具有以下三个独特特点：（i）ProbSparse自注意机制，其在时间复杂度和内存使用上达到O（L logL），并在序列的依赖性对齐方面具有可比性的性能。（ii）自注意去除突出显示了主导注意通过将层输入减半，高效地处理极长的输入序列。（iii）生成式解码器，概念上简单，一次性预测长时间序列序列，而不是逐步方式，大大提高了长序列预测的推断速度。对四个大型数据集的大量实验证明，Informer显着优于现有方法，并为LSTF问题提供了新的解决方案。*

这个模型由[elisim](https://huggingface.co/elisim)和[kashif](https://huggingface.co/kashif)贡献。
原始代码可以在[这里](https://github.com/zhouhaoyi/Informer2020)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）资源的列表，可帮助您入门。如果您有兴趣提交资源以包含在此处，请随时打开Pull Request，我们将进行审查！该资源应以展示新内容为理想，而不是重复现有资源。

- 在HuggingFace博客中阅读《Informer》的博文：[与Informer一起进行多变量概率时间序列预测](https://huggingface.co/blog/informer)

## InformerConfig

[[autodoc]] InformerConfig


## InformerModel

[[autodoc]] InformerModel
    - forward


## InformerForPrediction

[[autodoc]] InformerForPrediction
    - forward