<!--版权归2023年HuggingFace团队所有。

根据Apache许可证第2版（“许可证”）许可；除非符合许可证，否则你不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非根据适用法律或书面同意，按“原样”分发的软件是基于许可证分发的，
没有任何明示或暗示的担保或条件。查看许可证以获取特定的语言和限制条款。

⚠️请注意，此文件在Markdown中，但包含了我们文档生成器的特定语法（类似于MDX），
这可能无法在你的Markdown查看器中正确呈现。-->

# Autoformer

## 概述

Autoformer模型是由Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long在《Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting》一文中提出的。

该模型将Transformer作为一个深度分解架构，可以在预测过程中逐步分解趋势和季节成分。

论文摘要如下：

*扩展预测时间是现实应用的关键需求，例如极端天气预警和长期能源消耗规划。本文研究时间序列的长期预测问题。之前基于Transformer的模型采用各种自注意机制来发现长程依赖关系。然而，长期未来的复杂时间模式阻止模型发现可靠的依赖关系。此外，为了处理长序列的效率，Transformer必须采用稀疏版本的逐点自注意力，导致信息利用瓶颈。在超越Transformer的基础上，我们设计了Autoformer作为一种具有自相关机制的新型分解架构。我们打破了之前序列分解的预处理约定，将其改进为深度模型的基本内部块。这种设计赋予了Autoformer对复杂时间序列进行逐步分解的能力。此外，受随机过程理论的启发，我们设计了基于序列周期性的自相关机制，该机制在子系列级别进行依赖发现和表示聚合。自相关机制在效率和准确性方面优于自注意力。在长期预测方面，Autoformer具有最先进的准确性，对于五个实际应用（能源、交通、经济、天气和疾病）的六个基准测试中，相对改进了38%。*

该模型由[elisim](https://huggingface.co/elisim)和[kashif](https://huggingface.co/kashif)贡献。
原始代码可以在[此处](https://github.com/thuml/Autoformer)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）资源列表，可帮助你入门。如果你有兴趣提交资源以包含在此处，请随时发起拉取请求，我们将进行审核！该资源应该 ideally 展示出与现有资源不重复的新内容。

- 在HuggingFace博客中查看Autoformer的博文：[Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer)](https://huggingface.co/blog/autoformer)

## AutoformerConfig

[[autodoc]] AutoformerConfig


## AutoformerModel

[[autodoc]] AutoformerModel
    - forward


## AutoformerForPrediction

[[autodoc]] AutoformerForPrediction
    - forward