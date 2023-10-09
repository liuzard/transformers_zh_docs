<!--
版权所有 © 2022 并保留所有权利的 The HuggingFace 团队。

根据 Apache License, Version 2.0（“许可证”）许可；除非符合 许可证，否则不得使用此文件。您可以在此获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或经书面同意，按原样分发的软件按“现状”提供，不附带任何明示或暗示的担保或条件。有关详细信息，请参阅许可证。

⚠️ 请注意，此文件是 Markdown 格式，但包含特定的语法以用于我们的文档生成器（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。

-->

# UL2

## 概述

T5 模型在《Unifying Language Learning Paradigms》一文中被介绍，作者为 Yi Tay、Mostafa Dehghani、Vinh Q. Tran、Xavier Garcia、Dara Bahri、Tal Schuster、Huaixiu Steven Zheng、Neil Houlsby、Donald Metzler。

文章中的摘要如下：

*现有的预训练模型通常面向特定类别的问题。迄今为止，对于正确的架构和预训练设置似乎仍没有一致的共识。本文提出了一个统一的框架，用于预训练在数据集和设置上通用有效的模型。首先，我们解开架构原型与预训练目标之间的联系 —— 这两个概念通常混淆在一起。接下来，我们提出了自我监督在 NLP 中的广义和统一观点，并展示了不同的预训练目标如何相互关联以及如何通过插值不同目标来实现有效性。然后，我们提出了“混合去噪器”（Mixture-of-Denoisers，MoD），这是一种将多种预训练范例组合在一起的预训练目标。我们还引入了一种模式切换的概念，其中下游微调与特定的预训练方案相关联。我们进行了大量的剖析性实验，比较了多个预训练目标，并发现我们的方法通过在多个不同设置中优于 T5 和/或类似 GPT 的模型来推动 Pareto 前沿。最后，通过将我们的模型扩展到 200B 参数，我们在 50 个广泛建立的监督 NLP 任务上实现了最新性能，这些任务涵盖了语言生成（包括自动和人工评估）、语言理解、文本分类、问答、常识推理、长文本推理、结构化知识基础和信息检索。我们的模型在环境中学习方面也取得了强劲的效果，在零-shot SuperGLUE 上优于 175B GPT-3，一-shot 总结结果是 T5-XXL 的三倍。*

提示：

- UL2 是一个编码器-解码器模型，它在一系列下游任务中进行混合去噪函数预训练和微调。
- UL2 与 [T5v1.1](t5v1.1) 具有相同的体系结构，但使用 Gated-SiLU 激活函数，而不是 Gated-GELU。
- 作者发布了一个架构的检查点，可以在[此处](https://huggingface.co/google/ul2)查看。

原始代码可以在[这里](https://github.com/google-research/google-research/tree/master/ul2)找到。

本模型是由 [DanielHesslow](https://huggingface.co/Seledorn) 贡献的。