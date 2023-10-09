<!--
版权所有2020年HuggingFace团队保留。

根据Apache许可证第2.0版（“许可证”）授权;除非符合许可证，否则不得使用该文件。您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样” BASIS，没有任何明示或暗示的保证或条件。参见许可证以获取
许可证下的特定语言的管理权限和限制。

⚠️请注意，此文件是Markdown格式的，但包含我们的doc-builder的特定语法（类似于MDX），可能无法正确显示在您的Markdown查看器中。

-->

# DialoGPT

## 概述

DialoGPT是由Yizhe Zhang，Siqi Sun，Michel Galley，Yen-Chun Chen，Chris Brockett，Xiang Gao，Jianfeng Gao，Jingjing Liu和Bill Dolan在《大规模生成预训练用于对话式回复生成的DialoGPT》一文中提出的。它是在Reddit上从147M个类似对话的交流中提取的数据集上训练的GPT2模型。

论文摘要如下：

*我们提出了一个大型的、可调整的神经对话回复生成模型DialoGPT（对话生成预训练变压器）。DialoGPT的训练数据集从2005年到2017年Reddit的评论链中提取了147M个对话样本。与单回合对话环境中的自动评估和人工评估相比，DialoGPT在性能上接近于人类。我们展示了利用DialoGPT的对话系统生成的响应比强基线系统更具相关性、更丰富、更一致。预训练模型和训练pipeline已公开发布，以促进神经响应生成的研究和更智能的开放域对话系统的开发。*

提示：

- DialoGPT是一个带有绝对位置嵌入的模型，所以通常建议在输入中添加右填充而不是左填充。
- DialoGPT是通过针对对话数据进行因果语言建模（CLM）训练的，因此在开放域对话系统中生成响应非常强大。
- DialoGPT使用户能够只用10行代码创建一个聊天机器人，如[DialoGPT的模型卡片](https://huggingface.co/microsoft/DialoGPT-medium)所示。

训练：

为了训练或微调DialoGPT，可以使用因果语言建模训练。引用官方论文如下：*我们遵循OpenAI GPT-2的方法，将多轮对话会话建模为一段长文本，并将生成任务构建为语言建模。我们首先将对话会话中的所有对话转换为一段长文本x_1，...，x_N（N是序列长度），并以文本结束的标记结束。*有关更多信息，请参阅原始论文。

DialoGPT的架构基于GPT2模型，因此可以参考[GPT2的文档页面](gpt2)。

可以在[此处](https://github.com/microsoft/DialoGPT)找到原始代码。