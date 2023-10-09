<!--版权 2022 年 HuggingFace 团队。版权所有。

根据 Apache 许可证第 2.0 版（“许可证”）许可，除非符合许可证的规定，否则您无权使用此文件。
您可以获取许可证的副本，请参阅

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件。
请参阅许可证，了解许可证下的特定语言，以及许可证下的限制。

⚠️ 请注意，这个文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，
可能无法在您的 Markdown 查看器中正确显示。

-->

# 决策 Transformer

## 概览

决策 Transformer 模型的提出在[《Decision Transformer: Reinforcement Learning via Sequence Modeling》](https://arxiv.org/abs/2106.01345)一文中
由 Lili Chen、Kevin Lu、Aravind Rajeswaran、Kimin Lee、Aditya Grover、Michael Laskin、Pieter Abbeel、Aravind Srinivas、Igor Mordatch 提出。

论文摘要如下：

*我们提出了一个将强化学习（RL）抽象为序列建模问题的框架。
这使我们能够利用 Transformer 架构的简单性和可扩展性，以及与 GPT-x 和 BERT 等语言建模的相关进展。
特别是，我们提出了决策 Transformer，一种将 RL 问题视为条件序列建模问题的架构。
与先前的 RL 方法不同，先前的方法要么适应值函数，要么计算策略梯度，而决策 Transformer 仅通过利用因果遮蔽 Transformer 输出最佳动作。
通过将自回归模型的输入设置为期望的回报（奖励）、过去的状态和动作，我们的决策 Transformer 模型可以生成未来实现期望回报的动作。
尽管简单，决策 Transformer 在 Atari、OpenAI Gym 和 Key-to-Door 任务上可以达到或超过最先进的无模型离线 RL 基线的性能。*

提示：

该模型版本适用于状态为向量的任务，基于图像的状态将很快推出。

此模型由 [edbeeching](https://huggingface.co/edbeeching) 贡献。原始代码可在 [此处](https://github.com/kzl/decision-transformer) 找到。

## DecisionTransformerConfig

[[autodoc]] DecisionTransformerConfig


## DecisionTransformerGPT2Model

[[autodoc]] DecisionTransformerGPT2Model
    - forward

## DecisionTransformerModel

[[autodoc]] DecisionTransformerModel
    - forward