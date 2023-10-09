<!--版权所有2022 The HuggingFace团队。保留所有权利。

根据Apache License, Version 2.0 (许可证)进行许可；除非符合许可证的规定，否则不得使用此文件。
你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律需要或书面同意，根据许可证分发的软件是基于"AS IS"的基础上，无论是明示还是暗示的，不附带任何保证或条件。
有关更多细节，请参阅许可证中的规定。-->

# Trajectory Transformer

<Tip warning={true}>

该模型目前仅处于维护模式，因此不会接受任何更改其代码的新PR。

如果在运行该模型时遇到任何问题，请重新安装支持该模型的最后一个版本：v4.30.0。
你可以通过运行以下命令来执行此操作：`pip install -U transformers==4.30.0`。

</Tip>

## 概述

Trajectory Transformer模型是由Michael Janner，Qiyang Li，Sergey Levine在《Offline Reinforcement Learning as One Big Sequence Modeling Problem》中提出的。

论文摘要如下：

*强化学习（RL）通常涉及估计固定策略或单步模型，利用马尔科夫属性来在时间上分解问题。然而，我们也可以将RL视为一个通用的序列建模问题，其目标是生成一系列行动，以获得一系列高回报。从这个角度来看，我们不禁要考虑在其他领域（如自然语言处理）中表现良好的高容量序列预测模型是否也能为RL问题提供有效的解决方案。为此，我们探索了如何使用序列建模的工具来处理RL问题，使用Transformer架构对轨迹分布进行建模，并将波束搜索重新用作规划算法。将RL作为序列建模问题来构建简化了许多设计决策，使我们能够摒弃许多离线RL算法中常见的组件。我们展示了这种方法在长期动力学预测、模仿学习、目标条件RL和离线RL等方面的灵活性。此外，我们还展示了这种方法可以与现有的无模型算法相结合，从而在稀疏奖励、长期任务中产生最先进的规划器。*

提示：

该Transformer用于深度强化学习。要使用它，你需要从所有先前的时间步长中创建行动、状态和奖励的序列。该模型将把所有这些元素视为一个大序列（一个轨迹）。

该模型由[CarlCochet](https://huggingface.co/CarlCochet)贡献。原始代码可在[此处](https://github.com/jannerm/trajectory-transformer)找到。

## TrajectoryTransformerConfig

[[autodoc]] TrajectoryTransformerConfig


## TrajectoryTransformerModel

[[autodoc]] TrajectoryTransformerModel
    - 前向计算