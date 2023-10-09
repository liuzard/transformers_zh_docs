<!--
版权所有2020年HuggingFace团队。版权所有。

根据Apache许可证2.0版（“许可证”）的规定，你不得使用此文件，除非符合许可证的规定。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以“原样”分发的软件都不附带任何明示或暗示的担保或条件。详见许可证下的特定语言以及许可证的限制。

⚠️请注意，此文件使用Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），可能无法正确呈现在你的Markdown查看器中。

-->

# 优化

`.optimization`模块提供了以下内容：

- 带有固定权重衰减的优化器，可用于微调模型，并且
- 以对象形式继承自`_LRSchedule`的时间表：
- 用于累积多个批次梯度的梯度累积类

## AdamW (PyTorch)

[[autodoc]] AdamW

## AdaFactor (PyTorch)

[[autodoc]] Adafactor

## AdamWeightDecay (TensorFlow)

[[autodoc]] AdamWeightDecay

[[autodoc]] create_optimizer

## 时间表

### 学习率时间表（Pytorch）

[[autodoc]] SchedulerType

[[autodoc]] get_scheduler

[[autodoc]] get_constant_schedule

[[autodoc]] get_constant_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_constant_schedule.png"/>

[[autodoc]] get_cosine_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_schedule.png"/>

[[autodoc]] get_cosine_with_hard_restarts_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_hard_restarts_schedule.png"/>

[[autodoc]] get_linear_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_linear_schedule.png"/>

[[autodoc]] get_polynomial_decay_schedule_with_warmup

[[autodoc]] get_inverse_sqrt_schedule

### 启动（TensorFlow）

[[autodoc]] WarmUp

## 梯度策略

### GradientAccumulator（TensorFlow）

[[autodoc]] GradientAccumulator