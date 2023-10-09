<!--版权所有2022 HuggingFace团队。保留所有权利。

根据Apache License，Version 2.0（“许可证”）许可；除非符合许可证的规定，
否则你不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
不附带任何形式的担保或条件。请参阅授权写明的特定语言，
以及许可证下限制的具体内容。

⚠️ 请注意，此文件采用Markdown格式，但包含我们doc-builder（类似于MDX）的特定语法，
可能无法在你的Markdown查看器中正确渲染。

-->

# EfficientFormer

## 概述

EfficientFormer模型是由Yanyu Li，Geng Yuan，Yang Wen，Eric Hu，Georgios Evangelidis，Sergey Tulyakov，Yanzhi Wang，Jian Ren
在[EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)中提出的。
EfficientFormer提出了一种维度一致的纯Transformer，可在移动设备上运行，用于密集预测任务，如图像分类、目标检测和语义分割。

论文中的摘要如下：

*视觉Transformer(ViT)在计算机视觉任务中取得了快速进展，在各种基准上取得了有希望的结果。然而，由于巨大的参数数量和模型设计，例如注意机制，基于ViT的模型通常比轻量级的卷积网络慢几倍。因此，将ViT部署到实时应用中特别具有挑战性，尤其是在资源受限的硬件上，如移动设备。最近的努力通过网络架构搜索或与MobileNet块的混合设计来减少ViT的计算复杂性，但推理速度仍然不令人满意。这引发了一个重要问题：在获得高性能的同时，Transformer能够像MobileNet一样运行得快吗？为了回答这个问题，我们首先回顾了ViT-based模型中使用的网络架构和运算符，并确定了低效的设计。然后，我们引入了一个维度一致的纯Transformer（不含MobileNet块）作为设计范例。最后，我们进行了基于延迟的裁剪，得到了一系列名为EfficientFormer的最终模型。广泛的实验表明了EfficientFormer在移动设备上的性能和速度的优越性。我们最快的模型EfficientFormer-L1在ImageNet-1K上以仅1.6毫秒的推理延迟达到了79.2%的top-1准确率（在iPhone 12上编译为CoreML），这与MobileNetV2×1.4(1.6ms, 74.7% top-1)的速度相当，而我们最大的模型EfficientFormer-L7仅需7.0毫秒的延迟即可达到83.3%的准确率。我们的工作证明了经过适当设计的Transformer在移动设备上可以达到极低的延迟，同时保持高性能。*

此模型由[novice03](https://huggingface.co/novice03)和[Bearnardd](https://huggingface.co/Bearnardd)贡献。
原始代码可以在[此处](https://github.com/snap-research/EfficientFormer)找到。此模型的TensorFlow版本由[D-Roberts](https://huggingface.co/D-Roberts)添加。

## 文档资源

- [图像分类任务指南](../tasks/image_classification)

## EfficientFormerConfig

[[autodoc]] EfficientFormerConfig

## EfficientFormerImageProcessor

[[autodoc]] EfficientFormerImageProcessor
    - 预处理

## EfficientFormerModel

[[autodoc]] EfficientFormerModel
    - 前向传播

## EfficientFormerForImageClassification

[[autodoc]] EfficientFormerForImageClassification
    - 前向传播

## EfficientFormerForImageClassificationWithTeacher

[[autodoc]] EfficientFormerForImageClassificationWithTeacher
    - 前向传播

## TFEfficientFormerModel

[[autodoc]] TFEfficientFormerModel
    - 调用

## TFEfficientFormerForImageClassification

[[autodoc]] TFEfficientFormerForImageClassification
    - 调用

## TFEfficientFormerForImageClassificationWithTeacher

[[autodoc]] TFEfficientFormerForImageClassificationWithTeacher
    - 调用