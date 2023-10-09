<!--版权所有2022年由HuggingFace团队保留。

根据Apache License, Version 2.0（"许可证"）的规定，除非符合许可证的规定，否则您不得使用此文件。 您可以在下面的网址中获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”的基础进行分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言的权限和限制。

⚠️ 请注意，此文件是Markdown格式，但包含特定语法供我们的文档构建工具（类似于MDX）使用，可能无法在您的Markdown查看器中正确显示。-->

# GLPN

<Tip>

这是一种最近引入的模型，因此尚未对其API进行广泛测试。将来可能需要修复一些错误或进行轻微的破坏性更改。如果你看到了奇怪的东西，请提交一个[Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

</Tip>

## 概述

GLPN模型是由Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, Junmo Kim在[Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth](https://arxiv.org/abs/2201.07436)中提出的。GLPN结合了[SegFormer](segformer)的分层mix-Transformer和轻量级解码器，用于单目深度估计。所提出的解码器比先前提出的解码器表现更好，计算复杂度显著降低。

论文摘要如下：

*从单个图像中估计深度是计算机视觉中一项重要的任务，可以应用于各个领域，并且随着卷积神经网络的发展而快速增长。在本文中，我们提出了一种新颖的结构和训练策略，以进一步提高网络的预测准确性。我们部署了一个分层transformer编码器来捕获和传递全局上下文，并设计了一个轻量级但功能强大的解码器来生成估计的深度图，同时考虑局部连接性。通过在我们提出的选择性特征融合模块中构建多尺度局部特征和全局解码流之间的连接路径，网络可以整合两者的表示并恢复细节。此外，所提出的解码器的性能比先前提出的解码器有所提升，计算复杂度显著降低。此外，我们通过利用深度估计中的一个重要观察结果来改进了深度特定的数据增强方法，以增强模型。我们的网络在具有挑战性的深度数据集NYU Depth V2上实现了最先进的性能。我们已经进行了大量实验证明和展示了所提方法的有效性。最后，我们的模型比其他比较模型具有更好的概括能力和鲁棒性。*

提示：

- 可以使用[`GLPNImageProcessor`]来为模型准备图像。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/glpn_architecture.jpg"
alt="drawing" width="600"/>

<small> 方法概述。取自<a href="https://arxiv.org/abs/2201.07436" target="_blank">原始论文</a>。 </small>

此模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可以在[这里](https://github.com/vinvino02/GLPDepth)找到。

## 资源

官方Hugging Face和社区（标有🌎）资源列表，以帮助您开始使用GLPN。

- [`GLPNForDepthEstimation`]的演示笔记本可以在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/GLPN)找到。
- [单目深度估计任务指南](../tasks/monocular_depth_estimation)

## GLPNConfig

[[autodoc]] GLPNConfig

## GLPNFeatureExtractor

[[autodoc]] GLPNFeatureExtractor
    - __call__

## GLPNImageProcessor

[[autodoc]] GLPNImageProcessor
    - preprocess

## GLPNModel

[[autodoc]] GLPNModel
    - forward

## GLPNForDepthEstimation

[[autodoc]] GLPNForDepthEstimation
    - forward