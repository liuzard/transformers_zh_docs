<!--
版权2023年深情团队。版权所有。

根据Apache许可证，版本2.0（“许可证”）许可;您除非遵守许可证，否则不得使用本文件。您可以获得许可的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于这样的“AS IS”基础分发的,不带有任何种类的明示或暗示的保证和条件，包括但不限于为特定目的的隐含保证和条件。请参阅许可证以了解许可证下的具体语言和限制。

⚠️请注意，此文件是Markdown文件，但包含我们文档构建器的特定语法（类似于MDX），可能无法在Markdown查看器中正确呈现。

-->

# SwiftFormer

## 概述

SwiftFormer模型在[SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications](https://arxiv.org/abs/2303.15446)一文中被阿卜杜勒拉曼·沙克尔、穆罕默德·马兹、哈努娜·拉希德、萨尔曼·汗、明-欢杨、法哈德·沙赫巴兹·汗提出。

SwiftFormer论文提出了一种新颖的高效加性注意力机制，该机制通过线性逐元素乘法有效地替代了自注意计算中的二次矩阵乘法运算。基于此构建了一系列名为'SwiftFormer'的模型，其在准确性和移动端推理速度方面均达到了最先进的性能水平。即使是其较小的变种，在iPhone 14上只需0.8ms的延迟就能达到78.5%的Top-1 ImageNet1K准确率，相比MobileViT-v2更准确且快2倍。

论文摘要如下：

*自注意已成为各种视觉应用中捕获全局上下文的事实选择。然而，自注意计算复杂度与图像分辨率有关的二次计算复杂度限制了其在实时应用中的使用，特别是在资源受限的移动设备上部署。尽管已经提出了混合方法，将卷积和自注意的优点结合起来以获得更好的速度-准确性权衡，但自注意中昂贵的矩阵乘法运算仍然是一个瓶颈。在这项工作中，我们引入一种新颖的高效加性注意力机制，该机制通过线性逐元素乘法有效地替代了二次矩阵乘法运算。我们的设计表明，可以用线性层取代键值交互而不损失任何准确性。与先前的最先进方法不同，我们对自注意的高效制定使其在网络的所有阶段中都能使用。使用我们提出的高效加性注意力，我们构建了一系列名为"SwiftFormer"的模型，其准确性和移动端推理速度在最先进的性能水平上。我们的小型变体在iPhone 14上只需0.8ms的延迟就能达到78.5%的Top-1 ImageNet-1K准确率，相比MobileViT-v2更准确且快2倍。*

提示:
    - 可以使用[`ViTImageProcessor`] API来为模型准备图像。

这个模型由[shehan97](https://huggingface.co/shehan97)贡献。
原始代码可以在[这里](https://github.com/Amshaker/SwiftFormer)找到。


## SwiftFormerConfig

[[autodoc]] SwiftFormerConfig

## SwiftFormerModel

[[autodoc]] SwiftFormerModel
    - forward

## SwiftFormerForImageClassification

[[autodoc]] SwiftFormerForImageClassification
    - forward
