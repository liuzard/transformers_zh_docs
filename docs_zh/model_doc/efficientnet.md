<!--版权所有2023年HuggingFace团队。

根据Apache许可证，第2版（“许可证”），除非遵守许可证，否则不能使用此文件。
您可以在以下网址获得许可证副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，否则根据许可证分发的软件将按照“原样”方式分发，没有任何形式的明示或暗示担保。
请注意，此文件采用Markdown格式，但包含了我们的文档构建器的特定语法（类似于MDX），这可能在您的Markdown查看器中无法正确呈现。

-->

# EfficientNet

## 概述

EfficientNet模型是由Mingxing Tan和Quoc V. Le在论文《EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks》中提出的。EfficientNet是一系列图像分类模型，具有最先进的准确性，而且比以前的模型小一个数量级并且更快。

论文中的摘要如下：

*卷积神经网络（ConvNets）通常在固定的资源预算上开发，如果有更多资源可用，则按比例扩大以获得更好的准确性。在本文中，我们系统地研究了模型的扩放，并确定了仔细平衡网络的深度、宽度和分辨率可以提高性能。基于这个观察，我们提出了一种新的扩放方法，使用一个简单但非常有效的复合系数来均匀扩放深度/宽度/分辨率的所有维度。我们通过对MobileNets和ResNet扩放的有效性进行了验证。为了进一步，我们使用神经架构搜索设计了一个新的基线网络并对其进行扩放，得到一系列模型，称为EfficientNets，其在准确性和效率方面比以前的ConvNets要好得多。特别是，我们的EfficientNet-B7在ImageNet上实现了最新的84.3%的top-1准确性，同时比最好的现有ConvNet小了8.4倍，推断速度快了6.1倍。我们的EfficientNets在CIFAR-100（91.7%），Flowers（98.8%）和其他3个迁移学习数据集上也表现出色，参数少了一个数量级。*

该模型由[adirik](https://huggingface.co/adirik)贡献。
原始代码可以在[这里](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)找到。

## EfficientNetConfig

[[autodoc]] EfficientNetConfig

## EfficientNetImageProcessor

[[autodoc]] EfficientNetImageProcessor
    - preprocess

## EfficientNetModel

[[autodoc]] EfficientNetModel
    - forward

## EfficientNetForImageClassification

[[autodoc]] EfficientNetForImageClassification
    - forward