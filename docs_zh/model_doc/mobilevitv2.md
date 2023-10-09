<!--版权所有 2023年HuggingFace团队保留。

基于Apache许可证2.0版（“许可证”）获得许可；除非符合许可证要求，否则不能使用此文件。您可以获取许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，按“原样”分发的软件仅依据许可证发布，不附带任何形式的明示或暗示保证。有关许可证下特定语言的规定，请参阅许可证。

⚠️请注意，此文件采用Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），在您的Markdown查看器中可能无法正确显示。

-->

# MobileViTV2

## 概述

MobileViTV2模型是由Sachin Mehta和Mohammad Rastegari于[《Separable Self-attention for Mobile Vision Transformers》](https://arxiv.org/abs/2206.02680)中提出的。

MobileViTV2是MobileViT的第二个版本，通过用可分离的自注意力替换MobileViT中的多头自注意力来构建。

该论文的摘要如下：

*移动视觉Transformer（MobileViT）在多个移动视觉任务（包括分类和检测）中可以达到最先进的性能。尽管这些模型的参数较少，但与基于卷积神经网络的模型相比，它们具有较高的延迟。MobileViT的主要效率瓶颈在于Transformer中的多头自注意力（MHA），其相对于令牌（或块）数量k的时间复杂度为O(k2)。此外，MHA在计算自注意力时需要昂贵的操作（如批次矩阵乘法），从而影响了资源受限设备上的延迟。本论文引入了一种具有线性复杂度（即O(k)）的可分离自注意力方法。所提出的方法的一个简单而有效的特点是，它使用逐元素操作来计算自注意力，使其成为资源受限设备的良好选择。改进的MobileViTV2模型在多个移动视觉任务（包括ImageNet对象分类和MS-COCO对象检测）上达到了最先进的性能。MobileViTV2仅有大约300万个参数，在ImageNet数据集上实现了75.6％的top-1准确率，相比于MobileViT提高了约1％，同时在移动设备上运行速度提高了3.2倍。*

提示：

- MobileViTV2更像是卷积神经网络（CNN）而不是Transformer模型。它不适用于序列数据，而是适用于图像批次。与ViT不同，它没有嵌入层。骨干模型输出特征图。
- 可以使用`MobileViTImageProcessor`来为模型准备图像。请注意，如果您自己进行预处理，则预训练检查点要求图像以BGR像素顺序（而不是RGB）呈现。
- 可用的图像分类检查点已在[ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)上进行了预训练（也称为ILSVRC 2012，包含130万张图像和1000个类别）。
- 分割模型使用[DeepLabV3](https://arxiv.org/abs/1706.05587)头部。可用的语义分割检查点已在[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)上进行了预训练。

该模型由[shehan97](https://huggingface.co/shehan97)贡献。
原始代码可在[此处](https://github.com/apple/ml-cvnets)找到。


## MobileViTV2Config

[[autodoc]] MobileViTV2Config

## MobileViTV2Model

[[autodoc]] MobileViTV2Model
    - forward

## MobileViTV2ForImageClassification

[[autodoc]] MobileViTV2ForImageClassification
    - forward

## MobileViTV2ForSemanticSegmentation

[[autodoc]] MobileViTV2ForSemanticSegmentation
    - forward