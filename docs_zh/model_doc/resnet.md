<!--版权所有2022年The HuggingFace团队保留。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证的规定，否则不得使用此文件。
你可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件按“现状”分发，不附带任何形式的明示或暗示的担保
。请参阅许可证以获取许可的特定语言和限制-->

# ResNet

## 概述

ResNet模型提出于《深度残差学习用于图像识别》一文，作者为Kaiming He、Xiangyu Zhang、Shaoqing Ren和Jian Sun。我们的实现遵循[Nvidia](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch)所做的小改动，我们将`3x3`卷积层中的瓶颈部分应用了`stride=2`来进行下采样，而在第一个`1x1`卷积层中没有应用。这通常被称为“ResNet v1.5”。

ResNet引入了残差连接，它们能够训练具有未知层数的网络（最多可达1000层）。ResNet在2015年的ILSVRC和COCO竞赛中获胜，这是深度计算机视觉中的一个重要里程碑。

论文中的摘要如下所示：

*更深的神经网络更难训练。我们提出了一种残差学习框架，以便训练比以前使用的网络更深的网络变得更容易。我们明确地重新定义了层，将层与引用层输入的残差函数学习联系起来，而不是学习未引用的函数。我们提供了全面的经验证据，表明这些残差网络更容易优化，并且可以从大大增加的深度中获得准确性。在ImageNet数据集上，我们评估具有最多152层深度的残差网络，比VGG网络深8倍，但复杂度较低。这组残差网络的集合在ImageNet测试集上的错误率为3.57%。这个结果赢得了ILSVRC 2015分类任务的第一名。我们还在CIFAR-10上对100层和1000层进行了分析。
表示的深度对许多视觉识别任务非常重要。仅仅由于我们极其深的表示，我们在COCO目标检测数据集上获得了28%的相对改进。深残差网络是我们提交ILSVRC & COCO 2015竞赛的基础，我们还在ImageNet检测、ImageNet本地化、COCO检测和COCO分割任务上获得了第一名。*

提示：

- 可以使用[`AutoImageProcessor`]来准备模型的图片。

下图展示了ResNet的架构。取自[原始论文](https://arxiv.org/abs/1512.03385)。

<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/resnet_architecture.png"/>

这个模型由[Francesco](https://huggingface.co/Francesco)贡献。TensorFlow版本的这个模型由[amyeroberts](https://huggingface.co/amyeroberts)添加。原始代码可以在[这里](https://github.com/KaimingHe/deep-residual-networks)找到。

## 资源

以下是官方Hugging Face和社区（用🌎表示）为帮助你开始使用ResNet而提供的资源列表。

<PipelineTag pipeline="image-classification"/>

- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持[`ResNetForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果你有兴趣提交资源以包含在此处，请随时打开Pull Request，我们将对其进行审核！资源应该理想地展示出与现有资源不同的内容，而不是重复现有资源。

## ResNetConfig

[[autodoc]] ResNetConfig


## ResNetModel

[[autodoc]] ResNetModel
    - forward


## ResNetForImageClassification

[[autodoc]] ResNetForImageClassification
    - forward


## TFResNetModel

[[autodoc]] TFResNetModel
    - call


## TFResNetForImageClassification

[[autodoc]] TFResNetForImageClassification
    - call

## FlaxResNetModel

[[autodoc]] FlaxResNetModel
    - __call__

## FlaxResNetForImageClassification

[[autodoc]] FlaxResNetForImageClassification
    - __call__