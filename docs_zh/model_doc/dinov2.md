<!--版权所有2023年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）授权；除非符合许可证规定，否则您不得使用此文件。
您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样”的，没有任何明示或暗示的保证或条件。请参阅许可证以获取有关许可的具体语言和限制的信息。-->

# DINOv2

## 概述

[DINOv2：无监督学习强大的视觉特征](https://arxiv.org/abs/2304.07193)是由Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski提出的。DINOv2是[DINO](https://arxiv.org/abs/2104.14294)的升级版，它是应用于[视觉变换器](vit)的自监督方法。该方法可以生成适用于各种图像分布和任务的通用视觉特征，即无需微调即可工作的特征。

论文中的摘要如下：

*自然语言处理在大量数据的模型预训练方面取得的最新突破为计算机视觉中的类似基础模型铺平了道路。通过生成适用于各种图像分布和任务的通用视觉特征，这些模型可以极大地简化任何系统中对图像的使用，而无需进行微调。本文表明，如果在足够多的来自不同来源的筛选数据上进行训练，现有的预训练方法，特别是自监督方法，可以产生这种特征。我们重新审视现有方法，并结合不同技术来扩大我们在数据和模型规模方面的预训练。大多数技术贡献旨在加速和稳定大规模训练。在数据方面，我们提出了一个自动流程来构建一个专用的、多样化的和筛选的图像数据集，而不是像自监督文献中通常所做的未经筛选的数据。在模型方面，我们使用1B参数训练一个ViT模型(Dosovitskiy et al., 2020)，并将其压缩为一系列更小的模型，超越了现有的最佳通用特征OpenCLIP(Ilharco et al., 2021)在大多数图像和像素水平的基准测试中。*

提示：

- 您可以使用[`AutoImageProcessor`]类来为模型准备图像。

此模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[此处](https://github.com/facebookresearch/dinov2)找到。


## Dinov2Config

[[autodoc]] Dinov2Config

## Dinov2Model

[[autodoc]] Dinov2Model
    - forward

## Dinov2ForImageClassification

[[autodoc]] Dinov2ForImageClassification
    - forward