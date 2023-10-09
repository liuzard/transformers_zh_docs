<!--版权所有2022年HuggingFace团队。 版权所有。

根据Apache许可证，版本2.0进行许可（“许可证”）;你不得使用此文件，除非要遵守许可。
您可以在以下位置获取许可的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以软件形式分发的软件在
基于“原样”，无论是明示或默示的任何种类的保证或条件。有关许可下的特定语言，请参阅许可。

⚠️请注意，此文件采用Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），可能无法在您的Markdown查看器中正确呈现。

-->

# PoolFormer

## 概述

PoolFormer模型在Sea AI Labs的论文[MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)中提出。与设计复杂的标记混合器以实现SOTA性能不同，该工作的目标是展示transformer模型的能力主要源于通用架构MetaFormer。

论文摘要如下：

*Transformer在计算机视觉任务中展示了巨大的潜力。一种共识是它们基于注意力的标记混合器模块对模型的能力做出了最大贡献。然而，最近的研究表明，transformer中的基于注意力的模块可以被空间MLP替代，而得到的模型仍然表现出色。基于这一观察，我们假设transformer的一般架构，而不是特定的标记混合器模块，对模型的性能更为关键。为了验证这一点，我们故意用尴尬的简单空间池化算子替换transformer中的注意力模块，以进行最基本的标记混合。令人惊讶的是，我们观察到得到的模型（称为PoolFormer）在多个计算机视觉任务上都取得了竞争性的性能。例如，在ImageNet-1K上，PoolFormer实现了82.1%的top-1准确率，比经过精调的vision transformer/MLP-like基线DeiT-B/ResMLP-B24分别提高了0.3%/1.1%，参数减少了35%/52%，MAC减少了48%/60%。PoolFormer的有效性验证了我们的假设，并促使我们提出了“MetaFormer”的概念，这是从transformer中抽象出来的通用架构，而不限定于标记混合器。基于广泛的实验，我们认为MetaFormer是实现最近transformer和MLP-like模型在视觉任务上取得优越结果的关键因素。这项工作呼吁未来开展更多研究来改进MetaFormer，而不是专注于标记混合器模块。此外，我们提出的PoolFormer可能成为未来MetaFormer架构设计的起点基准。*

下图显示了PoolFormer的架构。来自[原论文](https://arxiv.org/abs/2111.11418)。

<img width="600" src="https://user-images.githubusercontent.com/15921929/142746124-1ab7635d-2536-4a0e-ad43-b4fe2c5a525d.png"/>

提示：

- PoolFormer具有分层架构，其中使用简单的平均池化层代替Attention。可以在[hub](https://huggingface.co/models?other=poolformer)上找到该模型的所有检查点。
- 可以使用[`PoolFormerImageProcessor`]为模型准备图像。
- 与大多数模型一样，PoolFormer有不同的规模，详细信息如下表所示。

| **模型变体** | **深度**     | **隐藏大小**       | **参数（百万）** | **ImageNet-1k Top 1** |
| :----------: | ------------ | ------------------ | :-------------: | :-------------------: |
|     s12      | [2, 2, 6, 2] | [64, 128, 320, 512] |       12        |         77.2          |
|     s24      | [4, 4, 12, 4]| [64, 128, 320, 512]|       21        |         80.3          |
|     s36      | [6, 6, 18, 6]| [64, 128, 320, 512]|       31        |         81.4          |
|     m36      | [6, 6, 18, 6]| [96, 192, 384, 768]|       56        |         82.1          |
|     m48      | [8, 8, 24, 8]| [96, 192, 384, 768]|       73        |         82.5          |

此模型由[heytanay](https://huggingface.co/heytanay)贡献。原始代码可以在[这里](https://github.com/sail-sg/poolformer)找到。

## 资源

Hugging Face官方和社区（用🌎表示）资源列表，以帮助您开始使用PoolFormer。

<PipelineTag pipeline="image-classification"/>

- [`PoolFormerForImageClassification`]受到此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)的支持。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果您有兴趣提交要包含在此处的资源，请随时打开拉取请求，我们将进行审核！该资源应该展示出一些新的东西，而不是重复现有的资源。

## PoolFormerConfig

[[autodoc]] PoolFormerConfig

## PoolFormerFeatureExtractor

[[autodoc]] PoolFormerFeatureExtractor
    - __call__

## PoolFormerImageProcessor

[[autodoc]] PoolFormerImageProcessor
    - preprocess

## PoolFormerModel

[[autodoc]] PoolFormerModel
    - forward

## PoolFormerForImageClassification

[[autodoc]] PoolFormerForImageClassification
    - forward
