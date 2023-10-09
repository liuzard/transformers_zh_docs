<!--版权所有2021年The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可，除非遵守许可证，否则不得使用此文件。你可以在以下位置获取许可证的副本http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证的条款分发，
<dl>”基于“AS IS” BASIS的，没有明示或暗示的任何保证或条件，
包括但不限于关于特定用途无侵权，权利或适销性的保证或条件
以及具体规定或与许可证附带的限制有关的法律文件。

⚠️请注意，此文件为Markdown格式，但包含特定于我们的文档生成器（类似于MDX）的语法，可能无法在Markdown查看器中正确呈现。-->

# BEiT

## 概述

BEiT模型是由Hangbo Bao，Li Dong和Furu Wei在[BEiT：BERT图像变压器的预训练](https://arxiv.org/abs/2106.08254)提出的。受BERT的启发，BEiT是第一篇通过自监督预训练使自我预训练的图像变压器（ViTs）胜过监督预训练的论文。BEiT模型不是通过预训练来预测图像的类别（如[原始ViT论文](https://arxiv.org/abs/2010.11929)中所做的那样），而是通过预训练来预测OpenAI的[DALL-E模型](https://arxiv.org/abs/2102.12092)的代码本中的视觉标记。

该论文的摘要如下：

*我们引入一种自监督视觉表示模型BEiT，代表从图像变压器(BERT)的编码器的双向编码器表示。如同自然语言处理领域中开发的BERT一样，我们提出了一个掩码图像建模任务，用于对视觉变换进行预训练。具体而言，我们的预训练图像有两个视图，即图像块（例如16×16像素）和视觉标记（即离散标记）。我们首先将原始图像“标记化”为视觉标记。然后，我们随机对一些图像块进行掩码处理，并将它们输入到骨干变换器中。预训练目标是根据受损图像块恢复原始的视觉标记。在BEiT预训练之后，我们通过附加到预训练编码器上的任务层直接微调模型参数。图像分类和语义分割的实验结果表明，我们的模型与之前的预训练方法取得了竞争性的结果，例如基础大小的BEiT在ImageNet-1K上达到了83.2％的top-1准确率，显著优于使用相同设置的DeiT训练（81.8%）。此外，使用ImageNet-1K，大型BEiT仅获得了86.3%，甚至优于在ImageNet-22K上进行了监督预训练的ViT-L（85.2％）。*

提示：

- BEiT模型是常规的视觉变压器，但采用自监督方式进行预训练，而不是监督方式。当在ImageNet-1K和CIFAR-100上进行微调时，它们的性能优于[原始模型（ViT）](vit)和[Data-efficient Image Transformers (DeiT)](deit)。你可以在这里查看有关推理以及在自定义数据上进行微调的演示笔记本[（此处）](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer)（你可以使用[`BeitImageProcessor`]替换[`ViTFeatureExtractor`]以及[`ViTForImageClassification`]替换[`BeitForImageClassification`]）。
- 这里还有一个演示笔记本，展示了如何结合DALL-E的图像标记器和BEiT来执行掩码图像建模。你可以在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/BEiT)找到它。
- 由于BEiT模型期望每个图像具有相同的尺寸（分辨率），因此可以使用[`BeitImageProcessor`]对模型的图像进行调整（或重新缩放）和归一化。
- 预训练或微调期间使用的块分辨率和图像分辨率反映在每个检查点的名称中。例如，`microsoft/beit-base-patch16-224`指的是具有16x16块分辨率和224x224微调分辨率的基本大小架构。所有检查点都可以在[hub](https://huggingface.co/models?search=microsoft/beit)上找到。
- 可用的检查点要么（1）仅在[ImageNet-22k](http://www.image-net.org/)（包含1400万图像和22k类别）上进行了预训练，要么（2）在ImageNet-22k上进行了微调，或者（3）在[ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/)上进行了微调（也称为ILSVRC 2012，包含130万图像和1000个类别）。
- BEiT使用相对位置嵌入，受T5模型的启发。在预训练期间，作者在几个自注意层之间共享了相对位置偏差。在微调期间，每层的相对位置偏差是使用预训练后获得的共享相对位置偏差进行初始化的。请注意，如果要从头开始预训练模型，则需要将[`BeitConfig`]的`use_relative_position_bias`或`use_relative_position_bias`属性设置为`True`以添加位置嵌入。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/beit_architecture.jpg"
alt="drawing" width="600"/>

<small> BEiT预训练。来源：[原始论文](https://arxiv.org/abs/2106.08254)。</small>

此模型由[nielsr](https://huggingface.co/nielsr)贡献。这个模型的JAX/FLAX版本是由[kamalkraj](https://huggingface.co/kamalkraj)贡献的。原始代码可以在[这里](https://github.com/microsoft/unilm/tree/master/beit)找到。

## 资源

以下是官方Hugging Face和社区（用🌎表示）的资源列表，可帮助你开始使用BEiT。

<PipelineTag pipeline="image-classification"/>

- 本[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持[`BeitForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

**语义分割**
- [语义分割任务指南](../tasks/semantic_segmentation)

如果你有兴趣提交资源以包含在此处，请随时发起拉取请求，我们将进行审核！该资源应该最好展示新东西，而不是重复现有资源。

## BEiT特定输出

[[autodoc]] models.beit.modeling_beit.BeitModelOutputWithPooling

[[autodoc]] models.beit.modeling_flax_beit.FlaxBeitModelOutputWithPooling

## BeitConfig

[[autodoc]] BeitConfig

## BeitFeatureExtractor

[[autodoc]] BeitFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## BeitImageProcessor

[[autodoc]] BeitImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## BeitModel

[[autodoc]] BeitModel
    - forward

## BeitForMaskedImageModeling

[[autodoc]] BeitForMaskedImageModeling
    - forward

## BeitForImageClassification

[[autodoc]] BeitForImageClassification
    - forward

## BeitForSemanticSegmentation

[[autodoc]] BeitForSemanticSegmentation
    - forward

## FlaxBeitModel

[[autodoc]] FlaxBeitModel
    - __call__

## FlaxBeitForMaskedImageModeling

[[autodoc]] FlaxBeitForMaskedImageModeling
    - __call__

## FlaxBeitForImageClassification

[[autodoc]] FlaxBeitForImageClassification
    - __call__