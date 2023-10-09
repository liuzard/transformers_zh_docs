<!--版权所有 © 2022 HuggingFace 团队保留所有权利。

根据 Apache 许可证第2版（"许可证"），除非符合许可证的规定，否则你不得使用此文件。
你可以在以下位置获得许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，按原样分发的软件在"AS IS"的基础上提供，无论是明示的还是暗示的，不附带任何形式的保证或条件。请查阅许可证以获取特定语言下的权限和限制。

注意，此文件是Markdown格式，但它包含的语法是为我们的文档生成器（类似于MDX）设计的，可能无法在你的Markdown阅读器中正确渲染。-->

# Mask2Former

## 概述

Mask2Former模型是由Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar在论文[Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)中提出的。Mask2Former是一个统一的框架，用于全景分割、实例分割和语义分割，并在性能和效率上显著提高了[MaskFormer](maskformer)。

论文中的摘要如下：

*图像分割将具有不同语义（例如类别或实例成员资格）的像素分组。每个语义选择都定义了一个任务。虽然每个任务的语义不同，但当前的研究集中于为每个任务设计专门的体系结构。我们提出的Masked-attention Mask Transformer（Mask2Former）是一种新的体系结构，能够处理任何图像分割任务（全景、实例或语义）。其关键组件包括掩膜注意力，通过在预测的掩膜区域内限制交叉注意力来提取局部特征。除了将研究工作量减少至少三倍外，它在四个流行数据集上的性能大大优于最佳专门体系结构。值得注意的是，Mask2Former在全景分割（COCO上的57.8 PQ），实例分割（COCO上的50.1 AP）和语义分割（ADE20K上的57.7 mIoU）方面取得了新的最优结果。*

提示：
- Mask2Former使用与[MaskFormer](maskformer)相同的预处理和后处理步骤。可以使用[`Mask2FormerImageProcessor`]或[`AutoImageProcessor`]来准备模型的图像和可选的目标。
- 要获取最终的分割结果，根据任务的不同，可以调用[`~Mask2FormerImageProcessor.post_process_semantic_segmentation`]、[`~Mask2FormerImageProcessor.post_process_instance_segmentation`]或[`~Mask2FormerImageProcessor.post_process_panoptic_segmentation`]。这三个任务都可以使用[`Mask2FormerForUniversalSegmentation`]输出进行解决，全景分割还可以接受一个可选的`label_ids_to_fuse`参数，将目标对象（例如天空）的实例合并在一起。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mask2former_architecture.jpg" alt="drawing" width="600"/>

<small> Mask2Former体系结构。摘自[原论文](https://arxiv.org/abs/2112.01527)。 </small>

此模型由[Shivalika Singh](https://huggingface.co/shivi)和[Alara Dirik](https://huggingface.co/adirik)贡献。原始代码可在[这里](https://github.com/facebookresearch/Mask2Former)找到。

## 资源

用于开始使用Mask2Former的官方Hugging Face和社区（🌎表示）资源列表。

- 有关在自定义数据上进行推断+微调Mask2Former的演示笔记本可在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Mask2Former)找到。

如果你有兴趣提交一个要包含在这里的资源，请随时发起拉取请求，我们将进行审核。
这些资源理想情况下应该展示一些新的东西，而不是重复现有的资源。

## MaskFormer特定输出

[[autodoc]] models.mask2former.modeling_mask2former.Mask2FormerModelOutput

[[autodoc]] models.mask2former.modeling_mask2former.Mask2FormerForUniversalSegmentationOutput

## Mask2FormerConfig

[[autodoc]] Mask2FormerConfig

## Mask2FormerModel

[[autodoc]] Mask2FormerModel
    - forward

## Mask2FormerForUniversalSegmentation

[[autodoc]] Mask2FormerForUniversalSegmentation
    - forward

## Mask2FormerImageProcessor

[[autodoc]] Mask2FormerImageProcessor
    - preprocess
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation