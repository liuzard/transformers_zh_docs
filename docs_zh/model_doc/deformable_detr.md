<!--版权所有2022 The HuggingFace Team。保留所有权利。

根据Apache License，Version 2.0（“许可证”）获得许可；除非遵守许可，否则不得使用此文件。您可以在以下位置获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基础上，“按原样”分发，不附带任何明示或暗示的保证或条件。有关许可的特定语言，请参阅许可的特定语言。

⚠请注意，此文件使用Markdown格式，但包含我们文档生成器（类似于MDX）的特定语法，您的Markdown查看器可能无法正确呈现。

-->

# 可变形DETR

## 概述

《可变形DETR: 可变形Transformer实现端到端目标检测》是由朱锡洲、苏伟杰、鲁乐伟、李斌、王晓刚、代继峰提出的。可变形DETR通过利用一种新的可变形注意模块，仅注意参考周围的一小组关键采样点，解决了原始[DETR](detr)在收敛速度慢、特征空间分辨率有限等问题。

论文中的摘要如下：

*DETR是最近提出的一种消除了目标检测中许多手工设计组件的模型，但由于Transformer注意力模块在处理图像特征图时的局限性，它存在收敛速度慢和特征空间分辨率有限的问题。为了解决这些问题，我们提出了可变形DETR，其注意力模块只关注参考周围的一小组关键采样点。可变形DETR在比DETR（尤其是对于小目标）用10倍较少训练时期实现了更好的性能。在COCO基准上的广泛实验证明了我们方法的有效性。*

提示：

- 您可以使用[`DeformableDetrImageProcessor`]来为模型准备图像（和可选的目标）。
- 对可变形DETR的训练相当于对原始[DETR](detr)模型的训练。有关演示笔记本，请参阅下面的[资源](#resources)部分。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deformable_detr_architecture.png"
alt="绘图" width="600"/>

<small> 可变形DETR架构。来自<a href="https://arxiv.org/abs/2010.04159">原始论文</a>。</small>

这个模型是由[nielsr](https://huggingface.co/nielsr)贡献的。原始代码可以在[这里](https://github.com/fundamentalvision/Deformable-DETR)找到。

## 资源

一个官方的Hugging Face和社区（由🌎表示）的资源列表，可帮助您开始使用可变形DETR。

<PipelineTag pipeline="object-detection"/>

- 有关演练+对自定义数据集进行微调的演示笔记本，可在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Deformable-DETR)找到[`DeformableDetrForObjectDetection`]。
- 参见：[目标检测任务指南](../tasks/object_detection)。

如果您有兴趣提交要包含在此处的资源，请随时提出拉取请求，我们将进行审查！该资源应该展示出一些新东西，而不是重复现有资源。

## DeformableDetrImageProcessor

[[autodoc]] DeformableDetrImageProcessor
    - 预处理
    - 目标检测后处理

## DeformableDetrFeatureExtractor

[[autodoc]] DeformableDetrFeatureExtractor
    - __call__
    - 目标检测后处理

## DeformableDetrConfig

[[autodoc]] DeformableDetrConfig

## DeformableDetrModel

[[autodoc]] DeformableDetrModel
    - forward

## DeformableDetrForObjectDetection

[[autodoc]] DeformableDetrForObjectDetection
    - forward