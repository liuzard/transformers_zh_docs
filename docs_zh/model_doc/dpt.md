<!--版权所有 © 2022年 HuggingFace团队。保留所有权利。

根据Apache许可证2.0版本（“许可证”）发布。除非符合许可证的规定，否则您不得使用此文件。
您可以在以下链接处获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

注意，此文件采用Markdown格式，但包含我们的文档构建器（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正确显示。

-->

# DPT

## 概述

DPT模型是由René Ranftl、Alexey Bochkovskiy和Vladlen Koltun在[《基于Vision Transformer的密集预测》](https://arxiv.org/abs/2103.13413)一文中提出的。
DPT模型是一种利用[Vision Transformer（ViT）](vit)作为密集预测任务（如语义分割和深度估计）的骨干网络的模型。

文章的摘要如下所示：

*我们引入了密集视觉Transformer（dense vision transformers），这种架构将视觉Transformer作为密集预测任务的骨干网络，而不是卷积网络。我们将来自视觉Transformer各个阶段的token组装成具有不同分辨率的类似图像的表示，并使用卷积解码器逐渐将它们组合成全分辨率的预测。Transformer骨干网络以恒定且相对较高的分辨率处理表示，并在每个阶段都具有全局感受野。这些特性使得密集视觉Transformer在细粒度和全局一致性方面相较于完全卷积网络提供了更好的预测。我们的实验结果表明，这种架构在密集预测任务上取得了显著的改进，尤其是在具有大量训练数据的情况下。对于单目深度估计，相较于最先进的完全卷积网络，我们观察到相对性能提高了28%。应用于语义分割时，密集视觉Transformer在ADE20K上实现了49.02%的mIoU，创造了新的state-of-the-art。我们进一步展示了该架构可以在较小的数据集（如NYUv2、KITTI和Pascal Context）上进行微调，并在这些数据集上实现了新的state-of-the-art。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dpt_architecture.jpg"
alt="drawing" width="600"/>

<small> DPT架构。取自<a href="https://arxiv.org/abs/2103.13413" target="_blank">原始文章</a>。 </small>

此模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可在[此处](https://github.com/isl-org/DPT)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）提供的资源列表，可帮助您开始使用DPT。

- [`DPTForDepthEstimation`]的演示笔记本可在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DPT)找到。

- [语义分割任务指南](../tasks/semantic_segmentation)
- [单目深度估计任务指南](../tasks/monocular_depth_estimation)

如果您有兴趣提交资源以包含在此处，请随时发起拉取请求，我们会进行审核！资源应该展示出新的东西，而不是重复现有的资源。

## DPTConfig

[[autodoc]] DPTConfig

## DPTFeatureExtractor

[[autodoc]] DPTFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## DPTImageProcessor

[[autodoc]] DPTImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## DPTModel

[[autodoc]] DPTModel
    - forward

## DPTForDepthEstimation

[[autodoc]] DPTForDepthEstimation
    - forward

## DPTForSemanticSegmentation

[[autodoc]] DPTForSemanticSegmentation