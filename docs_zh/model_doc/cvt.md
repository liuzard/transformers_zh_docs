<!--版权 2022 年 HuggingFace 团队。保留所有权利。

根据 Apache 许可证第 2.0 版（以下简称“许可证”）许可；除非符合许可证的规定，否则你不得使用此文件。
你可以在以下网址获得许可证的副本：

http://www.apache.orglicenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于“原样” 的基础上分发，没有任何明示或暗示的担保或条件。
详见许可证中关于特定语言许可的限制和禁止。

⚠️ 请注意，此文件的格式为 Markdown，但包含我们的文档构建器（类似于 MDX）的特定语法，
这可能无法在你的 Markdown 查看器中正确呈现。-->

# 卷积视觉变压器（CvT）

## 概览

CvT 模型由 Haiping Wu、Bin Xiao、Noel Codella、Mengchen Liu、Xiyang Dai、Lu Yuan 和 Lei Zhang 在 [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808) 中提出。卷积视觉变压器（CvT）通过将卷积引入 ViT，以在性能和效率上改进 [视觉变压器（ViT）](vit) 来实现这两种设计的最佳结果。

来自论文的摘要如下：

*我们在本文中提出了一种新的架构，名为卷积视觉变压器（CvT），通过将卷积引入 ViT，以在性能和效率上改进视觉变压器（ViT）。这通过两个主要修改来实现：包含新卷积令牌嵌入的变压器层次结构，以及利用卷积投影的卷积变压器块。这些修改引入了卷积神经网络（CNN）对 ViT 架构的良好特性（即位移、缩放和扭曲不变性），同时保留了变压器的优势（即动态注意、全局上下文和更好的泛化能力）。我们通过进行大量实验证实了 CvT 的有效性，并展示了这种方法在 ImageNet-1k 上达到了其他视觉变压器和 ResNets 的最新性能水平，同时参数更少，FLOP 更低。此外，在对较大的数据集（如 ImageNet-22k ）进行预训练和下游任务的微调时，性能提升得以保持。我们的 CvT-W24 在 ImageNet-1k 验证集上获得了 87.7\% 的 top-1 准确率。最后，我们的结果表明，在现有的视觉变压器中，位置编码是一个至关重要的组成部分，可以在我们的模型中安全地删除，从而简化更高分辨率的视觉任务的设计。*

提示：

- CvT 模型是常规的视觉变压器，但是训练时使用了卷积。在 ImageNet-1K 和 CIFAR-100 上微调时，它们优于[原始模型（ViT）](vit)。
- 你可以在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer)查看有关推理和自定义数据微调的演示笔记本（你可以将 [`ViTFeatureExtractor`] 替换为 [`AutoImageProcessor`]，并将 [`ViTForImageClassification`] 替换为 [`CvtForImageClassification`]）。
- 可用的检查点要么（1）仅在 [ImageNet-22k](http://www.image-net.org/) 上进行预训练（包含 1,400 万张图像和 22k 个类别），要么（2）在 ImageNet-22k 上进行了微调，要么（3）在 [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/) 上进行了微调（也称为 ILSVRC 2012，包含 130 万张图像和 1,000 个类别）。

该模型由 [anugunj](https://huggingface.co/anugunj) 贡献。原始代码可以在[此处](https://github.com/microsoft/CvT)找到。

## 资源

以下是官方 Hugging Face 和社区（由 🌎 表示）资源列表，可帮助你开始使用 CvT。

<PipelineTag pipeline="image-classification"/>

- 使用此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)，支持 [`CvtForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果你有兴趣提交资源以包含在此处，请随时发起拉取请求，我们将进行审核！资源应该展示出一些新的东西，而不是重复现有的资源。

## CvtConfig

[[autodoc]] CvtConfig

## CvtModel

[[autodoc]] CvtModel
    - forward

## CvtForImageClassification

[[autodoc]] CvtForImageClassification
    - forward

## TFCvtModel

[[autodoc]] TFCvtModel
    - call

## TFCvtForImageClassification

[[autodoc]] TFCvtForImageClassification
    - call