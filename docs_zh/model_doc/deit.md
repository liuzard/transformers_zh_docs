<!--
版权 © 2021 HuggingFace 团队。保留所有权利。

根据 Apache 许可证，版本 2.0（"许可证"），除非符合许可证规定，
否则不能使用此文件。你可以在以下网址获取该许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按
"原样" BASIS，无任何明示或暗示的保证或条件。
有关许可证下的特定语言的权限和限制，请参阅许可证。

⚠️ 请注意，该文件是 Markdown 格式，但包含特定的语法，适用于我们的文档构建工具（类似于 MDX），
可能在你的 Markdown 查看器中呈现不正确。

-->

# DeiT

<Tip>

这是最近提出的一个模型，所以 API 还未经过全面测试。将来可能会有一些错误或轻微的
变动。如果发现出现异常的情况，请[提交 Github 问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

</Tip>

## 概述

DeiT 模型出自 Hugo Touvron 等人的论文[《通过注意力进行高效的图像转换与蒸馏》](https://arxiv.org/abs/2012.12877)。DeiT
模型是通过 Transformer 编码器（类似于 BERT）来实现图像分类任务的。论文中介绍的[视觉 Transformer (ViT)](vit)模型表明，使用 Transformer
编码器可与现有的卷积神经网络相匹配或超越其性能。然而，论文中介绍的 ViT 模型需要在昂贵的基础设施上进行多周训练，
并使用外部数据。相比之下，DeiT（数据高效图像转换）模型更高效，能够使用较少的数据和计算资源进行图像分类训练。

该论文的摘要如下：

*最近，仅基于注意力的神经网络已经能够解决诸如图像分类之类的图像理解任务。
然而，这些视觉 Transformer 的预训练需要使用数亿张图片，并使用昂贵的基础设施进行，
从而限制了它们的应用。在本研究中，我们通过仅在 ImageNet 上进行训练，构建了一个具有竞争力的无卷积
transfomer 模型。我们通过在一台计算机上进行少于 3 天的训练，完成了模型的训练。我们的参考视觉
transformer（8600 万个参数）在 ImageNet 上的 top-1 准确率达到 83.1%（单剪裁评估），
并且不使用任何外部数据。更重要的是，我们引入了一种特定于 Transformer 的教师-学生策略。
它依赖于蒸馏令牌，确保学生通过注意力从教师那里学习。我们展示了这种基于令牌的蒸馏的优点，
特别是在使用卷积神经网络作为教师时。这使我们能够报告与卷积神经网络在
ImageNet 上（我们获得的准确率高达 85.2%）以及在转移学习到其他任务时相竞争的结果。我们分享我们的代码和模型。*

提示：

- 与 ViT 模型相比，DeiT 模型使用所谓的蒸馏令牌来有效地从教师模型（在 DeiT 论文中是 ResNet 风格的模型）学习。
  蒸馏令牌是通过反向传播与类别（[CLS]）和图像块令牌之间的自注意层进行交互来学习的。
- 对于蒸馏模型的微调，有两种方式：（1）经典方式，只在类别令牌的最终隐藏状态上放置一个预测头，不使用蒸馏信号；
  或者（2）在类别令牌和蒸馏令牌上全部放置预测头。在第二种方式中，[CLS] 预测头使用预测结果和真实标签之间的
  交叉熵进行训练，而蒸馏预测头使用硬蒸馏进行训练（即教师模型预测结果和蒸馏模型预测结果之间的交叉熵）。
  在推理阶段，通过取两个预测头的平均结果作为最终预测。第二种方式也称为“蒸馏微调”，因为它依赖于教师模型已经在下游数据集上经过微调。
  在模型方面，（1）对应于[`DeiTForImageClassification`]，（2）对应于[`DeiTForImageClassificationWithTeacher`]。
- 值得注意的是，作者还尝试了对于（2）而言的软蒸馏（蒸馏预测头使用 KL 散度来匹配教师预测结果的 softmax 输出），
  但发现硬蒸馏效果更好。
- 所有发布的检查点仅在 ImageNet-1k 上进行了预训练和微调。没有使用外部数据。
  这与原始的 ViT 模型不同，后者使用了外部数据，如 JFT-300M 数据集/Imagenet-21k 进行预训练。
- DeiT 的作者还发布了效率更高的 ViT 模型，你可以直接将其插入到[`ViTModel`]或[`ViTForImageClassification`]中。
  为了模拟在更大数据集上训练的效果（但仅使用 ImageNet-1k 进行预训练），使用了数据增强、优化和正则化等技术。
  有 4 个变种可供选择（3 种不同大小）：*facebook/deit-tiny-patch16-224*、*facebook/deit-small-patch16-224*、
  *facebook/deit-base-patch16-224* 和 *facebook/deit-base-patch16-384*。请注意，你应该使用[`DeiTImageProcessor`]来准备模型输入图像。

该模型由 [nielsr](https://huggingface.co/nielsr) 贡献。该模型的 TensorFlow 版本由 [amyeroberts](https://huggingface.co/amyeroberts) 添加。

## 资源

此处列出了官方 Hugging Face 资源和社区 (🌎) 资源，以帮助你开始使用 DeiT。

<PipelineTag pipeline="image-classification"/>

- 你可以使用此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和
  [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)来支持[`DeiTForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

除此之外：

- 你可以使用此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)来支持[`DeiTForMaskedImageModeling`]。

如果你有兴趣提交资源以包含在此处，请随时打开一份 Pull Request，我们将进行审查！
该资源应该理想地展示一些新内容，而不是重复现有资源。

## DeiTConfig

[[autodoc]] DeiTConfig

## DeiTFeatureExtractor

[[autodoc]] DeiTFeatureExtractor
    - __call__

## DeiTImageProcessor

[[autodoc]] DeiTImageProcessor
    - preprocess

## DeiTModel

[[autodoc]] DeiTModel
    - forward

## DeiTForMaskedImageModeling

[[autodoc]] DeiTForMaskedImageModeling
    - forward

## DeiTForImageClassification

[[autodoc]] DeiTForImageClassification
    - forward

## DeiTForImageClassificationWithTeacher

[[autodoc]] DeiTForImageClassificationWithTeacher
    - forward

## TFDeiTModel

[[autodoc]] TFDeiTModel
    - call

## TFDeiTForMaskedImageModeling

[[autodoc]] TFDeiTForMaskedImageModeling
    - call

## TFDeiTForImageClassification

[[autodoc]] TFDeiTForImageClassification
    - call

## TFDeiTForImageClassificationWithTeacher

[[autodoc]] TFDeiTForImageClassificationWithTeacher
    - call
-->