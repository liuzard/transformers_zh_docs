<!--版权 2022 年HuggingFace 团队。版权所有。

在Apache许可证下，版本2.0进行许可（“许可证”）；除遵守本许可外，您不得使用此文件。
您可以在以下位置获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样” BASIS分发的，
不附带任何形式的明示或暗示的保证或条件。有关更多详细信息，请参阅许可证。

⚠ 注意，此文件是Markdown格式，但包含我们的文档构建器的特定语法（类似于MDX），可能无法正确呈现在您的Markdown查看器中。

-->

# LeViT

## 概览

LeViT模型是由Ben Graham，Alaaeldin El-Nouby，Hugo Touvron，Pierre Stock，Armand Joulin，Hervé Jégou，Matthijs Douze在 [LeViT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2104.01136)中提出的。LeViT通过一些架构上的差异（如变换器中具有降低分辨率的激活图和引入关注偏差以整合位置信息）来提高[Vision Transformer(ViT)](vit)的性能和效率。

论文中的摘要如下：

 *我们设计了一系列的图像分类架构，以在高速模式中优化准确性和效率之间的权衡。我们的工作利用了注意力
 基于架构的最新研究成果，在高度并行处理硬件上具有竞争力。我们重新审视了来自广泛的原则
 卷积神经网络的文献，将它们应用于变换器，尤其是激活映射
 具有降低分辨率。我们还引入了注意偏差，一种整合位置信息的新方法
 在视觉变形器中。因此，我们提出了LeVIT：一种用于快速推理图像分类的混合神经网络。
 我们考虑了不同硬件平台上的不同效率度量，以便最好地反映广泛的
 应用场景。我们的广泛实验证实了我们的技术选择，并表明它们适用于大多数架构。总的来说，LeViT在速度/准确性权衡方面表现出色。
 例如，在80％ ImageNet top-1准确率下，LeViT比EfficientNet快了5倍。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/levit_architecture.png"
alt="drawing" width="600"/>

<small>LeViT架构。来自于<a href="https://arxiv.org/abs/2104.01136">原始论文</a>。</small>

提示：

- 与ViT相比，LeViT使用了额外的蒸馏头来有效地从教师（在LeViT论文中是类似于ResNet的模型）中学习。蒸馏头通过在ResNet类似模型的监督下进行反向传播学习。他们还从卷积神经网络中汲取灵感，使用分辨率逐渐降低的激活图来提高效率。
- 对于蒸馏模型的微调有两种方法，要么（1）以传统方式，在最终隐藏状态之上只放置一个预测头，而不使用蒸馏头，要么（2）在最终隐藏状态之上同时放置预测头和蒸馏头。在这种情况下，预测头使用正常交叉熵训练来预测头的预测和ground-truth标签之间的交叉熵，而蒸馏预测头使用硬蒸馏训练（蒸馏头的预测和教师预测的标签之间的交叉熵）。在推理时，取两个头之间的平均预测作为最终预测。（2）也被称为“蒸馏微调”，因为它依靠已经在下游数据集上进行了微调的教师。在模型方面，（1）对应于 [`LevitForImageClassification`]，（2）对应于 [`LevitForImageClassificationWithTeacher`]。
- 所有发布的检查点都是在 [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) 上进行的预训练和微调
  (也称为ILSVRC 2012，包含130万张图像和1,000个类别)。没有使用外部数据。这与原始的ViT模型相反，原始模型在预训练中使用了外部数据，如JFT-300M数据集/Imagenet-21k。
- LeViT的作者发布了5个训练好的LeViT模型，您可以直接将其插入[`LevitModel`]或[`LevitForImageClassification`]中。
  为了模拟在更大的数据集上训练（仅使用ImageNet-1k进行预训练），使用了数据增强、优化和正则化等技术。可用的5个变体为（所有的模型都在尺寸为224x224的图像上进行训练）：
  *facebook/levit-128S*，*facebook/levit-128*，*facebook/levit-192*，*facebook/levit-256* 和
  *facebook/levit-384*。请注意，应该使用[`LevitImageProcessor`]来准备模型的图像。
- [`LevitForImageClassificationWithTeacher`]目前仅支持推理，而不支持训练或微调。
- 您可以在这里查看有关推理以及在自定义数据上微调的演示笔记本[here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer)
  (您只需将[`ViTFeatureExtractor`]替换为[`LevitImageProcessor`]，[`ViTForImageClassification`]替换为[`LevitForImageClassification`]或[`LevitForImageClassificationWithTeacher`]）。

该模型由[anugunj](https://huggingface.co/anugunj)贡献。原始代码可以在[这里](https://github.com/facebookresearch/LeViT)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）资源列表，可帮助您开始使用LeViT。

<PipelineTag pipeline="image-classification"/>

- 通过此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持[`LevitForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果您有兴趣提交资源以包含在这里，请随时提出拉取请求，我们将对其进行审查！资源最好能展示一些新内容，而不是重复现有资源。

## LevitConfig

[[autodoc]] LevitConfig

## LevitFeatureExtractor

[[autodoc]] LevitFeatureExtractor
    - __call__

## LevitImageProcessor

[[autodoc]] LevitImageProcessor
    - preprocess

## LevitModel

[[autodoc]] LevitModel
    - forward

## LevitForImageClassification

[[autodoc]] LevitForImageClassification
    - forward

## LevitForImageClassificationWithTeacher

[[autodoc]] LevitForImageClassificationWithTeacher
    - forward