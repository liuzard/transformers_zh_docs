<!--
版权所有2021年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）的规定，你不得使用此文件，除非符合
许可证的要求。你可以在下面获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是根据
“按原样提供”的基础上，没有任何明示或暗示的保证或条件。请参阅许可证的规定
特定语言管理的种类（不管是明示还是暗示）视图转换器，具有相当不错的结果
与熟悉的卷积架构相比，模型在ImageNet上成功地训练Transformer编码器。


论文摘要如下：

尽管Transformer架构已成为自然语言处理任务的事实标准，但其
对计算机视觉的应用仍然有限。在视觉中，注意力通常与
卷积网络一起使用，或者用于替换卷积网络的某些组件同时保持其整体
结构不变。我们表明，这种对卷积网络的依赖并非必需，直接应用纯变压器到
图像块序列可以在图像分类任务上表现出很好的性能。在大量预先训练的情况下
数据，并转移到多个中型或小型图像识别基准（ImageNet，CIFAR-100，VTAB等）上，
与最先进的卷积网络相比，Vision Transformer（ViT）在性能上取得了出色的结果
的资源要求低得多。


提示：

- 有关推断以及在自定义数据上微调ViT的演示笔记本可以在[此处找到](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer)。
- 为了将图像馈送到Transformer编码器中，将每个图像分成一系列固定大小且没有重叠的块，
  然后进行线性嵌入。添加了[CLS]令牌以用作整个图像的表示，可用于分类。作者还添加了绝对位置嵌入，
  并将结果向量序列馈送到标准的Transformer编码器。
- 由于Vision Transformer期望每个图像的大小（分辨率）相同，因此可以使用
  [`ViTImageProcessor`]来为模型调整图像的大小（或重新缩放）并进行规范化。
- 在预训练或微调期间使用的块分辨率和图像分辨率反映在
  每个检查点的名称中。例如，`google/vit-base-patch16-224`是指块为
  16x16的基本架构和224x224的微调分辨率。所有检查点都可以在[hub](https://huggingface.co/models?search=vit)上找到。
- 可用的检查点要么是仅在[ImageNet-21k](http://www.image-net.org/)（一个收集），上进行预训练
  了1400万张图像和21000个类），要么还在[ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) （也称为ILSVRC 2012，一个收集）上进行了微调。
  130万张图像和1000个类）。
- Vision Transformer的预训练分辨率为224x224。在微调过程中，常常会受益于使用比预训练更高的分辨率[(Touvron et al。，2019)](https://arxiv.org/abs/1906.06423)，[(Kolesnikov
  et al。，2020)](https://arxiv.org/abs/1912.11370)。为了在更高的分辨率下进行微调，作者对已预训练的位置嵌入执行二维插值，根据其在原始图像中的位置。
- 最佳结果是通过监督预训练获得的，这在NLP中并非如此。作者还进行了一次实验
  采用自我监督的预训练目标，即蒙版斑块预测（受蒙版启发）
  语言建模）。通过这种方法，较小的ViT-B/16模型在ImageNet上达到了79.9％的准确性，这比从头开始训练的准确性提高了2％，但仍然比监督预训练差4％。

![图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg)

<small> Vision Transformer架构。来自于<a href="https://arxiv.org/abs/2010.11929">原始论文。</a> </small>

随着原始Vision Transformer的成功，出现了一些后续作品：

- 由Facebook AI提供的[DeiT](deit）（数据高效图像转换）。DeiT模型是蒸馏视觉变换器。
  DeiT的作者还发布了训练效率更高的ViT模型，你可以直接插入[`ViTModel`] 或
  [`ViTForImageClassification`]。有4个变体可用（3种不同大小）：*facebook/deit-tiny-patch16-224*,
  *facebook/deit-small-patch16-224*, *facebook/deit-base-patch16-224*和 *facebook/deit-base-patch16-384*。请注意
  使用[`DeiTImageProcessor`]以为模型准备图像。

- Microsoft Research的[BEiT](beit]（基于BERT的视觉图像变换的BERT预训练）。 BEiT模型优于监督预训练的
  使用受BERT（蒙面图像建模）启发的自我监督方法和VQ-VAE的基础的变换器。

- Facebook AI的DINO（自我监督训练视觉转换器的方法）。使用DINO方法训练的视觉变换器显示出与卷积模型不同的非常有趣的属性。他们能够分割
  对象，而不必经过训练以执行此操作。DINO检查点可以在[hub](https://huggingface.co/models?other=dino)上找到。

- Facebook AI的[MAE](vit_mae)（蒙版自动编码器）。通过预训练Vision Transformer重构高比例（75％）的遮罩补丁的像素值（使用非对称编码器-解码器架构），作者表明，这种简单方法在微调后胜过了监督预训练。

这个模型是由[nielsr](https://huggingface.co/nielsr)贡献的。该原始代码（用JAX编写）可以
在[这里](https://github.com/google-research/vision_transformer)找到。
我们将权重从Ross Wightman的[timm库](https://github.com/rwightman/pytorch-image-models)转换为PyTorch，他已经将权重从JAX转换为PyTorch。积分
归于他！

## 资源

官方Hugging Face和社区（由🌎表示）资源列表，可帮助你开始使用ViT。

<PipelineTag pipeline="image-classification"/>

- 由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和这个[notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持[`ViTForImageClassification`]。
- 关于在自定义数据集上微调[`ViTForImageClassification`]的博客文章可以在[这里](https://huggingface.co/blog/fine-tune-vit)找到。
- 可以找到更多有关微调[`ViTForImageClassification`]的演示笔记本[在这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer)。
- [图像分类任务指南](../tasks/image_classification)

除此之外：

- 由以下资源支持[`ViTForMaskedImageModeling`]：

    - [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)。

    - [示例代码](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)。
    
如果你有兴趣提交一个资源以包含在此处，请随时打开一个Pull Request，我们将检查它！资源最好能演示一些新东西，而不是重复现有资源。

## ViTConfig

[[autodoc]] ViTConfig

## ViTFeatureExtractor

[[autodoc]] ViTFeatureExtractor
    - __call__

## ViTImageProcessor

[[autodoc]] ViTImageProcessor
    - preprocess

## ViTModel

[[autodoc]] ViTModel
    - forward

## ViTForMaskedImageModeling

[[autodoc]] ViTForMaskedImageModeling
    - forward

## ViTForImageClassification

[[autodoc]] ViTForImageClassification
    - forward

## TFViTModel

[[autodoc]] TFViTModel
    - call

## TFViTForImageClassification

[[autodoc]] TFViTForImageClassification
    - call

## FlaxVitModel

[[autodoc]] FlaxViTModel
    - __call__

## FlaxViTForImageClassification

[[autodoc]] FlaxViTForImageClassification
    - __call__
-->