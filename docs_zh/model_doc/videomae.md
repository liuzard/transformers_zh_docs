<!--
版权所有2022年的抱抱脸团队。版权所有。

根据Apache许可证第2.0版（“许可证”）进行许可；除非符合许可证的规定，否则你不得使用此文件。
你可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律另有规定或书面同意，根据许可证分发的软件基于“AS IS”原则，不附带任何明示或隐含的担保或条件。
请注意，该文件采用Markdown格式，但包含适用于我们的文档生成器（类似于MDX）的特定语法，这可能无法在你的Markdown查看器中正确渲染。

-->

# VideoMAE

## 概述

VideoMAE模型是由Zhan Tong, Yibing Song, Jue Wang, Limin Wang在论文《VideoMAE：基于遮罩自编码器的高效学习者用于自监督视频预训练》中提出的[VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)。
VideoMAE将Masked Autoencoders（MAE）扩展到视频领域，在几个视频分类基准测试上声称具有最先进的性能。

论文中的摘要如下：

*在相对较小的数据集上，通常需要在额外大规模数据集上对视频Transformer进行预训练，以实现出色的性能。本文中，我们展示了视频遮罩自编码器（VideoMAE）是用于自监督视频预训练（SSVP）的高效学习者。我们受到最近的ImageMAE的启发，提出了定制的视频管遮罩和重构。这些简单的设计证明能有效地克服由视频重构过程中的时间相关性造成的信息泄露。关于SSVP，我们取得了三个重要的发现：（1）相当高比例的遮罩比例（即，90%到95%）仍然可以产生良好的VideoMAE性能。比起图像，时间冗余的视频内容可实现更高的遮罩比例。（2）VideoMAE在非常小的数据集上（即约3k-4k个视频）取得了令人印象深刻的结果，而无需使用任何额外的数据。这部分归因于视频重构任务的挑战性，以实现高级结构学习。（3）VideoMAE显示了数据质量对于SSVP而言比数据数量更为重要。预训练数据集与目标数据集之间的领域偏移是SSVP中的重要问题。值得注意的是，我们的Vanilla ViT骨干网络的VideoMAE在Kinects-400上可以达到83.9%，在Something-Something V2上可以达到75.3%，在UCF101上可以达到90.8%，在HMDB51上可以达到61.1%，而无需使用任何额外的数据。*

提示：
- 你可以使用[`VideoMAEImageProcessor`]来为模型准备视频。它将为你调整大小和标准化视频的所有帧。
- [`VideoMAEForPreTraining`]包含了自监督预训练模型的解码器。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/videomae_architecture.jpeg"
alt="drawing" width="600"/>

<small> VideoMAE预训练架构。摘自<a href="https://arxiv.org/abs/2203.12602">原始论文</a>。</small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[这里](https://github.com/MCG-NJU/VideoMAE)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）的资源列表，可帮助你开始使用VideoMAE。如果你有兴趣提交资源以包含在此处，请随时提出Pull Request，我们将进行审查！资源应该显示一些新内容，而不是重复现有的资源。

**视频分类**
- [一个笔记本](https://github.com/huggingface/notebooks/blob/main/examples/video_classification.ipynb)，展示如何在自定义数据集上微调VideoMAE模型。
- [视频分类任务指南](../tasks/video-classification)
- [一个🤗空间](https://huggingface.co/spaces/sayakpaul/video-classification-ucf101-subset)，演示如何使用视频分类模型进行推理。


## VideoMAEConfig

[[autodoc]] VideoMAEConfig

## VideoMAEFeatureExtractor

[[autodoc]] VideoMAEFeatureExtractor
    - __call__

## VideoMAEImageProcessor

[[autodoc]] VideoMAEImageProcessor
    - preprocess

## VideoMAEModel

[[autodoc]] VideoMAEModel
    - forward

## VideoMAEForPreTraining

[[autodoc]] transformers.VideoMAEForPreTraining
    - forward

## VideoMAEForVideoClassification

[[autodoc]] transformers.VideoMAEForVideoClassification
    - forward
-->

# VideoMAE

## 概述

[VideoMAE](https://arxiv.org/abs/2203.12602)模型是由Zhan Tong、Yibing Song、Jue Wang、Limin Wang在论文中提出的[VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)。
VideoMAE将遮罩自编码器（MAE）扩展到视频领域，并声称在几个视频分类基准测试中具有最先进的性能。

这篇论文的摘要如下：

*通常需要在大规模的额外数据集上进行预训练视频Transformer，以在相对较小的数据集上获得最佳性能。本文中，我们展示了视频遮罩自编码器（VideoMAE）是适用于自监督视频预训练（SSVP）的高效学习者。我们受到最近的ImageMAE的启发，提出了定制的视频管遮罩和重构。这些简单的设计被证明对于克服视频重构过程中由时间相关性引起的信息泄漏非常有效。我们在SSVP方面得出了三个重要发现：（1）即使是非常高比例的遮罩比例（即90％到95％），VideoMAE仍可以产生出色的性能。由于视频的时间冗余内容，可以使用更高的遮罩比例。（2）VideoMAE在非常小的数据集上（即大约3k-4k个视频）取得了令人印象深刻的结果，而无需使用任何额外的数据。这部分归因于视频重构任务的挑战性，能够促进高级结构的学习。（3）VideoMAE表明对于SSVP而言，数据质量比数据数量更加重要。预训练和目标数据集之间的领域转变是SSVP中的重要问题。值得注意的是，使用Vanilla ViT骨干网络，即使不使用任何额外的数据，我们的VideoMAE模型在Kinects-400上的准确率达到了83.9％，在Something-Something V2上达到了75.3％，在UCF101上达到了90.8％，在HMDB51上达到了61.1％。*

提示：

- 你可以使用[`VideoMAEImageProcessor`]来为模型准备视频。它会为你调整视频的大小并对每一帧进行标准化。
- [`VideoMAEForPreTraining`]中包含了自监督预训练模型的解码器。

![VideoMAE架构](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/videomae_architecture.jpeg)

<small>VideoMAE预训练架构。摘自<a href="https://arxiv.org/abs/2203.12602">原始论文</a>。</small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[这里](https://github.com/MCG-NJU/VideoMAE)找到。

## 资源

下面是用于帮助你入门VideoMAE的官方Hugging Face资源和社区资源（由🌎表示）。如果你有兴趣提交资源以包含在这里，请随时打开一个Pull Request，我们将会进行审查！资源应该展示一些新的内容，而不是重复现有的资源。

**视频分类**
- [一个笔记本](https://github.com/huggingface/notebooks/blob/main/examples/video_classification.ipynb)展示如何在定制的数据集上微调VideoMAE模型。
- [视频分类任务指南](../tasks/video-classification)
- [一个🤗Space](https://huggingface.co/spaces/sayakpaul/video-classification-ucf101-subset)展示如何使用视频分类模型进行推理。

## VideoMAEConfig

[[autodoc]] VideoMAEConfig

## VideoMAEFeatureExtractor

[[autodoc]] VideoMAEFeatureExtractor
    - __call__

## VideoMAEImageProcessor

[[autodoc]] VideoMAEImageProcessor
    - preprocess

## VideoMAEModel

[[autodoc]] VideoMAEModel
    - forward

## VideoMAEForPreTraining

[[autodoc]] transformers.VideoMAEForPreTraining
    - forward

## VideoMAEForVideoClassification

[[autodoc]] transformers.VideoMAEForVideoClassification
    - forward