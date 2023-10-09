<!--版权2022年HuggingFace团队。版权所有。

根据Apache许可证2.0版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下地址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，没有任何明示或暗示的保证或条件。有关详细信息，请参阅许可证。

⚠️ 请注意，此文件是Markdown格式，但包含对doc-builder（类似于MDX）的特定语法，可能在Markdown查看器中无法正确渲染。

-->

# Swin Transformer V2

## 概述

Swin Transformer V2模型是由Ze Liu，Han Hu，Yutong Lin，Zhuliang Yao，Zhenda Xie，Yixuan Wei，Jia Ning，Yue Cao，Zheng Zhang，Li Dong，Furu Wei和Baining Guo在[Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)一文中提出的。

该论文的摘要如下：

*大规模的自然语言处理（NLP）模型已经显示出在语言任务上显著提高性能且没有饱和的迹象。它们还展示了与人类一样令人惊讶的一次性能力。本文旨在探索大规模计算机视觉模型。我们解决了培训和应用大视觉模型中的三个主要问题，包括训练的不稳定性、预训练和微调之间的分辨率差距以及对标记数据的需求。提出了三种主要技术：1）将残差后归一化方法与余弦注意力相结合，以改善训练稳定性；2）一种对数间隔连续位置偏置方法，可以将使用低分辨率图像预训练的模型有效地转移到具有高分辨率输入的下游任务中；3）一种自监督的预训练方法，SimMIM，以减少对大量标记图像的需求。通过这些技术，本文成功地训练出了一个30亿参数的Swin Transformer V2模型，这是迄今为止最大的密集视觉模型，并使其能够处理高达1536×1536分辨率的图像。它在四个代表性的视觉任务中取得了新的性能记录，包括ImageNet-V2图像分类、COCO目标检测、ADE20K语义分割和Kinetics-400视频动作分类。还请注意，与Google的十亿级视觉模型相比，我们的训练效率要高得多，消耗的标记数据少40倍，训练时间少40倍。*

提示：
- 您可以使用[`AutoImageProcessor`] API来为模型准备图像。

此模型由[nandwalritik](https://huggingface.co/nandwalritik)贡献。
原始代码可在[此处](https://github.com/microsoft/Swin-Transformer)找到。

## 资源

以下是官方Hugging Face和社区（通过🌎表示）提供的资源列表，帮助您开始使用Swin Transformer v2。

<PipelineTag pipeline="image-classification"/>

- 本 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 支持[`Swinv2ForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)。

除此之外：

- 本 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining) 支持[`Swinv2ForMaskedImageModeling`]。

如果您有兴趣提交资源以包含在此处，请随时打开Pull Request，我们会进行审查！资源应该展示一些新的东西，而不是重复现有的资源。

## Swinv2Config

[[autodoc]] Swinv2Config

## Swinv2Model

[[autodoc]] Swinv2Model
    - forward

## Swinv2ForMaskedImageModeling

[[autodoc]] Swinv2ForMaskedImageModeling
    - forward

## Swinv2ForImageClassification

[[autodoc]] transformers.Swinv2ForImageClassification
    - forward