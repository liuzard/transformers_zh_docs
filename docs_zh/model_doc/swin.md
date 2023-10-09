<!--
版权 2022 The HuggingFace 团队。 保留所有权利。

根据 Apache 许可证 Version 2.0（"许可证"）许可；除非符合许可证，否则不得使用此文件。
您可以在以下处获得许可证副本：
http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按"原样"基础分发的，
不附带任何明示或暗示的担保或条件。有关特定语言的规定，请参阅许可证。
-->

# Swin Transformer

## 概览

Swin Transformer 是由 Ze Liu、Yutong Lin、Yue Cao、Han Hu、Yixuan Wei、Zheng Zhang、Stephen Lin 和 Baining Guo 提出的，
发表在[文章](https://arxiv.org/abs/2103.14030)中，题为 Swin Transformer: Hierarchical Vision Transformer using Shifted Windows。

文章的摘要如下：

*本文提出了一种名为 Swin Transformer 的新形态变阻器，它可以作为计算机视觉的通用骨架。由于两个领域之间存在许多差异，例如视觉实体的尺度变化较大，图像中的像素分辨率与文本中的词汇相比较大。
为了解决这些差异，我们提出了一种使用\bold{S}hifted\bold{win}dows计算表示的分层变压器。位移的窗口方案通过将自注意力计算限制在非重叠的局部窗口中，同时允许跨窗口连接，从而提高了效率。
这种分层结构可以在不同尺度上进行建模，且其计算复杂度与图像大小呈线性关系。
Swin Transformer 的这些特点使其与包括图像分类（ImageNet-1K 上的87.3% top-1 准确度）以及密集预测任务（COCO test-dev 上的58.7 box AP 和 51.1 mask AP、ADE20K val 上的53.5 mIoU）在内的广泛的视觉任务兼容。
其性能大大超过了先前的最新技术，COCO 上的 box AP 和 mask AP 分别增大了 +2.7 和 +2.6，ADE20K 上的 mIoU 增大了 +3.2，显示了基于变压器的模型作为视觉骨干的潜力。
分层设计和位移窗口方法对所有 MLP 架构也具有益处。*

提示：
- 您可以使用 [`AutoImageProcessor`] API 准备模型的图片。
- Swin 补入并支持任何输入高度和宽度（如果是 `32` 的倍数）。
- Swin 可以用作 *骨干网络*。当 `output_hidden_states = True` 时，它将输出 `hidden_states` 和 `reshaped_hidden_states`。`reshaped_hidden_states` 的形状为 `(batch, num_channels, height, width)`，而不是 `(batch_size, sequence_length, num_channels)`。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/swin_transformer_architecture.png"
alt="drawing" width="600"/>

<small> Swin Transformer architecture. Taken from the <a href="https://arxiv.org/abs/2102.03334">original paper</a>.</small>

此模型由[novice03](https://huggingface.co/novice03)贡献。此模型的 Tensorflow 版本由[amyeroberts](https://huggingface.co/amyeroberts)贡献。可以在[这里](https://github.com/microsoft/Swin-Transformer)找到原始代码。


## 资源

以下是官方 Hugging Face 资源以及社区（由 🌎 表示）资源列表，以帮助您开始使用 Swin Transformer。

<PipelineTag pipeline="image-classification"/>

- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和此[notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持 [`SwinForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

除此之外：

- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)支持 [`SwinForMaskedImageModeling`]。

如果您有兴趣提交资源以供包含在此处，请随时发起 Pull Request，我们将审查它！该资源应该理想地展示出一些新东西，而不是重复现有的资源。

## SwinConfig

[[autodoc]] SwinConfig


## SwinModel

[[autodoc]] SwinModel
    - forward

## SwinForMaskedImageModeling

[[autodoc]] SwinForMaskedImageModeling
    - forward

## SwinForImageClassification

[[autodoc]] transformers.SwinForImageClassification
    - forward

## TFSwinModel

[[autodoc]] TFSwinModel
    - call

## TFSwinForMaskedImageModeling

[[autodoc]] TFSwinForMaskedImageModeling
    - call

## TFSwinForImageClassification

[[autodoc]] transformers.TFSwinForImageClassification
    - call
-->