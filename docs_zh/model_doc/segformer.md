<!--版权所有2021年The HuggingFace团队保留。

根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证的要求，否则你不得使用此文件。
你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”的分发，没有任何形式的明示或暗示担保或条件。
请注意，此文件是Markdown格式的，但包含我们的文档构建器（类似于MDX）的特定语法，可能在Markdown查看器中无法正确渲染。

-->

# SegFormer

## 概览

SegFormer模型是由Enze Xie、Wenhai Wang、Zhiding Yu、Anima Anandkumar、Jose M. Alvarez和Ping Luo在[A Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)中提出的。该模型由一个分层Transformer编码器和一个轻量级的全MLP解码头组成，可以在ADE20K和Cityscapes等图像分割基准上取得出色的结果。

论文中的摘要如下：

*我们提出了SegFormer，这是一个简单、高效且强大的语义分割框架，它将Transformer与轻量级多层感知器（MLP）解码器结合起来。SegFormer具有两个吸引人的特点：1）SegFormer采用了一种新颖的分层结构Transformer编码器，输出多尺度特征。它不需要位置编码，避免了由于测试分辨率与训练分辨率不同时插值位置代码而导致的性能降低。2）SegFormer避免了复杂的解码器。所提出的MLP解码器聚合来自不同层的信息，从而结合了局部注意力和全局注意力生成强大的表示。我们表明，这种简单轻量级的设计是在Transformer上进行高效分割的关键。我们扩展我们的方法，从SegFormer-B0到SegFormer-B5获得了一系列模型，其性能和效率显著优于之前的对应方法。例如，SegFormer-B4在ADE20K上使用64M参数实现了50.3%的mIoU，比之前的最佳方法更小5倍且性能提升2.2%。我们的最佳模型SegFormer-B5在Cityscapes验证集上达到了84.0%的mIoU，并且在Cityscapes-C上显示出了出色的零样本鲁棒性。*

下图展示了SegFormer的架构，摘自[原始论文](https://arxiv.org/abs/2105.15203)。

<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/segformer_architecture.png"/>

这个模型由[nielsr](https://huggingface.co/nielsr)贡献。模型的TensorFlow版本由[sayakpaul](https://huggingface.co/sayakpaul)贡献。原始代码可以在[这里](https://github.com/NVlabs/SegFormer)找到。

提示：

- SegFormer由一个分层Transformer编码器和一个轻量级的全MLP解码器头组成。
  [`SegformerModel`]是分层Transformer编码器（在论文中也被称为Mix Transformer或MiT），
  [`SegformerForSemanticSegmentation`]在其上添加了全MLP解码器头以执行图像的语义分割。此外，还有
  [`SegformerForImageClassification`]可以用于对图像进行分类。SegFormer的作者首先在ImageNet-1k上预训练Transformer编码器以对图像进行分类，然后丢弃分类头，用全MLP解码头替换它。然后，他们在ADE20K、Cityscapes和COCO-stuff上整体微调模型，这些都是语义分割的重要基准。所有检查点都可以在[hub](https://huggingface.co/models?other=segformer)上找到。
- 使用SegFormer最快的方法是检查[示例笔记本](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SegFormer)（展示推理和对自定义数据进行微调）。还可以查看介绍SegFormer并说明如何在自定义数据上进行微调的[博客文章](https://huggingface.co/blog/fine-tune-segformer)。
- TensorFlow用户应参考[此存储库](https://github.com/deep-diver/segformer-tf-transformers)，展示开箱即用的推理和微调。
- 你还可以在Hugging Face Spaces上的[此交互式演示](https://huggingface.co/spaces/chansung/segformer-tf-transformers)上尝试SegFormer模型对自定义图像的应用。
- SegFormer适用于任何输入大小，因为它将输入填充到可以被`config.patch_sizes`整除的大小。
- 你可以使用[`SegformerImageProcessor`]为模型准备图像和相应的分割地图。请注意，这个图像处理器非常基础，不包括原始论文中使用的所有数据增强。原始的预处理流程（例如ADE20k数据集）可以在[这里](https://github.com/NVlabs/SegFormer/blob/master/local_configs/_base_/datasets/ade20k_repeat.py)找到。最重要的预处理步骤是将图像和分割地图随机裁剪并填充到相同的大小，例如512x512或640x640，然后进行归一化。
- 还要注意的一点是，可以将[`SegformerImageProcessor`]初始化为`reduce_labels`设置为`True`或`False`。在某些数据集（如ADE20K）中，注释的分割地图中使用0索引表示背景。但是，ADE20k的150个标签中不包括"background"类。因此，`reduce_labels`用于减少所有标签1，并确保不计算背景类的损失（即，它将标注地图中的0替换为255，这是[`SegformerForSemanticSegmentation`]使用的损失函数的*ignore_index*）。然而，其他数据集使用0索引作为背景类，并将该类包含在所有标签中。在这种情况下，`reduce_labels`应设置为`False`，因为背景类也应计算损失。
- 和大多数模型一样，SegFormer有不同大小的变体，详情可以在下表中找到（摘自[原始论文](https://arxiv.org/abs/2105.15203)的表7）。

| **Model variant** | **Depths**    | **Hidden sizes**    | **Decoder hidden size** | **Params (M)** | **ImageNet-1k Top 1** |
| :---------------: | ------------- | ------------------- | :---------------------: | :------------: | :-------------------: |
| MiT-b0            | [2, 2, 2, 2]  | [32, 64, 160, 256]  | 256                     | 3.7            | 70.5                  |
| MiT-b1            | [2, 2, 2, 2]  | [64, 128, 320, 512] | 256                     | 14.0           | 78.7                  |
| MiT-b2            | [3, 4, 6, 3]  | [64, 128, 320, 512] | 768                     | 25.4           | 81.6                  |
| MiT-b3            | [3, 4, 18, 3] | [64, 128, 320, 512] | 768                     | 45.2           | 83.1                  |
| MiT-b4            | [3, 8, 27, 3] | [64, 128, 320, 512] | 768                     | 62.6           | 83.6                  |
| MiT-b5            | [3, 6, 40, 3] | [64, 128, 320, 512] | 768                     | 82.0           | 83.8                  |

上表中的MiT是SegFormer中引入的Mix Transformer编码器骨干的缩写。SegFormer在像ADE20k这样的分割数据集上的结果，请参阅[论文](https://arxiv.org/abs/2105.15203)。

## 资源

官方Hugging Face和社区（标有🌎）提供的帮助你入门SegFormer的资源列表。

<PipelineTag pipeline="image-classification"/>

- [`SegformerForImageClassification`]由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和这个[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持。
- [图像分类任务指南](../tasks/image_classification)

语义分割：

- [`SegformerForSemanticSegmentation`]由这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation)支持。
- 有关将SegFormer微调到自定义数据集上的博客文章，请参考[这里](https://huggingface.co/blog/fine-tune-segformer)。
- 更多关于SegFormer的演示笔记本（包括推理和对自定义数据集的微调）可以在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SegFormer)找到。
- [`TFSegformerForSemanticSegmentation`]由这个[笔记本](https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation-tf.ipynb)支持。
- [语义分割任务指南](../tasks/semantic_segmentation)

如果你有兴趣提交资源以包含在此处，请随时提交拉取请求，我们将进行审查！该资源应尽量展示一些新的东西，而不是重复现有资源。

## SegformerConfig

[[autodoc]] SegformerConfig

## SegformerFeatureExtractor

[[autodoc]] SegformerFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## SegformerImageProcessor

[[autodoc]] SegformerImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## SegformerModel

[[autodoc]] SegformerModel
    - forward

## SegformerDecodeHead

[[autodoc]] SegformerDecodeHead
    - forward

## SegformerForImageClassification

[[autodoc]] SegformerForImageClassification
    - forward

## SegformerForSemanticSegmentation

[[autodoc]] SegformerForSemanticSegmentation
    - forward

## TFSegformerDecodeHead

[[autodoc]] TFSegformerDecodeHead
    - call

## TFSegformerModel

[[autodoc]] TFSegformerModel
    - call

## TFSegformerForImageClassification

[[autodoc]] TFSegformerForImageClassification
    - call

## TFSegformerForSemanticSegmentation

[[autodoc]] TFSegformerForSemanticSegmentation
    - call
