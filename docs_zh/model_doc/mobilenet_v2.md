<!--版权2022由HuggingFace团队保留。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证要求，否则你不得使用此文件。
你可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按"原样"分发的，不附带任何明示或暗示的担保或条件。请参阅许可证了解许可的特定语言和限制。

⚠ 注意，此文件是Markdown文件，但包含我们文档构建器的特定语法（类似于MDX），你的Markdown查看器可能无法正确呈现。-->

# MobileNet V2

## 概述

MobileNet模型由Mark Sandler、Andrew Howard、Menglong Zhu、Andrey Zhmoginov、Liang-Chieh Chen在[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)中提出。

论文的摘要如下：

*在本文中，我们描述了一种新的移动架构MobileNetV2，它在多个任务和基准测试上改进了移动模型的最新性能，并且在不同模型尺寸的光谱上也如此。我们还介绍了在我们称之为SSDLite的新颖框架中应用这些移动模型到目标检测的有效方法。此外，我们展示了如何通过称之为Mobile DeepLabv3的DeepLabv3简化形式构建移动语义分割模型。*

*MobileNetV2架构基于反转残差结构，其中残差块的输入和输出是瘦化的瓶颈层，而传统的残差模型在输入处使用扩展表示，MobileNetV2则使用轻量级的深度可分离卷积来过滤中间扩展层的特征。此外，我们发现在狭窄层中去除非线性是保持表示能力的重要因素。我们证明这改善了性能，并提供了导致这种设计的直觉。最后，我们的方法允许解耦输入/输出域和变换的表达能力，为进一步的分析提供了一个方便的框架。我们通过在Imagenet分类、COCO目标检测和VOC图像分割上测量了我们的性能。我们评估了准确性和乘加（MAdd）乘法操作数以及参数数量之间的权衡。*

提示：

- 检查点的命名为**mobilenet\_v2\_*depth*\_*size***，例如**mobilenet\_v2\_1.0\_224**，其中**1.0**是深度倍增器（有时也称为"alpha"或宽度倍增器），**224**是模型训练时输入图像的分辨率。

- 即使检查点是基于特定大小的图像训练的，该模型也可以处理任意大小的图像。支持的最小图像大小为32x32。

- 你可以使用[`MobileNetV2ImageProcessor`]为模型准备图像。

- 可用的图像分类检查点是在[ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)（也称为ILSVRC 2012，包含130万张图像和1000个类别的数据集）上进行预训练的。然而，该模型预测了1001个类别：ImageNet的1000个类别加上额外的“background”类别（索引0）。

- 分割模型使用了[DeepLabV3+](https://arxiv.org/abs/1802.02611)头部。可用的语义分割检查点是在[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)上进行预训练的。

- 原始的TensorFlow检查点使用了与PyTorch不同的填充规则，因此模型需要在推理时确定填充量，因为这取决于输入图像的大小。要使用原生的PyTorch填充行为，请创建一个`tf_padding = False`的[`MobileNetV2Config`]。

不支持的功能：

- [`MobileNetV2Model`]输出最后隐藏状态的全局池化版本。在原始模型中，可以使用固定的7x7窗口和步幅1的平均池化层代替全局池化。对于大于推荐图像大小的输入，这会产生一个大于1x1的池化输出。Hugging Face实现不支持此功能。

- 原始的TensorFlow检查点包括量化模型。我们不支持这些模型，因为它们含有额外的“FakeQuantization”操作来取消量化权重。

- 通常提取索引10和13处的扩展层输出，以及最后的1x1卷积层输出，供下游使用。使用`output_hidden_states=True`返回所有中间层的输出。目前无法将其限制为特定层。

- DeepLabV3+分割头部不使用骨干网络的最终卷积层，但该层仍会计算。目前无法告知[`MobileNetV2Model`]应运行到哪一层。

此模型由[matthijs](https://huggingface.co/Matthijs)贡献。原始代码和权重可以在[这里获取主模型](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)和[这里获取DeepLabV3+](https://github.com/tensorflow/models/tree/master/research/deeplab)。

## 资源

官方Hugging Face和社区（🌎表示）资源列表，可帮助你开始使用MobileNetV2。

<PipelineTag pipeline="image-classification"/>

- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持[`MobileNetV2ForImageClassification`]。
- 参见：[图像分类任务指南](../tasks/image_classification)

**语义分割**
- [语义分割任务指南](../tasks/semantic_segmentation)

如果你有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将对其进行审查！资源应尽量呈现新的内容，而不是重复现有资源。

## MobileNetV2Config

[[autodoc]] MobileNetV2Config

## MobileNetV2FeatureExtractor

[[autodoc]] MobileNetV2FeatureExtractor
    - 预处理
    - 后处理语义分割

## MobileNetV2ImageProcessor

[[autodoc]] MobileNetV2ImageProcessor
    - 预处理
    - 后处理语义分割

## MobileNetV2Model

[[autodoc]] MobileNetV2Model
    - 前向传播

## MobileNetV2ForImageClassification

[[autodoc]] MobileNetV2ForImageClassification
    - 前向传播

## MobileNetV2ForSemanticSegmentation

[[autodoc]] MobileNetV2ForSemanticSegmentation
    - 前向传播