<!--版权所有2022年The HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”），除非符合下列许可条件
否则你不能使用此文件。你可以在
http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证分发在
基础上"按原样"，没有任何担保或条件，无论是明示的还是隐含的。请参见许可证
特定语言的语句和限制。

⚠️请注意，此文件是Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），这可能无法
在你的Markdown视图器中正确显示。

-->

# MobileNet V1

## 概述

MobileNet模型是由Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam在《MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications》中提出的。

论文中的摘要如下：

*我们提出了一类称为MobileNets的高效模型，适用于移动和嵌入式视觉应用。MobileNets基于一种简化的架构，使用分离的可分离卷积来构建轻量级的深度神经网络。我们引入了两个简单的全局超参数，有效地权衡了延迟和准确性。这些超参数使模型构建者能够根据问题的约束选择适合其应用的合适大小的模型。我们进行了大量的资源和准确性的权衡实验，并在ImageNet分类任务上与其他流行模型相比展示了出色的性能。然后，我们展示了MobileNets在包括目标检测、细粒度分类、面部属性和大规模地理定位在内的广泛应用和用例中的有效性。*

提示：

- 检查点的名称为**mobilenet\_v1\_*depth*\_*size***，例如**mobilenet\_v1\_1.0\_224**，其中**1.0**是深度乘数（有时也称为"alpha"或宽度乘数），而**224**是模型训练所用输入图像的分辨率。

- 即使检查点是基于特定大小的图像训练的，模型也可以处理任意大小的图像。支持的最小图像大小为32x32。

- 可以使用[`MobileNetV1ImageProcessor`]来为模型准备图像。

- 可用的图像分类检查点是在[ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)（也称为ILSVRC 2012，包含130万张图像和1,000个类别）上进行预训练的。然而，模型预测的是1001个类别：ImageNet的1000个类别加上额外的“background”类别（索引为0）。

- 原始的TensorFlow检查点使用与PyTorch不同的填充规则，这要求模型在推理时确定填充量，因为这取决于输入图像的大小。要使用原生的PyTorch填充行为，请创建一个具有`tf_padding = False`的[`MobileNetV1Config`]。

不支持的特性：

- [`MobileNetV1Model`]输出最后一个隐藏状态的全局池化版本。在原始模型中，可以使用一个7x7的平均池化层，步幅为2，而不是全局池化。对于较大的输入，这会产生一个大于1x1像素的池化输出。HuggingFace的实现不支持这一点。

- 目前无法指定`output_stride`。对于较小的输出步幅，原始模型会使用膨胀卷积，以防止空间分辨率进一步减小。HuggingFace模型的输出步幅始终为32。

- 原始的TensorFlow检查点包括量化模型。我们不支持这些模型，因为它们包含额外的“FakeQuantization”操作以反量化权重。

- 在下游任务中，常常提取从运算维度层的输出的5、11、12、13索引。使用`output_hidden_states=True`将返回所有中间层的输出。目前没有办法将其限制在特定层。

此模型是由[matthijs](https://huggingface.co/Matthijs)贡献的。原始代码和权重可在[此处](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)找到。

## 资源

用于帮助你开始使用MobileNetV1的官方Hugging Face和社区资源列表（由🌎表示）。

<PipelineTag pipeline="image-classification"/>

- 通过此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持[`MobileNetV1ForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果你有兴趣提交要包括在此处的资源，请随时提出拉取请求，我们将进行审核！该资源应该理想地展示一些新的内容，而不是重复现有的资源。

## MobileNetV1Config

[[autodoc]] MobileNetV1Config

## MobileNetV1FeatureExtractor

[[autodoc]] MobileNetV1FeatureExtractor
    - preprocess

## MobileNetV1ImageProcessor

[[autodoc]] MobileNetV1ImageProcessor
    - preprocess

## MobileNetV1Model

[[autodoc]] MobileNetV1Model
    - forward

## MobileNetV1ForImageClassification

[[autodoc]] MobileNetV1ForImageClassification
    - forward