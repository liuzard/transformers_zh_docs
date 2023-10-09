<!--版权所有2022年HuggingFace团队。保留所有权利。 

根据Apache许可证2.0版（“许可证”）进行授权；你不得在未遵守许可证的情况下使用本文件。
你可以从下面的链接获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样” BASIS 分发的，没有任何明示或默示的保证、条件或其他条款。请详阅许可证中的特定语言这样的许可证可能在你的Markdown视图器中不正确渲染。
-->

# MobileViT

## 概述

MobileViT模型是由Sachin Mehta和Mohammad Rastegari在《MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer》中提出的。MobileViT通过使用Transformer将卷积中的局部处理替换为全局处理，引入了一个新的层。

论文中的摘要如下：

移动视觉任务的轻量级卷积神经网络（CNN）已成为事实上的标准。它们的空间归纳偏差使它们能够在不同的视觉任务中学习具有更少参数的表示。然而，这些网络是局部的。为了学习全局表示，采用了基于自注意力的视觉变换器（ViTs）。与CNNs不同，ViTs很重。在本文中，我们提出了以下问题：是否可能将CNNs和ViTs的优势相结合，构建一个轻量级和低延迟的移动视觉任务网络？为此，我们引入MobileViT，一种轻量级且通用的用于移动设备的视觉变换器。MobileViT提出了使用变换器作为卷积进行信息的全局处理的不同观点。我们的结果显示，MobileViT在不同的任务和数据集上明显优于基于CNNs和ViTs的网络。在ImageNet-1k数据集上，MobileViT以大约600万个参数实现了78.4%的top-1准确率，这比MobileNetv3（基于CNNs）和DeIT（基于ViTs）准确率提高了3.2%和6.2%。在MS-COCO目标检测任务中，MobileViT在准确率方面比MobileNetv3高出5.7%，参数数量相似。

提示：

- MobileViT更像是CNN模型，而不是Transformer模型。它不适用于序列数据，而是用于图像批次。与ViT不同，没有嵌入。骨干模型输出一个特征图。你可以参考[这个教程](https://keras.io/examples/vision/mobilevit)进行简单入门。
- 可以使用 [`MobileViTImageProcessor`](https://huggingface.co/transformers/main_classes/mobile_vit_image_processor.html) 准备模型的图像数据。请注意，如果你自己进行预处理，预训练的检查点要求图像采用BGR像素顺序（而不是RGB）。
- 可用的图像分类检查点是在[ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)上预训练的（也称为ILSVRC 2012，包含130万张图像和1000个类别）。
- 分割模型使用[DeepLabV3](https://arxiv.org/abs/1706.05587) head。可用的语义分割检查点是在[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)上预训练的。
- 正如名称所暗示的，MobileViT旨在在手机上表现出色并具有高效性。MobileViT模型的TensorFlow版本与[TensorFlow Lite](https://www.tensorflow.org/lite)完全兼容。

  你可以使用以下代码将MobileViT检查点（无论是图像分类还是语义分割）转换为生成一个TensorFlow Lite模型：

```py
from transformers import TFMobileViTForImageClassification
import tensorflow as tf

model_ckpt = "apple/mobilevit-xx-small"
model = TFMobileViTForImageClassification.from_pretrained(model_ckpt)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()
tflite_filename = model_ckpt.split("/")[-1] + ".tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_model)
```

  生成的模型大小约为**1MB**，非常适合资源和网络带宽受限的移动应用程序。

本模型由[matthijs](https://huggingface.co/Matthijs)贡献。模型的TensorFlow版本由[sayakpaul](https://huggingface.co/sayakpaul)贡献。可以在[这里](https://github.com/apple/ml-cvnets)找到原始代码和权重。

## 资源

以下是官方Hugging Face和社区（由🌎标示）提供的一些资源，以帮助你开始使用MobileViT。

<PipelineTag pipeline="image-classification"/>

- [`MobileViTForImageClassification`](https://huggingface.co/transformers/main_classes/mobile_vit/mobile_vit_for_image_classification.html) 在此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)中得到支持。
- 参见：[图像分类任务指南](../tasks/image_classification)

**语义分割**
- [语义分割任务指南](../tasks/semantic_segmentation)

如果你有兴趣提供一个资源以供包含在这里，请随时提出拉取请求，我们将进行审查！该资源理想上应该展示出一些新的东西，而不是重复现有的资源。

## MobileViTConfig

[[autodoc]] MobileViTConfig

## MobileViTN特征提取器

[[autodoc]] MobileViTFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## MobileViTImageProcessor

[[autodoc]] MobileViTImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## MobileViTModel

[[autodoc]] MobileViTModel
    - forward

## MobileViTForImageClassification

[[autodoc]] MobileViTForImageClassification
    - forward

## MobileViTForSemanticSegmentation

[[autodoc]] MobileViTForSemanticSegmentation
    - forward

## TFMobileViTModel

[[autodoc]] TFMobileViTModel
    - call

## TFMobileViTForImageClassification

[[autodoc]] TFMobileViTForImageClassification
    - call

## TFMobileViTForSemanticSegmentation

[[autodoc]] TFMobileViTForSemanticSegmentation
    - call