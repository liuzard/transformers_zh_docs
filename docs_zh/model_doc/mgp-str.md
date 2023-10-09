<!--版权所有2023年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），您可能不使用此文件，除非符合许可证的规定。
您可以从以下网址获得许可证副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按"原样"分发的，
不提供任何明示或暗示的担保或条件。请参阅许可证了解特定语言下的权限和限制。

⚠️请注意，此文件采用Markdown格式，但包含我们doc-builder（类似于MDX）的特定语法，
可能在您的Markdown查看器中无法正确渲染。

-->

# MGP-STR

## 概述

MGP-STR模型是由Peng Wang、Cheng Da和Cong Yao在[《Multi-Granularity Prediction for Scene Text Recognition》](https://arxiv.org/abs/2209.03592)中提出的。MGP-STR是一个在视觉场景文本识别（STR）模型中概念上简单但功能强大的模型，它基于[视觉Transformer（ViT）](vit)进行构建。为了整合语言知识，提出了多粒度预测（MGP）策略，以隐式方式将语言模态中的信息注入到模型中。

论文的摘要如下所示：

*多年来，场景文本识别（STR）一直是计算机视觉领域的一个热门研究课题。为了解决这个具有挑战性的问题，已经相继提出了许多创新方法，并将语言知识整合到STR模型中最近成为一个突出的趋势。在这项工作中，我们首先从视觉Transformer（ViT）的最新进展中汲取灵感，构建了一个概念上简单而又功能强大的视觉STR模型，该模型基于ViT构建，优于以前的场景文本识别的最先进模型，包括纯视觉模型和语言增强方法。为了整合语言知识，我们进一步提出了多粒度预测策略，以隐式方式将语言模态中的信息注入到模型中，即在输出空间中引入了在NLP中广泛使用的子词表示（BPE和WordPiece），除了传统的字符级表示，而不使用独立的语言模型（LM）。由此产生的算法（称为MGP-STR）能够将STR的性能推向一个更高的水平。具体而言，在标准基准测试中，它在六个标准的拉丁文场景文本基准中，包括3个常规文本数据集（IC13、SVT、IIIT）和3个非规则文本数据集（IC15、SVTP、CUTE），实现了93.35%的平均识别准确率。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mgp_str_architecture.png"
alt="图片" width="600"/>

<small> MGP-STR模型架构。来自[原始论文](https://arxiv.org/abs/2209.03592)。 </small>

提示：

- MGP-STR模型在两个合成数据集[MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/)（MJ）和SynthText(http://www.robots.ox.ac.uk/~vgg/data/scenetext/) (ST)上进行训练，没有在其他数据集上进行微调。它在六个标准拉丁文场景文本基准测试中，包括3个常规文本数据集（IC13、SVT、IIIT）和3个非规则文本数据集（IC15、SVTP、CUTE），取得了领先水平的结果。
- 此模型由[yuekun](https://huggingface.co/yuekun)贡献。原始代码可以在[这里](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR)找到。

## 推理

[`MgpstrModel`]接受图像作为输入，并生成三种类型的预测，这些预测代表不同的粒度上的文本信息。这三种类型的预测进行融合，得到最终的预测结果。

[`ViTImageProcessor`]类负责对输入图像进行预处理，并且[`MgpstrTokenizer`]对生成的字符标记解码为目标字符串。[`MgpstrProcessor`]将[`ViTImageProcessor`]和[`MgpstrTokenizer`]封装成一个单一实例，既提取输入特征又解码预测的标记ID。

- 逐步光学字符识别（OCR）

``` py
>>> from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
>>> import requests
>>> from PIL import Image

>>> processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
>>> model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')

>>> # 从IIIT-5k数据集加载图像
>>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> pixel_values = processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(pixel_values)

>>> generated_text = processor.batch_decode(outputs.logits)['generated_text']
```

## MgpstrConfig

[[autodoc]] MgpstrConfig

## MgpstrTokenizer

[[autodoc]] MgpstrTokenizer
    - save_vocabulary

## MgpstrProcessor

[[autodoc]] MgpstrProcessor
    - __call__
    - batch_decode

## MgpstrModel

[[autodoc]] MgpstrModel
    - forward

## MgpstrForSceneTextRecognition

[[autodoc]] MgpstrForSceneTextRecognition
    - forward