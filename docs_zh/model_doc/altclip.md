<!--版权2022 The HuggingFace团队。版权所有。

根据Apache许可证，版本2.0（“许可证”），您不得使用此文件，除非符合许可证的规定。
您可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律、或经书面同意，根据许可证分发的软件是基于“安基础”的，没有任何形式的担保或条件，无论是明示的还是暗示的。请参阅许可证，获取有关特定语言的权限和限制。

⚠️ 请注意，此文件以Markdown格式编写，但包含针对我们的文档生成器（类似于MDX）的特定语法，可能无法在Markdown查看器中正确呈现。-->

＃AltCLIP

## 概述

AltCLIP模型提出了[AltCLIP：更改CLIP中的语言编码器以获得扩展的语言功能](https://arxiv.org/abs/2211.06679v2) by Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Fulong Ye, Qinghong Yang, Ledell Wu。AltCLIP（更改CLIP中的语言编码器）是神经网络，通过对各种图像文本和文本文本对进行训练。通过将CLIP的文本编码器切换为预训练的多语言文本编码器XLM-R，我们可以获得与CLIP在几乎所有任务中非常接近的性能，并扩展了原始CLIP的多语言理解能力。

以下是该论文的摘要：

*在这项工作中，我们提出了一种概念上简单而有效的方法来训练强大的双语多模态表示模型。
从OpenAI发布的预先训练的多模态表示模型CLIP开始，我们切换了其文本编码器，并对两种语言和图像表示进行了对齐，方法是通过包含教师学习和对比学习的两阶段训练架构来训练。我们通过对广泛范围的任务进行评估来验证我们的方法。我们在包括ImageNet-CN、Flicker30k-CN和COCO-CN在内的一堆任务中取得了新的最佳表现。此外，我们在几乎所有任务中都获得了与CLIP非常接近的性能，这表明可以简单地更改CLIP中的文本编码器，以获得扩展的多语言理解等功能。*

## 用法

AltCLIP的用法与CLIP非常相似。 CLIP之间的区别在于文本编码器。请注意，我们使用双向注意力而不是正常的注意力，并且我们使用XLM-R中的[CLS]标记来表示文本嵌入。

AltCLIP是一个多模态视觉和语言模型。它可用于图像-文本相似性和零样本图像分类。 AltCLIP使用类似于ViT的转换器来获取视觉特征，并使用双向语言模型来获取文本特征。然后将文本和视觉特征投影到具有相同维度的潜在空间。然后，将投影图像和文本特征之间的点积用作相似分数。

为了将图像提供给Transformer编码器，将每个图像分割为一系列固定大小的非重叠图像块，然后进行线性嵌入。添加了[CLS]标记，用于表示整个图像的表示。作者还添加了绝对位置嵌入，并将结果向量序列馈送到标准Transformer编码器中。可以使用[`CLIPImageProcessor`]来调整（或重新调整）和规范化模型的图像。

[`AltCLIPProcessor`]将[`CLIPImageProcessor`]和[`XLMRobertaTokenizer`]封装到一个单一实例中，用于同时对文本进行编码和准备图像。以下示例显示了如何使用[`AltCLIPProcessor`]和[`AltCLIPModel`]获取图像-文本相似性得分。

```python
>>> from PIL import Image
>>> import requests

>>> from transformers import AltCLIPModel, AltCLIPProcessor

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # 这是图像-文本相似性得分
>>> probs = logits_per_image.softmax(dim=1)  # 我们可以使用softmax获取标签概率
```

提示：

此模型是基于`CLIPModel`构建的，因此可以像原始CLIP一样使用。

此模型由[jongjyh](https://huggingface.co/jongjyh)贡献。

## AltCLIPConfig

[[autodoc]] AltCLIPConfig
- from_text_vision_configs

## AltCLIPTextConfig

[[autodoc]] AltCLIPTextConfig

## AltCLIPVisionConfig

[[autodoc]] AltCLIPVisionConfig

## AltCLIPProcessor

[[autodoc]] AltCLIPProcessor

## AltCLIPModel

[[autodoc]] AltCLIPModel
- forward
- get_text_features
- get_image_features

## AltCLIPTextModel

[[autodoc]] AltCLIPTextModel
- forward

## AltCLIPVisionModel

[[autodoc]] AltCLIPVisionModel
- forward