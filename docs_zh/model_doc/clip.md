<!--
版权所有2021年The HuggingFace Team。保留所有权利。

根据Apache许可证第2.0版（以下简称“许可证”），除非符合许可证的规定，否则你不得使用此文件。你可以通过以下链接获取许可协议的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按"AS IS"原样分发，不附带任何保证或条件。请注意，此文件是Markdown格式，但含有特定于我们文档生成器（类似于MDX）的语法，可能无法在你的Markdown查看器中正确显示。

-->

# CLIP

## 概述

CLIP模型是由Alec Radford、Jong Wook Kim、Chris Hallacy、Aditya Ramesh、Gabriel Goh、Sandhini Agarwal、Girish Sastry、Amanda Askell、Pamela Mishkin、Jack Clark、Gretchen Krueger和Ilya Sutskever在《Learning Transferable Visual Models From Natural Language Supervision》中提出的。CLIP（Contrastive Language-Image Pre-Training）是一个在各种（图像，文本）对上训练的神经网络。类似于GPT-2和3的零样本能力，它可以通过自然语言指导，在给定图像的情况下预测最相关的文本片段，而无需直接针对任务进行优化。

摘要如下：

*当前最先进的计算机视觉系统被训练用于预测一组固定的预定义对象类别。这种受限的监督形式限制了它们的普适性和可用性，因为需要额外的标注数据来指定任何其他视觉概念。直接从原始文本学习图像是一种有前途的替代方法，它利用了更广泛的监督来源。我们证明，预测哪个标题与哪个图像相对应的简单预训练任务是一种从头开始高效可扩展地学习千巫图像表示的方法，这些图像表示是从互联网上收集的4亿（图像，文本）对的数据集中预训练得到的。在预训练之后，使用自然语言来参考学习到的视觉概念（或描述新的概念），使模型能够零样本地转移到下游任务。我们通过在30多个不同的计算机视觉数据集上进行基准测试，比如OCR、视频中的动作识别、地理定位和许多类型的细粒度对象分类等任务，研究了这种方法的性能。该模型通常可以非常有效地转移到大多数任务中，并且常常在无需任何特定数据集训练的情况下与完全有监督的基线模型竞争。例如，我们在ImageNet零样本上与原始ResNet-50的准确率相匹配，而无需使用原始ResNet-50训练的128万个示例之一。我们在此https URL上释放了我们的代码和预训练模型权重。*

## 用法

CLIP是一个多模态的视觉和语言模型。它可以用于图像-文本相似度和零样本图像分类。CLIP使用类似ViT的transformer获取视觉特征，并使用一种因果语言模型获取文本特征。然后将文本和视觉特征投影到一个相同维度的潜空间中，然后使用投影图像和文本特征之间的点积作为相似分数。

为了将图像输入Transformer编码器，需要将每个图像分割成一个固定大小的非重叠的补丁序列，然后进行线性嵌入。添加[CLS]令牌作为整个图像的表示。作者还添加了绝对位置嵌入，并将生成的向量序列馈送到标准Transformer编码器中。[`CLIPImageProcessor`]可用于调整（或缩放）和规范化模型的图像。

[`CLIPTokenizer`]用于编码文本。[`CLIPProcessor`]封装了[`CLIPImageProcessor`]和[`CLIPTokenizer`]，用于同时编码文本和准备图像。以下示例展示了如何使用[`CLIPProcessor`]和[`CLIPModel`]获取图像文本相似性得分。

```python
>>> from PIL import Image
>>> import requests

>>> from transformers import CLIPProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # 这是图像文本相似度得分
>>> probs = logits_per_image.softmax(dim=1)  # 我们可以使用softmax获取标签概率
```

此模型由[valhalla](https://huggingface.co/valhalla)贡献。原始代码可在[此处](https://github.com/openai/CLIP)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）的资源列表，可帮助你开始使用CLIP。

- 一篇关于[如何在10,000个图像-文本对上微调CLIP](https://huggingface.co/blog/fine-tune-clip-rsicd)的博客文章。
- CLIP支持的[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text)。

如果你有兴趣提交资源以包含在此处，请随时提出Pull Request，我们将进行审核。
资源理想上应该展示一些新东西，而不是重复现有的资源。

## CLIPConfig

[[autodoc]] CLIPConfig
    - from_text_vision_configs

## CLIPTextConfig

[[autodoc]] CLIPTextConfig

## CLIPVisionConfig

[[autodoc]] CLIPVisionConfig

## CLIPTokenizer

[[autodoc]] CLIPTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CLIPTokenizerFast

[[autodoc]] CLIPTokenizerFast

## CLIPImageProcessor

[[autodoc]] CLIPImageProcessor
    - preprocess

## CLIPFeatureExtractor

[[autodoc]] CLIPFeatureExtractor

## CLIPProcessor

[[autodoc]] CLIPProcessor

## CLIPModel

[[autodoc]] CLIPModel
    - forward
    - get_text_features
    - get_image_features

## CLIPTextModel

[[autodoc]] CLIPTextModel
    - forward

## CLIPTextModelWithProjection

[[autodoc]] CLIPTextModelWithProjection
    - forward

## CLIPVisionModelWithProjection

[[autodoc]] CLIPVisionModelWithProjection
    - forward


## CLIPVisionModel

[[autodoc]] CLIPVisionModel
    - forward

## TFCLIPModel

[[autodoc]] TFCLIPModel
    - call
    - get_text_features
    - get_image_features

## TFCLIPTextModel

[[autodoc]] TFCLIPTextModel
    - call

## TFCLIPVisionModel

[[autodoc]] TFCLIPVisionModel
    - call

## FlaxCLIPModel

[[autodoc]] FlaxCLIPModel
    - __call__
    - get_text_features
    - get_image_features

## FlaxCLIPTextModel

[[autodoc]] FlaxCLIPTextModel
    - __call__

## FlaxCLIPTextModelWithProjection

[[autodoc]] FlaxCLIPTextModelWithProjection
    - __call__

## FlaxCLIPVisionModel

[[autodoc]] FlaxCLIPVisionModel
    - __call__