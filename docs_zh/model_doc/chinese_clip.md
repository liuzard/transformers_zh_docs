<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。

根据 Apache 许可证，版本 2.0（“许可证”）获得许可。您不得使用此文件，除非符合许可证的要求。您可以在此链接处获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”基础分发的，不带任何形式的明示或暗示担保或条件。详见许可证中有关特定语言的规定，以及许可证下的限制。

⚠️ 请注意，此文件是 Markdown 格式，但包含我们文档生成器的特定语法（类似于 MDX），可能在您的 Markdown 查看器中无法正确呈现。

-->

# Chinese-CLIP

## 概述

中文-CLIP 模型是由 An Yang、Junshu Pan、Junyang Lin、Rui Men、Yichang Zhang、Jingren Zhou 和 Chang Zhou 在论文《中文 CLIP：对比视觉语言预训练在中文上的应用》中提出的（[Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335)）。中文-CLIP 是 CLIP（Radford 等人，2021）在大规模中文图像-文本对数据集上的实现。它能够进行跨模态检索，并且还可以作为视觉任务（如零样本图像分类、开放域目标检测等）的视觉骨干模型。原始的中文-CLIP 代码已经在[此链接](https://github.com/OFA-Sys/Chinese-CLIP)上发布。

论文的摘要如下：

*CLIP（Radford 等人，2021）的巨大成功推动了对于视觉语言预训练的对比学习的研究和应用。本文通过构建一个大规模中文图像-文本对数据集（其中大部分数据来自公开可用的数据集），并在新数据集上预训练中文 CLIP 模型。我们开发了 5 个不同大小的中文 CLIP 模型，参数数量从 7700 万到 9.58 亿不等。此外，我们提出了一个两阶段的预训练方法，首先冻结图像编码器训练模型，然后优化所有参数以获得增强模型性能。我们广泛的实验证明中文 CLIP 在零样本学习和微调的 MUGE、Flickr30K-CN 和 COCO-CN 设置中可以取得最先进的性能，并且通过 ELEVATER 基准测试（Li 等人，2022）基于零样本图像分类获得了有竞争力的性能。我们已经发布了代码、预训练模型和演示。*

## 使用方法

下面的代码段演示了如何计算图像和文本特征以及相似性：

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import ChineseCLIPProcessor, ChineseCLIPModel

>>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> # 英文中的 Squirtle（杰尼龟）、Bulbasaur（妙蛙种子）、Charmander（小火龙）和 Pikachu（皮卡丘）
>>> texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

>>> # 计算图像特征
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_features = model.get_image_features(**inputs)
>>> image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # 归一化

>>> # 计算文本特征
>>> inputs = processor(text=texts, padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
>>> text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # 归一化

>>> # 计算图像与文本的相似性得分
>>> inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # 这是图像与文本的相似性得分
>>> probs = logits_per_image.softmax(dim=1)  # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
```

目前，我们在 HF Model Hub 上发布了以下规模的预训练中文-CLIP 模型：

- [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
- [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14)
- [OFA-Sys/chinese-clip-vit-large-patch14-336px](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14-336px)
- [OFA-Sys/chinese-clip-vit-huge-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14)

中文-CLIP 模型由 [OFA-Sys](https://huggingface.co/OFA-Sys) 贡献。

## ChineseCLIPConfig

[[autodoc]] ChineseCLIPConfig
    - from_text_vision_configs

## ChineseCLIPTextConfig

[[autodoc]] ChineseCLIPTextConfig

## ChineseCLIPVisionConfig

[[autodoc]] ChineseCLIPVisionConfig

## ChineseCLIPImageProcessor

[[autodoc]] ChineseCLIPImageProcessor
    - preprocess

## ChineseCLIPFeatureExtractor

[[autodoc]] ChineseCLIPFeatureExtractor

## ChineseCLIPProcessor

[[autodoc]] ChineseCLIPProcessor

## ChineseCLIPModel

[[autodoc]] ChineseCLIPModel
    - forward
    - get_text_features
    - get_image_features

## ChineseCLIPTextModel

[[autodoc]] ChineseCLIPTextModel
    - forward

## ChineseCLIPVisionModel

[[autodoc]] ChineseCLIPVisionModel
    - forward