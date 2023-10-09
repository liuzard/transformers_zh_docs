<!--版权 2022 HuggingFace团队。版权所有。

根据Apache许可证，版本2.0（“许可证”），你不得使用此文件，除非符合
许可证。你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于
“按原样”方法，无论是明示的还是暗示的，在特定的任何保证或条件下，不得承担
任何类型的责任，无论是明示的还是暗示的，包括但不限于，对于
特定目的适销性和合适性的暗示的保证法。有关的
特定语言管理权限的限制和限制，请参阅许可证。

⚠️选项卡中的文件是Markdown，但包含我们的文档生成器的具体语法（类似于MDX），可能无法
在你的Markdown查看器中正常呈现。--> 

# OWL-ViT

## 概述

OWL-ViT（Open-World Localization的Vision Transformer）是由Matthias Minderer、Alexey Gritsenko、Austin Stone、Maxim Neumann、Dirk Weissenborn、Alexey Dosovitskiy、Aravindh Mahendran、Anurag Arnab、Mostafa Dehghani、Zhuoran Shen、Xiao Wang、Xiaohua Zhai、Thomas Kipf和Neil Houlsby在[Simple Open-Vocabulary Object Detection with Vision Transformers]中提出的（https://arxiv.org/abs/2205.06230）。 OWL-ViT是一个基于多种（图像，文本）对训练的开放词汇目标检测网络。可以使用一个或多个文本查询查询图像，以搜索和检测文本中描述的目标对象。

论文摘要如下：

*使用简单的体系结构和大规模预训练已经在图像分类方面取得了巨大的改进。对于物体检测，预训练和扩展方法还不太成熟，特别是在长尾和开放词汇的环境中，训练数据相对稀缺。在本文中，我们提出了一种将图像文本模型转移到开放词汇目标检测的强大方法。我们使用标准的Vision Transformer体系结构进行微小修改，对比图像文本预训练，并进行端到端的检测微调。我们的这种设置的扩展性属性分析表明，增加图像级预训练和模型大小可以持续改进下游的检测任务。我们提供了适应策略和规范化所需的工具，以在零-shot文本条件和一-shot图像条件物体检测中获得非常强大的性能。代码和模型可以在GitHub上获取。*

## 使用说明

OWL-ViT是一个零-shot文本条件的目标检测模型。OWL-ViT使用[CLIP](clip)作为其多模态主干，其中ViT样式的Transformer用于获取视觉特征，而因果语言模型用于获取文本特征。为了使用CLIP进行检测，OWL-ViT删除了视觉模型的最后一个标记池化层，并将轻量级分类和盒子头部附加到每个Transformer输出标记上。通过使用从文本模型中获得的类名称嵌入替换固定的分类层权重，实现了开放词汇分类。作者首先从头开始训练CLIP，并使用双部分匹配损失在标准检测数据集上端到端地对其进行微调并带有分类和盒子头。可以使用一张或多张图像的一个或多个文本查询来执行零-shot文本条件的目标检测。

[`OwlViTImageProcessor`]可用于调整（或重新缩放）和规范化模型输入的图像，[`CLIPTokenizer`]用于编码文本。 [`OwlViTProcessor`]将[`OwlViTImageProcessor`]和[`CLIPTokenizer`]包装成一个实例，以便同时编码文本和准备图像。以下示例展示了如何使用[`OwlViTProcessor`]和[`OwlViTForObjectDetection`]执行目标检测。

```python
>>> import requests
>>> from PIL import Image
>>> import torch
>>> from transformers import OwlViTProcessor, OwlViTForObjectDetection

>>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
>>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = [["a photo of a cat", "a photo of a dog"]]
>>> inputs = processor(text=texts, images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
>>> target_sizes = torch.Tensor([image.size[::-1]])
>>> # Convert outputs (bounding boxes and class logits) to COCO API
>>> results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
>>> i = 0  # Retrieve predictions for the first image for the corresponding text queries
>>> text = texts[i]
>>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
>>> for box, score, label in zip(boxes, scores, labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]
Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]
```

该模型由[adirik](https://huggingface.co/adirik)提供。原始代码可以在[这里]（https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit）找到。

## OwlViTConfig

[[autodoc]] OwlViTConfig
    - from_text_vision_configs

## OwlViTTextConfig

[[autodoc]] OwlViTTextConfig

## OwlViTVisionConfig

[[autodoc]] OwlViTVisionConfig

## OwlViTImageProcessor

[[autodoc]] OwlViTImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_image_guided_detection

## OwlViTFeatureExtractor

[[autodoc]] OwlViTFeatureExtractor
    - __call__
    - post_process
    - post_process_image_guided_detection

## OwlViTProcessor

[[autodoc]] OwlViTProcessor

## OwlViTModel

[[autodoc]] OwlViTModel
    - forward
    - get_text_features
    - get_image_features

## OwlViTTextModel

[[autodoc]] OwlViTTextModel
    - forward

## OwlViTVisionModel

[[autodoc]] OwlViTVisionModel
    - forward

## OwlViTForObjectDetection

[[autodoc]] OwlViTForObjectDetection
    - forward
    - image_guided_detection