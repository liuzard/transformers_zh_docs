<!--版权 2023年HuggingFace团队。 保留所有权利。

根据Apache许可证第2版（“许可证”）获得许可； 除非符合许可证，否则不得使用此文件。
您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非依法要求或书面同意，依据许可证分发的软件是基于“按原样”提供的，不附带任何明示或暗示的担保或条件。
有关特定语言下的许可证的限制和条件，请参阅许可证。

⚠️请注意，此文件为Markdown文件，但包含我们的文档构建器（类似于MDX）的特定语法，可能无法在Markdown查看器中正确渲染。

-->

# ALIGN

## 概述

ALIGN模型在[Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918)一文中由Chao Jia、Yinfei Yang、Ye Xia、Yi-Ting Chen、Zarana Parekh、Hieu Pham、Quoc V. Le、Yunhsuan Sung、Zhen Li和Tom Duerig提出。ALIGN是一个多模态的视觉和语言模型。它可用于图像-文本相似性和零样本图像分类。ALIGN采用双编码器架构，其中以[EfficientNet](efficientnet)作为其视觉编码器，以[BERT](bert)作为其文本编码器，并学习通过对比学习来对齐视觉和文本表示。与以前的工作不同，ALIGN利用了一个庞大的噪声数据集，并表明语料库的规模可以用一个简单的配方来实现SOTA表示。

论文中的摘要如下：

*预训练表示对于许多自然语言处理（NLP）和感知任务正在变得至关重要。虽然NLP中的表示学习已经过渡到在原始文本上进行训练而不需要人类注释，但视觉和视觉语言表示仍然严重依赖于昂贵或需要专业知识的精心策划的训练数据集。对于视觉应用程序，表示主要是使用具有显式类标签的数据集（例如ImageNet或OpenImages）进行学习的。对于视觉语言，像Conceptual Captions、MSCOCO或CLIP这样的流行数据集都涉及到一个非平凡的数据收集（和清理）过程。这个昂贵的策划过程限制了数据集的规模，从而阻碍了训练模型的规模化。在本文中，我们利用一亿多个图像替代文本对的噪声数据集，在Conceptual Captions数据集中没有昂贵的过滤或后处理步骤的情况下获得。一个简单的双编码器架构通过对比损失来学习图像和文本对的视觉和语言表示的对齐。我们表明，我们的语料库的规模可以弥补其噪声，并通过这样一个简单的学习方案实现SOTA表示。我们的视觉表示在转移到诸如ImageNet和VTAB之类的分类任务时取得了很好的性能。对齐的图像和语言表示使得零样本图像分类成为可能，并且在Flickr30K和MSCOCO图像-文本检索基准上取得了新的SOTA结果，即使与更复杂的交叉注意力模型进行比较。这些表示还使得复杂文本和文本+图像查询的跨模态搜索成为可能。*

## 用法

ALIGN使用EfficientNet获取视觉特征，使用BERT获取文本特征。然后将文本和视觉特征投影到相同维度的潜空间中。然后使用投影图像和文本特征之间的点积作为相似性评分。

[`AlignProcessor`]将[`EfficientNetImageProcessor`]和[`BertTokenizer`]封装为单个实例，用于对文本进行编码和对图像进行预处理。以下示例展示了如何使用[`AlignProcessor`]和[`AlignModel`]获取图像-文本相似性分数。

```python
import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["an image of a cat", "an image of a dog"]

inputs = processor(text=candidate_labels, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 这是图像-文本相似性分数
logits_per_image = outputs.logits_per_image

# 我们可以使用softmax来获得标签概率
probs = logits_per_image.softmax(dim=1)
print(probs)
```

这个模型由[Alara Dirik](https://huggingface.co/adirik)贡献。
原始代码未发布，此实现基于Kakao Brain基于原始论文的实现。

## 资源

官方的和社区（🌎标志）资源列表，这些资源可以帮助您开始使用ALIGN。

- 一篇关于[ALIGN和COYO-700M数据集](https://huggingface.co/blog/vit-align)的博客文章。
- 零样本图像分类的[演示](https://huggingface.co/spaces/adirik/ALIGN-zero-shot-image-classification)。
- `kakaobrain/align-base`模型的[模型卡片](https://huggingface.co/kakaobrain/align-base)。

如果您有兴趣提交要包含在此处的资源，请随时打开拉取请求，我们将进行审核。资源应该尽量展示一些新的东西，而不是重复现有的资源。

## AlignConfig

[[autodoc]] AlignConfig
    - from_text_vision_configs

## AlignTextConfig

[[autodoc]] AlignTextConfig

## AlignVisionConfig

[[autodoc]] AlignVisionConfig

## AlignProcessor

[[autodoc]] AlignProcessor

## AlignModel

[[autodoc]] AlignModel
    - forward
    - get_text_features
    - get_image_features

## AlignTextModel

[[autodoc]] AlignTextModel
    - forward

## AlignVisionModel

[[autodoc]] AlignVisionModel
    - forward