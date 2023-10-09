<!--版权2023 The Intel Labs团队作者、微软研究团队作者和HuggingFace Inc.团队。版权所有。

根据Apache License，Version 2.0（“许可证”）的规定，除非符合许可证的规定，否则（除非符合许可证的规定）你不得使用此文件。
你可以在以下位置获取许可证的拷贝：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面协议，否则依照许可证分发的软件以“按现状”分发，
无论明示或暗示，都不附带任何担保或条件。请参阅许可证中明确的特定语言以及许可证下的限制。

⚠️请注意，此文件为Markdown格式，但包含特定的语法，用于我们的文档构建器（类似于MDX），可能无法在你的Markdown查看器中正确呈现。

-->

# BridgeTower

## 概述

BridgeTower模型是由Xiao Xu、Chenfei Wu、Shachar Rosenman、Vasudev Lal、Wanxiang Che、Nan Duan在[《BridgeTower: Building Bridges Between Encoders in Vision-Language Representative Learning》](https://arxiv.org/abs/2206.08657)中提出的。该模型的目标是在每个交叉模态编码器的每一层之间建立每个单模态编码器和交叉模态编码器之间的桥梁，以实现全面和详细的交互，从而在几乎可忽略的附加性能和计算成本的情况下在各种下游任务上实现卓越的性能。

该论文已被接受到[AAAI'23](https://aaai.org/Conferences/AAAI-23/)会议。 

文章摘要如下：

*近年来，基于视觉-语言(TWO-TOWER)架构的视觉-语言(VL)模型在视觉-语言表示学习中占主导地位。
当前的VL模型要么使用轻量级的单模态编码器学习同时提取、对齐和融合两种模态的深度交叉模态编码器，要么将深度预训练的单模态编码器的最后一层单模态表示馈送到顶层交叉模态编码器中。
这两种方法都可能限制视觉-语言表示学习，并限制了模型的性能。在本文中，我们提出了BRIDGETOWER，它引入了多个桥接层，建立了单模态编码器的顶层和交叉模态编码器的每一层之间的连接。
这使得预训练的单模态编码器的不同语义层次的视觉和文本表示在交叉模态编码器中实现了有效的自下而上的跨模态对齐和融合。使用仅400万张图像进行预训练，BRIDGETOWER在各种下游视觉-语言任务上实现了最先进的性能。
特别是，在VQAv2测试集上，BRIDGETOWER的准确率达到了78.73％，超过了以同样的预训练数据和几乎可忽略的附加参数和计算成本得到的之前最先进的模型METER 1.09％。
值得注意的是，在进一步扩展模型时，BRIDGETOWER的准确率达到了81.15％，超过了在数量级更大的数据集上进行预训练的模型。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/bridgetower_architecture%20.jpg"
alt="drawing" width="600"/>

<small> BridgeTower架构。摘自<a href="https://arxiv.org/abs/2206.08657">原论文。</a> </small>

## 使用方法

BridgeTower由视觉编码器、文本编码器和带有多个轻量级桥接层的交叉模态编码器组成。
这种方法的目标是在每个单模态编码器和交叉模态编码器之间建立一座桥梁，以实现交叉模态编码器每一层的全面和详细的交互。
原则上，可以在所提出的架构中应用任何视觉、文本或交叉模态编码器。

[`BridgeTowerProcessor`]将[`RobertaTokenizer`]和[`BridgeTowerImageProcessor`]封装到一个实例中，以对文本进行编码和准备图像。

以下示例展示了如何使用[`BridgeTowerProcessor`]和[`BridgeTowerForContrastiveLearning`]运行对比学习。
```python
>>> from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
>>> import requests
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
>>> model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

>>> # forward pass
>>> scores = dict()
>>> for text in texts:
...     # prepare inputs
...     encoding = processor(image, text, return_tensors="pt")
...     outputs = model(**encoding)
...     scores[text] = outputs
```

以下示例展示了如何使用[`BridgeTowerProcessor`]和[`BridgeTowerForImageAndTextRetrieval`]运行图像-文本检索。
```python
>>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
>>> import requests
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
>>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

>>> # forward pass
>>> scores = dict()
>>> for text in texts:
...     # prepare inputs
...     encoding = processor(image, text, return_tensors="pt")
...     outputs = model(**encoding)
...     scores[text] = outputs.logits[0, 1].item()
```

以下示例展示了如何使用[`BridgeTowerProcessor`]和[`BridgeTowerForMaskedLM`]运行遮蔽语言建模。
```python
>>> from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000360943.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
>>> text = "a <mask> looking out of the window"

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
>>> model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

>>> # prepare inputs
>>> encoding = processor(image, text, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**encoding)

>>> results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())

>>> print(results)
.a cat looking out of the window.
```

此模型由[Anahita Bhiwandiwalla](https://huggingface.co/anahita-b)、[Tiep Le](https://huggingface.co/Tile)和[Shaoyen Tseng](https://huggingface.co/shaoyent)贡献。原始代码可在[此处](https://github.com/microsoft/BridgeTower)找到。


提示：

- BridgeTower的这个实现使用[`RobertaTokenizer`]生成文本嵌入，并使用OpenAI的CLIP/ViT模型计算图像嵌入。
- 已发布了预训练的桥塔基础（bridgeTower-base）和桥塔遮蔽语言建模和图像文本匹配（bridgetower masked language modeling and image text matching）的检查点。
- 请参考[表5](https://arxiv.org/pdf/2206.08657.pdf)了解桥塔在图像检索和其他下游任务上的表现。
- 该模型的PyTorch版本仅适用于torch 1.10及更高版本。


## BridgeTowerConfig

[[autodoc]] BridgeTowerConfig

## BridgeTowerTextConfig

[[autodoc]] BridgeTowerTextConfig

## BridgeTowerVisionConfig

[[autodoc]] BridgeTowerVisionConfig

## BridgeTowerImageProcessor

[[autodoc]] BridgeTowerImageProcessor
    - preprocess

## BridgeTowerProcessor

[[autodoc]] BridgeTowerProcessor
    - __call__

## BridgeTowerModel

[[autodoc]] BridgeTowerModel
    - forward

## BridgeTowerForContrastiveLearning

[[autodoc]] BridgeTowerForContrastiveLearning
    - forward

## BridgeTowerForMaskedLM

[[autodoc]] BridgeTowerForMaskedLM
    - forward

## BridgeTowerForImageAndTextRetrieval

[[autodoc]] BridgeTowerForImageAndTextRetrieval
    - forward