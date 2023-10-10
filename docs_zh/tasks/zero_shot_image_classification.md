<!--版权2023年HuggingFace团队保留所有权利。

根据Apache许可证2.0版（“许可证”）许可；除非依法要求或书面同意，否则不得使用本文件。
你可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

但请注意，此文件是使用Markdown格式编写的，但包含了我们的doc-builder的特定语法（类似于MDX），可能无法在Markdown查看器中正确呈现。

-->

# 零样本图像分类

[[open-in-colab]]

零样本图像分类是一项任务，其中涉及使用一个没有明确训练过包含来自这些特定类别的标记示例的数据的模型，将图像分类到不同的类别中。

传统上，图像分类需要在一组特定标记图像上训练模型，该模型学习将某些图像特征“映射”到标签上。当需要使用此类模型进行引入新标签的分类任务时，需要对其进行微调以“重新校准”模型。

相比之下，零样本或开放词汇图像分类模型通常是多模态模型，其已在大量图像及其关联的描述的数据集上进行了训练。这些模型学习了对齐的视觉-语言表示，可以用于许多下游任务，包括零样本图像分类。

这种更灵活的图像分类方法允许模型在没有额外训练数据的情况下泛化到新的和未知的类别，并使用户可以使用目标对象的自由文本描述查询图像。

在本指南中，你将学习如何：

* 创建一个零样本图像分类流程
* 手动运行零样本图像分类推断

在开始之前，请确保已安装所有必要的库：

```bash
pip install -q transformers
```

## 零样本图像分类流程

尝试使用支持零样本图像分类的模型进行推断的最简单方法是使用相应的[`pipeline`]。
从Hugging Face Hub上的[检查点](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads)实例化一个pipeline：

```python
>>> from transformers import pipeline

>>> checkpoint = "openai/clip-vit-large-patch14"
>>> detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
```

接下来，选择要分类的图像。

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/owl.jpg" alt="一只猫头鹰的照片"/>
</div>

将图像和候选对象标签传递给pipeline。在本示例中，我们直接传递图像；其他合适的选项包括图像的本地路径或图像的URL。
候选标签可以是简单的单词，就像这个示例中一样，也可以更详细描述。

```py
>>> predictions = detector(image, candidate_labels=["狐狸", "熊", "海鸥", "猫头鹰"])
>>> predictions
[{'score': 0.9996670484542847, 'label': '猫头鹰'},
 {'score': 0.000199399160919711, 'label': '海鸥'},
 {'score': 7.392891711788252e-05, 'label': '狐狸'},
 {'score': 5.96074532950297e-05, 'label': '熊'}]
```

## 手动零样本图像分类

现在你已经了解了如何使用零样本图像分类流程，让我们来看看如何手动运行零样本图像分类。

首先，从Hugging Face Hub的[检查点](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads)加载模型和相关处理器。
这次我们仍然使用之前相同的检查点:

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

>>> model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

我们来用一张不同的图片进行尝试。

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" alt="一辆汽车的照片"/>
</div>

使用处理器为模型准备输入。处理器包括一个图像处理器，它通过调整大小和归一化来准备模型的图像，以及一个token处理器，它负责处理文本输入。

```py
>>> candidate_labels = ["树", "汽车", "自行车", "猫"]
>>> inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
```

将输入传递给模型，并对结果进行后处理：

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits_per_image[0]
>>> probs = logits.softmax(dim=-1).numpy()
>>> scores = probs.tolist()

>>> result = [
...     {"score": score, "label": candidate_label}
...     for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
... ]

>>> result
[{'score': 0.998572, 'label': '汽车'},
 {'score': 0.0010570387, 'label': '自行车'},
 {'score': 0.0003393686, 'label': '树'},
 {'score': 3.1572064e-05, 'label': '猫'}]
```