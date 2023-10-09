<!--版权所有2023年HuggingFace团队。保留所有权利。

根据Apache License第2版（“许可证”）进行许可；除非按适用法律要求或书面同意，否则您不得使用本文件。
您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

请注意，此文件虽然是Markdown格式，但包含我们的doc-builder的特定语法（类似于MDX），
可能无法在您的Markdown查看器中正确呈现。

-->

# 零样本目标检测

[[open-in-colab]]

传统上，用于目标检测的模型需要有标记的图像数据集进行训练，并且仅能检测训练集中的类别。

OWL-ViT模型支持零样本目标检测，它使用了一种不同的方法。OWL-ViT是一种开放词汇表的目标检测器，它可以根据自由文本查询在图像中检测对象，而无需在标记的数据集上对模型进行微调。

OWL-ViT利用多模态表示来进行开放词汇表的检测。它将CLIP与轻量级的对象分类和定位头部结合起来。开放词汇表的检测是通过将自由文本查询与CLIP的文本编码器进行嵌入，并将其作为对象分类和定位头部的输入来实现的。
将图像及其相应的文本描述关联起来，ViT将图像块作为输入。OWL-ViT的作者首先从头开始训练CLIP，然后使用二分匹配损失在标准目标检测数据集上端到端地对OWL-ViT进行微调。

采用这种方法，模型可以根据文本描述检测对象，而无需在标记的数据集上进行先前的训练。

在本指南中，您将学习如何使用OWL-ViT：
- 根据文本提示来检测对象
- 进行批量目标检测
- 进行图像引导的目标检测
                                                        
开始之前，请确保您已安装所有必要的库:

```bash
pip install -q transformers
```

## 零样本目标检测流程

使用OWL-ViT进行推理的最简单方法是在[`pipeline`]中使用它。从[Hugging Face Hub上的检查点](https://huggingface.co/models?other=owlvit) 实例化一个零样本目标检测的pipeline：

```python
>>> from transformers import pipeline

>>> checkpoint = "google/owlvit-base-patch32"
>>> detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
```

接下来，选择一张您想要检测对象的图片。这里我们将使用NASA Great Images数据集中的宇航员Eileen Collins的照片。

```py
>>> import skimage
>>> import numpy as np
>>> from PIL import Image

>>> image = skimage.data.astronaut()
>>> image = Image.fromarray(np.uint8(image)).convert("RGB")

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_1.png" alt="宇航员Eileen Collins"/>
</div>

将图片和要查找的候选对象标签传递给pipeline。
这里我们直接传递了图片；其他合适的选项包括图像的本地路径或图像URL。我们还传递了要查询图像的所有物体的文本描述。

```py
>>> predictions = detector(
...     image,
...     candidate_labels=["human face", "rocket", "nasa badge", "star-spangled banner"],
... )
>>> predictions
[{'score': 0.3571370542049408,
  'label': 'human face',
  'box': {'xmin': 180, 'ymin': 71, 'xmax': 271, 'ymax': 178}},
 {'score': 0.28099656105041504,
  'label': 'nasa badge',
  'box': {'xmin': 129, 'ymin': 348, 'xmax': 206, 'ymax': 427}},
 {'score': 0.2110239565372467,
  'label': 'rocket',
  'box': {'xmin': 350, 'ymin': -1, 'xmax': 468, 'ymax': 288}},
 {'score': 0.13790413737297058,
  'label': 'star-spangled banner',
  'box': {'xmin': 1, 'ymin': 1, 'xmax': 105, 'ymax': 509}},
 {'score': 0.11950037628412247,
  'label': 'nasa badge',
  'box': {'xmin': 277, 'ymin': 338, 'xmax': 327, 'ymax': 380}},
 {'score': 0.10649408400058746,
  'label': 'rocket',
  'box': {'xmin': 358, 'ymin': 64, 'xmax': 424, 'ymax': 280}}]
```

让我们可视化预测结果：

```py
>>> from PIL import ImageDraw

>>> draw = ImageDraw.Draw(image)

>>> for prediction in predictions:
...     box = prediction["box"]
...     label = prediction["label"]
...     score = prediction["score"]

...     xmin, ymin, xmax, ymax = box.values()
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_2.png" alt="NASA图片上可视化的预测结果"/>
</div>

## 手动进行文本启发的零样本目标检测

现在，您已经了解了如何使用零样本目标检测的pipeline，让我们手动复制相同的结果。

首先从[Hugging Face Hub上的检查点](https://huggingface.co/models?other=owlvit)加载模型和相关的processor。
这里我们将使用与之前相同的检查点：

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

我们选取不同的图片来稍微改变一下。

```py
>>> import requests

>>> url = "https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTR8fHBpY25pY3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640"
>>> im = Image.open(requests.get(url, stream=True).raw)
>>> im
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_3.png" alt="海滩照片"/>
</div>

使用processor来准备模型的输入。processor使用图像处理器对图像进行调整和归一化，以便模型处理，并且使用相应的[`CLIPTokenizer`]处理文本输入。

```py
>>> text_queries = ["hat", "book", "sunglasses", "camera"]
>>> inputs = processor(text=text_queries, images=im, return_tensors="pt")
```

将输入传递给模型，进行后处理和可视化结果。由于图像处理器在将图像馈送给模型之前会调整图像大小，因此您需要使用[`~OwlViTImageProcessor.post_process_object_detection`]方法来确保预测的边界框相对于原始图像具有正确的坐标：

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(**inputs)
...     target_sizes = torch.tensor([im.size[::-1]])
...     results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

>>> draw = ImageDraw.Draw(im)

>>> scores = results["scores"].tolist()
>>> labels = results["labels"].tolist()
>>> boxes = results["boxes"].tolist()

>>> for box, score, label in zip(boxes, scores, labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")

>>> im
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_4.png" alt="带有检测到的对象的海滩照片"/>
</div>

## 批量处理

您可以传递多组图像和文本查询来搜索一个或多个图像中的不同（或相同）对象。
让我们一起使用宇航员图像和海滩图像。
对于批量处理，您应该将文本查询作为处理器的嵌套列表传递，并将图像作为PIL图像，PyTorch张量或NumPy数组的列表传递。

```py
>>> images = [image, im]
>>> text_queries = [
...     ["human face", "rocket", "nasa badge", "star-spangled banner"],
...     ["hat", "book", "sunglasses", "camera"],
... ]
>>> inputs = processor(text=text_queries, images=images, return_tensors="pt")
```

之前后处理时使用了图片的大小的单个张量，但是也可以传递一个元组，或者在存在多张图像的情况下，传递一个元组列表。让我们为两个例子创建预测，并可视化第二个例子（`image_idx = 1`）。

```py
>>> with torch.no_grad():
...     outputs = model(**inputs)
...     target_sizes = [x.size[::-1] for x in images]
...     results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)

>>> image_idx = 1
>>> draw = ImageDraw.Draw(images[image_idx])

>>> scores = results[image_idx]["scores"].tolist()
>>> labels = results[image_idx]["labels"].tolist()
>>> boxes = results[image_idx]["boxes"].tolist()

>>> for box, score, label in zip(boxes, scores, labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{text_queries[image_idx][label]}: {round(score,2)}", fill="white")

>>> images[image_idx]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_5.png" alt="带有检测到的对象的海滩照片"/>
</div>

## 图像引导的目标检测

除了使用文本查询进行零样本目标检测外，OWL-ViT还提供了图像引导的目标检测。这意味着您可以使用图像查询在目标图像中查找相似的对象。
与文本查询不同，图像查询只允许一个示例图像。

让我们以一个有两只猫的沙发图像作为目标图像，并以单个猫的图像作为查询：

```py
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image_target = Image.open(requests.get(url, stream=True).raw)

>>> query_url = "http://images.cocodataset.org/val2017/000000524280.jpg"
>>> query_image = Image.open(requests.get(query_url, stream=True).raw)
```

让我们快速查看一下这些图像：

```py
>>> import matplotlib.pyplot as plt

>>> fig, ax = plt.subplots(1, 2)
>>> ax[0].imshow(image_target)
>>> ax[1].imshow(query_image)
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_6.png" alt="两只猫的图像"/>
</div>

在预处理步骤中，不再使用文本查询，而是需要使用`query_images`：

```py
>>> inputs = processor(images=image_target, query_images=query_image, return_tensors="pt")
```

对于预测，不再将输入传递给模型，而是将其传递给[`~OwlViTForObjectDetection.image_guided_detection`]。像以前一样绘制预测，只是现在没有标签了。

```py
>>> with torch.no_grad():
...     outputs = model.image_guided_detection(**inputs)
...     target_sizes = torch.tensor([image_target.size[::-1]])
...     results = processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)[0]

>>> draw = ImageDraw.Draw(image_target)

>>> scores = results["scores"].tolist()
>>> boxes = results["boxes"].tolist()

>>> for box, score, label in zip(boxes, scores, labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="white", width=4)

>>> image_target
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_6.png" alt="带有边界框的猫"/>
</div>

如果您想交互式地尝试与OWL-ViT进行推理，请查看此演示:

<iframe
	src="https://adirik-owl-vit.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
