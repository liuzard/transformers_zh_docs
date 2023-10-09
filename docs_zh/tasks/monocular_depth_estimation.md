<!--版权2023 HuggingFace团队。版权所有。

根据Apache许可证，版本2.0（“许可证”）获得许可；除非符合许可证的规定，否则不得使用此文件。你可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件。有关

特定语言的权限和限制的详细信息，请参见许可证。

⚠️注意，此文件以Markdown格式编写，但包含特定于我们的文档生成器（类似于MDX）的语法，可能无法正确显示在你的Markdown查看器中。

-->

# 单目深度估计

单目深度估计是一项计算机视觉任务，涉及从单个图像中预测场景的深度信息。换句话说，它是从单个摄像机视角估计场景中对象的距离的过程。

单目深度估计具有各种应用，包括3D重建、增强现实、自动驾驶和机器人技术。这是一项具有挑战性的任务，因为它要求模型理解场景中对象之间的复杂关系和相应的深度信息，这些信息可能受到照明条件、遮挡和纹理等因素的影响。

<Tip>
本教程中所示的任务由以下模型架构支持：

<!--此提示由 `make fix-copies` 自动生成，请勿手动填写！-->

[DPT](../model_doc/dpt)，[GLPN](../model_doc/glpn)

<!--生成提示结束-->

</Tip>

在本指南中，你将了解如何：

* 创建深度估计流水线
* 手动运行深度估计推理

开始之前，请确保已安装所有必要的库：

```bash
pip install -q transformers
```

## 深度估计流水线

尝试使用支持深度估计的模型进行推理的最简单方法是使用相应的 [`pipeline`] 实例化一个流水线。
可以从[Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads)上的检查点中实例化一个流水线：

```py
>>> from transformers import pipeline

>>> checkpoint = "vinvino02/glpn-nyu"
>>> depth_estimator = pipeline("depth-estimation", model=checkpoint)
```

接下来，选择一个要分析的图像：

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-estimation-example.jpg" alt="一条繁忙街道的照片"/>
</div>

将图像传递给流水线。

```py
>>> predictions = depth_estimator(image)
```

流水线返回一个包含两个条目的字典。第一个条目称为 `predicted_depth`，是一个张量，其值表示每个像素的深度，以米为单位。
第二个条目 `depth` 是一个 PIL 图像，用于可视化深度估计结果。

让我们来看看可视化结果：

```py
>>> predictions["depth"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization.png" alt="深度估计可视化"/>
</div>

## 手动进行深度估计推理

现在，你已经了解如何使用深度估计流水线，让我们看看如何手动复制相同的结果。

从[Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads)上的检查点加载模型和相关处理器。
这里我们将使用之前相同的检查点：

```py
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation

>>> checkpoint = "vinvino02/glpn-nyu"

>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
```

使用 `image_processor` 准备图像输入，它将负责必要的图像变换，如调整大小和归一化：

```py
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values
```

将准备好的输入传递给模型：

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(pixel_values)
...     predicted_depth = outputs.predicted_depth
```

可视化结果：

```py
>>> import numpy as np

>>> # 插值到原始大小
>>> prediction = torch.nn.functional.interpolate(
...     predicted_depth.unsqueeze(1),
...     size=image.size[::-1],
...     mode="bicubic",
...     align_corners=False,
... ).squeeze()
>>> output = prediction.numpy()

>>> formatted = (output * 255 / np.max(output)).astype("uint8")
>>> depth = Image.fromarray(formatted)
>>> depth
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization.png" alt="深度估计可视化"/>
</div>