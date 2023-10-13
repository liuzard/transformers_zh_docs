<!--版权2023年HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”）进行许可；除非符合许可证，否则你不得使用此文件。你可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”分发，不附带任何明示或暗示的保证或条件。有关许可证的详细信息，请参阅授权。

⚠️请注意，此文件采用Markdown格式，但包含我们的文档生成器（类似于MDX）的特定语法，这可能在你的Markdown查看器中无法正确显示。

-->

# 使用torch.compile()优化推理

本指南旨在提供使用[`torch.compile()`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)为[🤗Transformers中的计算机视觉模型](https://huggingface.co/models?pipeline_tag=image-classification&library=transformers&sort=trending)引入的推理加速的基准。

## torch.compile的优点
   
根据模型和GPU的不同，`torch.compile()`在推理过程中可提高高达30％的速度。要使用`torch.compile()`，只需安装2.0以上的任何版本的`torch`。

编译模型需要时间，因此如果你只在每次推理之前编译模型一次，则可以节省时间。要在你选择的任何计算机视觉模型上编译，只需在模型上调用`torch.compile()`，如下所示：

```diff
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to("cuda")
+ model = torch.compile(model)
```

`compile()`具有多种编译模式，这些模式在编译时间和推理开销上略有不同。`max-autotune`比`reduce-overhead`花费的时间更长，但推理速度更快。默认模式对于编译来说最快，但与`reduce-overhead`相比，对于推理时间来说效率不高。在本指南中，我们使用了默认模式。你可以在[此处](https://pytorch.org/get-started/pytorch-2.0/#user-experience)了解更多信息。

我们使用`torch`的2.0.1版本针对不同的计算机视觉模型、任务、硬件类型和批处理大小进行了`torch.compile`的基准测试。

## 基准测试代码

以下是每个任务的基准测试代码。我们在进行推理之前预热GPU，并使用相同的图像进行300次推理的平均时间。

### 使用ViT进行图像分类

```python 
import torch
from PIL import Image
import requests
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to("cuda")
model = torch.compile(model)

processed_input = processor(image, return_tensors='pt').to(device="cuda")

with torch.no_grad():
    _ = model(**processed_input)

```

#### 使用DETR进行对象检测

```python 
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50").to("cuda")
model = torch.compile(model)

texts = ["a photo of a cat", "a photo of a dog"]
inputs = processor(text=texts, images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    _ = model(**inputs)
```

#### 使用Segformer进行图像分割

```python 
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to("cuda")
model = torch.compile(model)
seg_inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    _ = model(**seg_inputs)
```

以下是我们进行基准测试的模型列表。

**图像分类** 
- [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- [microsoft/beit-base-patch16-224-pt22k-ft22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k)
- [facebook/convnext-large-224](https://huggingface.co/facebook/convnext-large-224)
- [microsoft/resnet-50](https://huggingface.co/)

**图像分割** 
- [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [facebook/mask2former-swin-tiny-coco-panoptic](https://huggingface.co/facebook/mask2former-swin-tiny-coco-panoptic)
- [facebook/maskformer-swin-base-ade](https://huggingface.co/facebook/maskformer-swin-base-ade)
- [google/deeplabv3_mobilenet_v2_1.0_513](https://huggingface.co/google/deeplabv3_mobilenet_v2_1.0_513)

**对象检测** 
- [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32)
- [facebook/detr-resnet-101](https://huggingface.co/facebook/detr-resnet-101)
- [microsoft/conditional-detr-resnet-50](https://huggingface.co/microsoft/conditional-detr-resnet-50)

以下是使用`compile()`和不使用`compile()`的每个模型在不同硬件和批处理大小上的推理持续时间的可视化及百分比改进。

<div class="flex">
  <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/a100_batch_comp.png" />
  </div>
  <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/v100_batch_comp.png" />
  </div>
   <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/t4_batch_comp.png" />
  </div>
</div>

<div class="flex">
  <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/A100_1_duration.png" />
  </div>
  <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/A100_1_percentage.png" />
  </div>
</div>


![V100批量大小为1的推理持续时间比较](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/v100_1_duration.png)

![T4批量大小为4的推理持续时间百分比改进](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/T4_4_percentage.png)

以下是每个模型在不使用`compile()`和使用`compile()`的情况下的毫秒推理持续时间。请注意，OwlViT在较大的批处理大小中会导致OOM。

### A100（批量大小：1）

| **任务/模型** | **2.0版本- <br>未编译** | **2.0版本- <br>编译** |
|:---:|:---:|:---:|
| 图像分类/ViT | 9.325 | 7.584 | 
| 图像分割/Segformer | 11.759 | 10.500 |
| 对象检测/OwlViT | 24.978 | 18.420 |
| 图像分类/BeiT | 11.282 | 8.448 | 
| 对象检测/DETR | 34.619 | 19.040 |
| 图像分类/ConvNeXT | 10.410 | 10.208 | 
| 图像分类/ResNet | 6.531 | 4.124 |
| 图像分割/Mask2former | 60.188 | 49.117 |
| 图像分割/Maskformer | 75.764 | 59.487 | 
| 图像分割/MobileNet | 8.583 | 3.974 |
| 对象检测/Resnet-101 | 36.276 | 18.197 |
| 对象检测/Conditional-DETR | 31.219 | 17.993 |

### A100（批量大小：4）

| **任务/模型** | **2.0版本- <br>未编译** | **2.0版本- <br>编译** |
|:---:|:---:|:---:|
| 图像分类/ViT | 14.832 | 14.499 | 
| 图像分割/Segformer | 18.838 | 16.476 |
| 图像分类/BeiT | 13.205 | 13.048 | 
| 对象检测/DETR | 48.657 | 32.418|
| 图像分类/ConvNeXT | 22.940 | 21.631 | 
| 图像分类/ResNet | 6.657 | 4.268 |
| 图像分割/Mask2former | 74.277 | 61.781 |
| 图像分割/Maskformer | 180.700 | 159.116 | 
| 图像分割/MobileNet | 14.174 | 8.515 |
| 对象检测/Resnet-101 | 68.101 | 44.998 |
| 对象检测/Conditional-DETR | 56.470 | 35.552 |

### A100（批量大小：16）

| **任务/模型** | **2.0版本- <br>未编译** | **2.0版本- <br>编译** |
|:---:|:---:|:---:|
| 图像分类/ViT | 40.944 | 40.010 | 
| 图像分割/Segformer | 37.005 | 31.144 |
| 图像分类/BeiT | 41.854 | 41.048 | 
| 对象检测/DETR | 164.382 | 161.902 |
| 图像分类/ConvNeXT | 82.258 | 75.561 | 
| 图像分类/ResNet | 7.018 | 5.024 |
| 图像分割/Mask2former | 178.945 | 154.814 |
| 图像分割/Maskformer | 638.570 | 579.826 | 
| 图像分割/MobileNet | 51.693 | 30.310 |
| 对象检测/Resnet-101 | 232.887 | 155.021 |
| 对象检测/Conditional-DETR | 180.491 | 124.032 |

### V100（批量大小：1）

| **任务/模型** | **2.0版本- <br>未编译** | **2.0版本- <br>编译** |
|:---:|:---:|:---:|
| 图像分类/ViT | 10.495 | 6.00 | 
| 图像分割/Segformer | 13.321 | 5.862 | 
| 对象检测/OwlViT | 25.769 | 22.395 | 
| 图像分类/BeiT | 11.347 | 7.234 | 
| 对象检测/DETR | 33.951 | 19.388 |
| 图像分类/ConvNeXT | 11.623 | 10.412 | 
| 图像分类/ResNet | 6.484 | 3.820 |
| 图像分割/Mask2former | 64.640 | 49.873 |
| 图像分割/Maskformer | 95.532 | 72.207 | 
| 图像分割/MobileNet | 9.217 | 4.753 |
| 对象检测/Resnet-101 | 52.818 | 28.367 |
| 对象检测/Conditional-DETR | 39.512 | 20.816 |

### V100（批量大小：4）

| **任务/模型** | **2.0版本- <br>未编译** | **2.0版本- <br>编译** |
|:---:|:---:|:---:|
| 图像分类/ViT | 15.181 | 14.501 | 
| 图像分割/Segformer | 16.787 | 16.188 |
| 图像分类/BeiT | 15.171 | 14.753 | 
| 对象检测/DETR | 88.529 | 64.195 |
| 图像分类/ConvNeXT | 29.574 | 27.085 | 
| 图像分类/ResNet | 6.109 | 4.731 |
| 图像分割/Mask2former | 90.402 | 76.926 |
| 图像分割/Maskformer | 234.261 | 205.456 | 
| 图像分割/MobileNet | 24.623 | 14.816 |
| 对象检测/Resnet-101 | 134.672 | 101.304 |
| 对象检测/Conditional-DETR | 97.464 | 69.739 |

### V100（批量大小：16）

| **任务/模型** | **2.0版本- <br>未编译** | **2.0版本- <br>编译** |
|:---:|:---:|:---:|
| 图像分类/ViT | 52.209 | 51.633 | 
| 图像分割/Segformer | 61.013 | 55.499 |
| 图像分类/BeiT | 53.938 | 53.581  |
| 对象检测/DETR | OOM | OOM |
| 图像分类/ConvNeXT | 109.682 | 100.771 | 
| 图像分类/ResNet | 14.857 | 12.089 |
| 图像分割/Mask2former | 249.605 | 222.801 |
| 图像分割/Maskformer | 831.142 | 743.645 | 
| 图像分割/MobileNet | 93.129 | 55.365 |
| 对象检测/Resnet-101 |  1619.505 | 1262.758 | 
| 对象检测/Conditional-DETR | 1137.513 | 897.390|

## PyTorch最新版本
我们还在夜间测量了PyTorch最新版本（2.1.0dev，下载地址[https://download.pytorch.org/whl/nightly/cu118](https://download.pytorch.org/whl/nightly/cu118)），观察到未编译和已编译模型的延迟都有所改善。

### A100

| **任务/模型** | **批量大小** | **2.0版本-<br>未编译** | **2.0版本-<br>编译** |
|:---:|:---:|:---:|:---:|
| 图像分类/BeiT | 非批处理 | 12.462 | 6.954 | 
| 图像分类/BeiT | 4 | 14.109 | 12.851 | 
| 图像分类/BeiT | 16 | 42.179 | 42.147 | 
| 对象检测/DETR | 非批处理 | 30.484 | 15.221 |
| 对象检测/DETR | 4 | 46.816 | 30.942 |
| 对象检测/DETR | 16 | 163.749 | 163.706  |

### T4

| **任务/模型** | **批量大小** | **2.0版本-<br>未编译** | **2.0版本-<br>编译** |
|:---:|:---:|:---:|:---:|
| 图像分类/BeiT | 非批处理 | 14.408 | 14.052 | 
| 图像分类/BeiT | 4 | 47.381 | 46.604 | 
| 图像分类/BeiT | 16 | 42.179 | 42.147  | 
| 对象检测/DETR | 非批处理 | 68.382 | 53.481 |
| 对象检测/DETR | 4 | 269.615 | 204.785 |
| 对象检测/DETR | 16 | OOM | OOM   |

### V100

| **任务/模型** | **批量大小** | **2.0版本-<br>未编译** | **2.0版本-<br>编译** |
|:---:|:---:|:---:|:---:|
| 图像分类/BeiT | 非批处理 | 13.477 | 7.926 | 
| 图像分类/BeiT | 4 | 15.103 | 14.378 | 
| 图像分类/BeiT | 16 | 52.517 | 51.691  | 
| 对象检测/DETR | 非批处理 | 28.706 | 19.077 |
| 对象检测/DETR | 4 | 88.402 | 62.949|
| 对象检测/DETR | 16 | OOM | OOM  |


| **任务/模型** | **批大小** | **torch 2.0 - 无编译** | **torch 2.0 - 编译** |
|:---:|:---:|:---:|:---:|
| 图像分类/ConvNeXT | 未分批 | 11.758 | 7.335 | 
| 图像分类/ConvNeXT | 4 | 23.171 | 21.490 | 
| 图像分类/ResNet | 未分批 | 7.435 | 3.801 | 
| 图像分类/ResNet | 4 | 7.261 | 2.187 | 
| 目标检测/Conditional-DETR | 未分批 | 32.823 | 11.627  | 
| 目标检测/Conditional-DETR | 4 | 50.622 | 33.831  | 
| 图像分割/MobileNet | 未分批 | 9.869 | 4.244 |
| 图像分割/MobileNet | 4 | 14.385 | 7.946 |

### T4

| **任务/模型** | **批大小** | **torch 2.0 - 无编译** | **torch 2.0 - 编译** | 
|:---:|:---:|:---:|:---:|
| 图像分类/ConvNeXT | 未分批 | 32.137 | 31.84 | 
| 图像分类/ConvNeXT | 4 | 120.944 | 110.209 | 
| 图像分类/ResNet | 未分批 | 9.761 | 7.698 | 
| 图像分类/ResNet | 4 | 15.215 | 13.871 | 
| 目标检测/Conditional-DETR | 未分批 | 72.150 | 57.660  | 
| 目标检测/Conditional-DETR | 4 | 301.494 | 247.543  | 
| 图像分割/MobileNet | 未分批 | 22.266 | 19.339  |
| 图像分割/MobileNet | 4 | 78.311 | 50.983 |