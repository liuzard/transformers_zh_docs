<!--版权2023由HuggingFace团队保留。

根据Apache许可证第2版（“许可证”）的规定，除非符合许可证的规定，否则您不得使用此文件。
您可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或默示的保证或条件。请参阅许可证以了解许可证下语言的特定权限和限制。

⚠ 注意，此文件是Markdown格式的，但包含特定的语法，用于我们的文档构建器（类似于MDX），可能不会在您的Markdown查看器中正确显示。

-->

# SAM

## 概述

SAM（Segment Anything Model）由Alexander Kirillov，Eric Mintun，Nikhila Ravi，Hanzi Mao，Chloe Rolland，Laura Gustafson，Tete Xiao，Spencer Whitehead，Alex Berg，Wan-Yen Lo，Piotr Dollar和Ross Girshick在《Segment Anything》一文中提出。

该模型可用于预测给定输入图像中任何感兴趣对象的分割掩码。

![实例图像](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-output.png)

摘要如下：

*我们介绍了Segment Anything（SA）项目：一个用于图像分割的新任务、模型和数据集。在使用我们的高效模型进行数据收集循环时，我们构建了迄今为止最大的分割数据集，拥有超过10亿个掩码并涵盖1100万张受许可和尊重隐私的图像。该模型被设计和训练成可提示，因此它可以在新的图像分布和任务上进行零样本转移。我们对其在众多任务上的能力进行了评估，并发现其零样本性能令人印象深刻，往往与甚至优于之前的完全监督结果竞争。我们发布了Segment Anything Model（SAM）和相应的数据集（SA-1B），其中包含10亿个掩码和1100万张图像，网址为[https://segment-anything.com](https://segment-anything.com)，以促进计算机视觉基础模型的研究。*

提示：

- 该模型会根据输入的图像预测二进制掩码，指示感兴趣对象的存在与否。
- 如果提供输入的二维点和/或输入的边界框，则模型可以预测更好的结果。
- 可以为同一图像提示多个点，并预测一个单独的掩码。
- 目前还不支持对模型进行微调。
- 根据论文，在此编写时支持文本输入。但是，根据[官方存储库的说明](https://github.com/facebookresearch/segment-anything/issues/4#issuecomment-1497626844)，目前似乎不支持文本输入。

此模型由[ybelkada](https://huggingface.co/ybelkada)和[ArthurZ](https://huggingface.co/ArthurZ)贡献。
原始代码可在[此处](https://github.com/facebookresearch/segment-anything)找到。

以下是给定图像和二维点的掩码生成示例：

```python
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 图像中窗口的二维位置

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

资源：

- [演示笔记本](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb)用于使用该模型。
- [演示笔记本](https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb)用于使用自动掩码生成流程。
- [演示笔记本](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb)用于在医疗领域上推断使用MedSAM的示例。
- [演示笔记本](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb)用于在自定义数据上对模型进行微调。

## SamConfig

[[autodoc]] SamConfig

## SamVisionConfig

[[autodoc]] SamVisionConfig

## SamMaskDecoderConfig

[[autodoc]] SamMaskDecoderConfig

## SamPromptEncoderConfig

[[autodoc]] SamPromptEncoderConfig


## SamProcessor

[[autodoc]] SamProcessor


## SamImageProcessor

[[autodoc]] SamImageProcessor


## SamModel

[[autodoc]] SamModel
    - forward


## TFSamModel

[[autodoc]] TFSamModel
    - call