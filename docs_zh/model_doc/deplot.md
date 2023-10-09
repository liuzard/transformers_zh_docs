<!--
版权所有 © 2021 并保留原文。

根据 Apache 许可证 2.0 版（"许可证"），您无权在不符合许可证的条件下使用此文件
除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础上提供的，没有任何形式的担保或条件。有关许可证的特定语言，请参阅许可证下的相关限制。

⚠️ 请注意，此文件是 Markdown 格式，但包含专用于我们的文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确显示。

-->

# DePlot

## 概览

DePlot 在 Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun 的论文 [DePlot: 一次性视觉语言推理（将可视化转化为表格）](https://arxiv.org/abs/2212.10505) 中提出。

论文的摘要如下所述：

*图表等可视化语言在人类世界中随处可见。理解图表和绘图需要强大的推理能力。之前的最先进模型需要至少成千上万个训练示例，而它们的推理能力仍然非常有限，特别是在复杂的人类编写的查询上。本论文提出了视觉语言推理的首个一次性解决方案。我们将视觉语言推理的挑战分解为两个步骤：（1）图表文本转换，和（2）对转换后的文本进行推理。此方法的关键是模态转换模块，称为 DePlot，它将图表的图像转化为线性化的表格。DePlot 的输出可以直接用于提示预训练的大型语言模型（LLM），利用 LLM 的少样本推理能力。为了获得 DePlot，我们通过建立统一的任务格式和指标规范化了图表到表格任务，并在该任务上进行了端到端的训练。DePlot 可以与 LLM 一起直接使用，无需任何额外设置。与在28,000多个数据点上微调的最先进模型相比，仅使用一次提示的 DePlot + LLM 在来自图表问答任务的人类编写查询上实现了24.0%的改进。*

## 模型描述

DePlot 是使用 `Pix2Struct` 架构训练的模型。您可以在 [Pix2Struct 文档](https://huggingface.co/docs/transformers/main/en/model_doc/pix2struct) 中找到有关 `Pix2Struct` 的更多信息。DePlot 是 `Pix2Struct` 架构的一部分，它是视觉问答的子集。它将输入问题渲染到图像上并预测答案。

## 用法

DePlot 目前有一个可用的检查点：

- `google/deplot`：在 ChartQA 数据集上微调的 DePlot 


```python
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```

## 微调

要对 DePlot 进行微调，请参考 pix2struct 的 [微调笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb)。对于 `Pix2Struct` 模型，我们发现使用 Adafactor 优化器和余弦学习率调度程序进行微调可以实现更快的收敛：
```python
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, lr=0.01, weight_decay=1e-05)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=40000)
```