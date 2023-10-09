<!--2021年版权归“拥抱面部”团队所有。

根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。
您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，不附带任何形式的明示或暗示的保证。请参阅许可证以获取特定语言下的权限和限制。

⚠️请注意，该文件是Markdown格式的，但包含专为我们的文档生成器（类似MDX）的特定语法，可能在您的Markdown查看器中无法正确显示。

-->

# MatCha

## 概述

MatCha是由Fangyu Liu、Francesco Piccinno、Syrine Krichene、Chenxi Pang、Kenton Lee、Mandar Joshi、Yasemin Altun、Nigel Collier和Julian Martin Eisenschlos在论文[MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering](https://arxiv.org/abs/2212.09662)中提出的。论文摘要如下：

*视觉语言数据如图表和信息图在人类世界中随处可见。然而，最先进的视觉语言模型在处理这些数据时表现不佳。我们提出MatCha（Math reasoning and Chart derendering pretraining），以提升视觉语言模型在联合建模图表/图和语言数据方面的能力。具体而言，我们提出了几个涵盖图表拆解和数值推理的预训练任务，这些任务是视觉语言建模的关键能力。我们从最近提出的图像到文本视觉语言模型Pix2Struct开始进行MatCha预训练。在PlotQA和ChartQA等标准基准测试中，MatCha模型的表现超过了最先进的方法近20%。我们还研究了MatCha预训练在屏幕截图、教科书图表和文档图表等领域的迁移情况，并观察到整体改进，验证了MatCha预训练在更广泛的视觉语言任务中的实用性。*

## 模型描述

MatCha是使用`Pix2Struct`架构进行训练的模型。您可以在[Pix2Struct文档](https://huggingface.co/docs/transformers/main/en/model_doc/pix2struct)中找到有关`Pix2Struct`的更多信息。MatCha是`Pix2Struct`架构的视觉问答子集，它将输入的问题渲染到图像上并预测答案。

## 使用方法

当前有6个MatCha的检查点可用：

- `google/matcha`：基本的MatCha模型，用于在下游任务上对MatCha进行微调。
- `google/matcha-chartqa`：在ChartQA数据集上对MatCha模型进行微调。可用于回答关于图表的问题。
- `google/matcha-plotqa-v1`：在PlotQA数据集上对MatCha模型进行微调。可用于回答关于图表的问题。
- `google/matcha-plotqa-v2`：在PlotQA数据集上对MatCha模型进行微调。可用于回答关于图表的问题。
- `google/matcha-chart2text-statista`：在Statista数据集上对MatCha模型进行微调。
- `google/matcha-chart2text-pew`：在Pew数据集上对MatCha模型进行微调。

在`chart2text-pew`和`chart2text-statista`上微调的模型更适合进行摘要，而在`plotqa`和`chartqa`上微调的模型更适合进行问答。

您可以按照以下方式使用这些模型（以ChatQA数据集为例）：

```python
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa").to(0)
processor = AutoProcessor.from_pretrained("google/matcha-chartqa")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/20294671002019.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Is the sum of all 4 places greater than Laos?", return_tensors="pt").to(0)
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```

## 微调

有关微调MatCha的详细信息，请参考pix2struct的[微调笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb)。对于`Pix2Struct`模型，我们发现使用Adafactor和余弦学习率调度程序对模型进行微调能够加快收敛速度：

```python
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, lr=0.01, weight_decay=1e-05)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=40000)
```