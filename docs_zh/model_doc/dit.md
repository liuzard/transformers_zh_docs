<!--版权所有2022年The HuggingFace团队。

根据Apache许可证第2版（"许可证"）获得许可；除非符合许可证，否则你不得使用此文件。你可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证发布的软件是"按原样"分发的，不附带任何担保或条件。参见许可证规定的特定语言及限制。

⚠️ 请注意，此文件是Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），在你的Markdown查看器中可能无法正确呈现。

-->

# DiT

## 概述

[DiT：面向文档图像Transformer的自监督预训练](https://arxiv.org/abs/2203.02378)是由Junlong Li，Yiheng Xu，Tengchao Lv，Lei Cui，Cha Zhang，Furu Wei于[BEiT](beit)（图像Transformer的BERT预训练）的自监督目标应用于4200万个文档图像中，可在以下任务中实现最先进的结果：

- 文档图像分类：[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)数据集（包含16个类别的40万张图片）。
- 文档布局分析：[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)数据集（由自动解析PubMed XML文件构建的超过36万个文档图像的集合）。
- 表格检测：[ICDAR 2019 cTDaR](https://github.com/cndplab-founder/ICDAR2019_cTDaR)数据集（包含600个训练图像和240个测试图像）。

论文摘要如下：

*Image Transformer最近在自然图像理解方面取得了重大进展，无论是使用有监督（ViT、DeiT等）还是自监督（BEiT、MAE等）的预训练技术。在本文中，我们提出了DiT，这是一种自监督预训练的文档图像Transformer模型，使用大规模无标签文本图像进行文档AI任务，这一点很重要，因为由于缺少人工标记的文档图像，从未存在过有监督的对应物。我们将DiT作为骨干网络在各种基于视觉的文档AI任务中使用，包括文档图像分类、文档布局分析以及表格检测。实验证明，经过自监督的预训练DiT模型在这些下游任务上取得了新的最先进结果，例如文档图像分类（91.11 → 92.69）、文档布局分析（91.0 → 94.9）和表格检测（94.23 → 96.55）。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dit_architecture.jpg"
alt="drawing" width="600"/> 

<small>方法概述。来源：[原始论文](https://arxiv.org/abs/2203.02378)。</small>

你可以直接使用DiT的权重和AutoModel API：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("microsoft/dit-base")
```

这将加载在遮蔽图像建模方面进行预训练的模型。请注意，这不包括用于预测视觉令牌的语言建模头部。

如果要包含头部，可以将权重加载到`BeitForMaskedImageModeling`模型中，如下所示：

```python
from transformers import BeitForMaskedImageModeling

model = BeitForMaskedImageModeling.from_pretrained("microsoft/dit-base")
```

你还可以从[hub](https://huggingface.co/models?other=dit)中加载一个经过微调的模型，如下所示：

```python
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
```

这个特定的检查点在[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)上进行了微调，这是一个重要的文档图像分类基准。
这里可以找到一份演示文档图像分类推理的笔记本[here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DiT/Inference_with_DiT_(Document_Image_Transformer)_for_document_image_classification.ipynb)。

由于DiT的架构与BEiT相同，可以参考[BEiT的文档页面](beit)获取所有的技巧、代码示例和笔记本。

此模型由[nielsr](https://huggingface.co/nielsr)提供。原始代码可以在[此处](https://github.com/microsoft/unilm/tree/master/dit)找到。

## 资源

以下是官方Hugging Face和社区（🌎）资源列表，可帮助你开始使用DiT。

<PipelineTag pipeline="image-classification"/>

- 本[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持`BeitForImageClassification`。

如果你有兴趣提交要包含在此处的资源，请随时发起拉取请求，我们将进行审核！该资源应该展示出新的东西，而不是重复现有的资源。