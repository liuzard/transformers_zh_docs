<!--版权所有2021年The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证中的规定，否则不得使用此文件。
你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则以“按原样”分发的软件不附带任何形式的明示或暗示担保。
有关特定语言的详细信息，请参阅许可证下的限制。

⚠️请注意，此文件的格式为Markdown，但包含我们doc-builder（类似MDX）的特定语法，这可能无法正确渲染在你的Markdown查看器中。

-->

# VisualBERT

## 概述

VisualBERT模型由Liunian Harold Li、Mark Yatskar、Da Yin、Cho-Jui Hsieh和Kai-Wei Chang在《VisualBERT: A Simple and Performant Baseline for Vision and Language》中提出。
VisualBERT是一个使用多个（图像，文本）对训练的神经网络模型。

论文中的摘要如下：

*我们提出了VisualBERT，这是一个简单灵活的框架，用于建模广泛的视觉与语言任务。
VisualBERT由一系列Transformer层组成，通过自注意力隐式地对输入文本和与之关联的输入图像的区域进行元素对齐。
我们还提出了两种在图像标题数据上预训练VisualBERT的具有视觉基础的语言模型目标。
对包括VQA、VCR、NLVR2和Flickr30K在内的四个视觉与语言任务的实验表明，VisualBERT在性能上优于或与最先进的模型相媲美，同时更简单。
进一步的分析表明，VisualBERT可以在没有任何明确监督的情况下将语言元素与图像区域相关联，甚至对句法关系（例如，跟踪动词和对应其实参的图像区域之间的关联）具有敏感性。*

提示：

1. 提供的大多数检查点都与[`VisualBertForPreTraining`]配置兼容。
   提供的其他检查点将用于下游任务的微调 - VQA（'visualbert-vqa'）、VCR（'visualbert-vcr'）、NLVR2（'visualbert-nlvr2'）。
   因此，如果你不在进行这些下游任务，建议使用预训练的检查点。

2. 对于VCR任务，作者使用一个经过微调的检测器生成视觉嵌入，对于所有的检查点。
   我们没有将该检测器及其权重作为软件包的一部分提供，但它将在研究项目中提供，并且可以直接加载到提供的检测器中。

## 使用方法

VisualBERT是一个多模态的视觉与语言模型。它可用于视觉问答、多项选择、视觉推理和区域到短语对齐任务。
VisualBERT使用类似BERT的变换器来为图像-文本对准备嵌入。然后将文本和视觉特征投影到具有相同维度的潜空间。

要将图像提供给模型，需要将每个图像通过预训练的对象检测器，并提取区域和边界框。作者使用经过预训练的CNN（如ResNet）将这些区域传递后生成的特征作为视觉嵌入。
他们还添加了绝对位置嵌入，并将结果向量序列输入到标准的BERT模型中。
文本输入在嵌入层的前部与视觉嵌入连接，并期望被[CLS]和[SEP]标记所包围，就像在BERT中一样。段ID也必须适当设置以适应文本和视觉部分。

使用[`BertTokenizer`]对文本进行编码。必须使用自定义的检测器/图像处理器来获取视觉嵌入。
以下示例笔记本展示了如何使用类似Detectron的模型使用VisualBERT：

- [VisualBERT VQA演示笔记本](https://github.com/huggingface/transformers/tree/main/examples/research_projects/visual_bert)：这个笔记本包含了VisualBERT VQA的示例。

- [生成VisualBERT嵌入（Colab笔记本）](https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing)：这个笔记本包含了如何生成视觉嵌入的示例。

以下示例展示了如何使用[`VisualBertModel`]获取最后的隐藏状态：

```python
>>> import torch
>>> from transformers import BertTokenizer, VisualBertModel

>>> model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("What is the man eating?", return_tensors="pt")
>>> # get_visual_embeddings是一个自定义函数，给定图像路径返回视觉嵌入
>>> visual_embeds = get_visual_embeddings(image_path)

>>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
>>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
>>> inputs.update(
...     {
...         "visual_embeds": visual_embeds,
...         "visual_token_type_ids": visual_token_type_ids,
...         "visual_attention_mask": visual_attention_mask,
...     }
... )
>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
```

此模型由[gchhablani](https://huggingface.co/gchhablani)贡献。原始代码可以在[这里](https://github.com/uclanlp/visualbert)找到。

## VisualBertConfig

[[autodoc]] VisualBertConfig

## VisualBertModel

[[autodoc]] VisualBertModel
    - forward

## VisualBertForPreTraining

[[autodoc]] VisualBertForPreTraining
    - forward

## VisualBertForQuestionAnswering

[[autodoc]] VisualBertForQuestionAnswering
    - forward

## VisualBertForMultipleChoice

[[autodoc]] VisualBertForMultipleChoice
    - forward

## VisualBertForVisualReasoning

[[autodoc]] VisualBertForVisualReasoning
    - forward

## VisualBertForRegionToPhraseAlignment

[[autodoc]] VisualBertForRegionToPhraseAlignment
    - forward