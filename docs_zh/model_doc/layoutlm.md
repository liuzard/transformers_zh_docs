<!--版权 2020 年 拥抱面部团队 保留所有权利。

根据 Apache 许可证 版本 2.0 （“许可证”）授予的许可；你除了遵守许可证外，不得使用此文件。
你可以在以下位置获取许可的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证的软件分发是基于“按原样”而不附带任何明示或暗示的担保或条件。有关的明示或暗示条件，请参见许可证下的限制。

⚠️请注意，此文件是Markdown格式，但它包含我们doc-builder的特定语法（类似于MDX），你的Markdown查看器可能无法正确呈现。

-->

# LayoutLM

<a id='Overview'></a>

## 概述

LayoutLM模型是由Yiheng Xu，Minghao Li，Lei Cui，Shaohan Huang，Furu Wei和Ming Zhou在论文[LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)中提出的。它是一种简单而有效的文本和布局预训练方法，用于文档图像理解和信息提取任务，例如表单理解和收据理解。它在几个下游任务中取得了最先进的结果：

- 表单理解：[FUNSD](https://guillaumejaume.github.io/FUNSD/)数据集（包括超过30,000个单词的199个带注释表单的集合）。
- 收据理解：[SROIE](https://rrc.cvc.uab.es/?ch=13)数据集（包括626个收据用于训练，347个收据用于测试）。
- 文档图像分类：[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)数据集（包括400,000张属于16个类别的图像）。

来自论文的摘要如下：

*近年来，预训练技术在各种NLP任务中取得了成功。尽管预训练模型在NLP应用中被广泛使用，但它们几乎只关注文本级别的操作，而忽略对于文档图像理解至关重要的布局和样式信息。在本文中，我们提出了LayoutLM，用于在扫描的文档图像中共同建模文本和布局信息的相互作用，这对于许多现实世界的文档图像理解任务（如从扫描文档中提取信息）非常有益。此外，我们还利用图像特征将单词的视觉信息整合到LayoutLM中。据我们所知，这是文本和布局首次在单一框架中进行联合学习的实例，它在几个下游任务中取得了最新的最先进的结果，包括表单理解（从70.72到79.27），收据理解（从94.02到95.24）和文档图像分类（从93.07到94.42）。

提示：

- 除了*input_ids*之外，[`~transformers.LayoutLMModel.forward`]还需要输入`bbox`，即输入标记的边界框（即2D位置）。可以使用外部OCR引擎（例如Google的[Tesseract](https://github.com/tesseract-ocr/tesseract)）获取这些边界框（有一个可用的[Python封装](https://pypi.org/project/pytesseract/)）。每个边界框的格式应为（x0，y0，x1，y1），其中（x0，y0）对应于边界框左上角的位置，（x1，y1）表示边界框右下角的位置。请注意，首先需要将边界框归一化为0-1000的比例。要归一化，可以使用以下函数：

```python
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

这里，`width`和`height`对应于令牌出现的原始文档的宽度和高度。可以使用Python Image Library（PIL）库来获取这些值，例如：

```python
from PIL import Image

# Document可以是png，jpg等。必须将PDF转换为图像。
image = Image.open(your_document的名称).convert("RGB")

width，height = image.size
```

## 资源

官方Hugging Face和社区（🌎表示）资源列表，可帮助你开始使用LayoutLM。如果你有兴趣提交资源以包含在这里，请随时打开拉取请求，我们将进行审查！该资源应理想地展示出新的东西，而不是重复现有的资源。

<PipelineTag pipeline="document-question-answering" />

- 一篇关于使用Keras和Hugging Face Transformers [Fine-tuning LayoutLM用于文档理解](https://www.philschmid.de/fine-tuning-layoutlm-keras)的博客文章。

- 一篇关于如何仅使用Hugging Face Transformers [Fine-tune LayoutLM用于文档理解](https://www.philschmid.de/fine-tuning-layoutlm)的博客文章。

- 一篇关于如何在FUNSD数据集上使用图像嵌入[Fine-tune LayoutLM](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb)的笔记本。

- 参见：[文档问答任务指南](../tasks/document_question_answering)。

<PipelineTag pipeline="text-classification" />

- 一篇关于如何在RVL-CDIP数据集上进行序列分类的[Fine-tune LayoutLM](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForSequenceClassification_on_RVL_CDIP.ipynb)的笔记本。
- [文本分类任务指南](../tasks/sequence_classification)。

<PipelineTag pipeline="token-classification" />

- 一篇关于如何在FUNSD数据集上进行令牌分类的[Fine-tune LayoutLM](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb)的笔记本。
- [令牌分类任务指南](../tasks/token_classification)。

**其他资源**
- [遮盖语言建模任务指南](../tasks/masked_language_modeling)。

🚀 部署

- 一篇关于如何[使用Hugging Face推断端点部署LayoutLM](https://www.philschmid.de/inference-endpoints-layoutlm)的博客文章。

## LayoutLMConfig

[[autodoc]] LayoutLMConfig

## LayoutLMTokenizer

[[autodoc]] LayoutLMTokenizer

## LayoutLMTokenizerFast

[[autodoc]] LayoutLMTokenizerFast

## LayoutLMModel

[[autodoc]] LayoutLMModel

## LayoutLMForMaskedLM

[[autodoc]] LayoutLMForMaskedLM

## LayoutLMForSequenceClassification

[[autodoc]] LayoutLMForSequenceClassification

## LayoutLMForTokenClassification

[[autodoc]] LayoutLMForTokenClassification

## LayoutLMForQuestionAnswering

[[autodoc]] LayoutLMForQuestionAnswering

## TFLayoutLMModel

[[autodoc]] TFLayoutLMModel

## TFLayoutLMForMaskedLM

[[autodoc]] TFLayoutLMForMaskedLM

## TFLayoutLMForSequenceClassification

[[autodoc]] TFLayoutLMForSequenceClassification

## TFLayoutLMForTokenClassification

[[autodoc]] TFLayoutLMForTokenClassification

## TFLayoutLMForQuestionAnswering

[[autodoc]] TFLayoutLMForQuestionAnswering