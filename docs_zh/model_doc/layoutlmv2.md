<!-- 版权 2021 年的 HuggingFace 团队保留所有权利。

根据 Apache 授权证许可，版本 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ 请注意，此文件采用 Markdown 格式，但包含用于我们文档生成器的特定语法（类似于 MDX 的语法），在你的 Markdown 查看器中可能无法正确显示。

-->

# LayoutLMV2

## 总览

LayoutLMV2 模型在 [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740) 一文中提出，作者是 Yang Xu、Yiheng Xu、Tengchao Lv、Lei Cui、Furu Wei、Guoxin Wang、Yijuan Lu、
Dinei Florencio、Cha Zhang、Wanxiang Che、Min Zhang 和 Lidong Zhou。LayoutLMV2 改进了 [LayoutLM](layoutlm) 以在多个文档图像理解基准测试中获得最新的结果：

- 扫描文档的信息提取：[FUNSD](https://guillaumejaume.github.io/FUNSD/) 数据集（包含 199 个带有超过 30,000 个单词的注释表单）、[CORD](https://github.com/clovaai/cord) 数据集（包含 800 张用于训练、100 张用于验证和 100 张用于测试的收据）、[SROIE](https://rrc.cvc.uab.es/?ch=13) 数据集（包含 626 张用于训练和 347 张用于测试的收据）和 [Kleister-NDA](https://github.com/applicaai/kleister-nda) 数据集（包含来自 EDGAR 数据库的非公开协议，包括 254 份训练文档、83 份验证文档和 203 份测试文档）。
- 文档图像分类：[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) 数据集（包含 40 万个属于 16 个类别的图像）。
- 文档视觉问答：[DocVQA](https://arxiv.org/abs/2007.00398) 数据集（包含在 12,000 多个文档图像上定义的 50,000 个问题）。

论文中的摘要如下：

*由于其有效的模型架构和大规模无标注扫描/数字出生文档的优势，文本和布局的预训练在各种视觉丰富的文档理解任务中已经证明是有效的。在本文中，我们提出了 LayoutLMv2，通过在多模态框架中预训练文本、布局和图像，利用了新的模型架构和预训练任务。具体而言，LayoutLMv2 不仅使用现有的遮盖的视觉语言建模任务，还使用新的文本-图像对齐和文本-图像匹配任务，在预训练阶段更好地学习了跨模态交互。同时，它还将空间感知的自注意机制整合到 Transformer 架构中，以便模型可以充分理解不同文本块之间的相对位置关系。实验结果表明，LayoutLMv2 在各种下游基于视觉丰富的文档理解任务上优于强基准模型，并取得了新的最新成果，例如 FUNSD (0.7895 -> 0.8420)、CORD (0.9493 -> 0.9601)、SROIE (0.9524 -> 0.9781)、Kleister-NDA (0.834 -> 0.852)、RVL-CDIP (0.9443 -> 0.9564) 和 DocVQA (0.7295 -> 0.8672)。该预训练的 LayoutLMv2 模型可以在此 URL 中公开获得。*

LayoutLMv2 依赖于 `detectron2`、`torchvision` 和 `tesseract`。运行以下命令安装它们：
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install torchvision tesseract
```
（如果你是为 LayoutLMv2 开发，请注意，通过测试文档还需要安装这些软件包。）

提示：

- LayoutLMv1 和 LayoutLMv2 的主要区别在于后者在预训练过程中引入了视觉嵌入（而 LayoutLMv1 仅在微调过程中添加了视觉嵌入）。
- LayoutLMv2 在自注意层的注意分数中添加了相对的一维注意偏差和空间二维注意偏差。详细信息请参阅论文的第 5 页。
- 有关如何在 RVL-CDIP、FUNSD、DocVQA、CORD 上使用 LayoutLMv2 模型的演示笔记本，可以参考[此处](https://github.com/NielsRogge/Transformers-Tutorials)。
- LayoutLMv2 使用 Facebook AI 的 [Detectron2](https://github.com/facebookresearch/detectron2/) 包作为其视觉骨干。有关安装说明，请参阅[此链接](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)。
- 除 `input_ids` 之外，[`~LayoutLMv2Model.forward`] 预期还提供另外两个输入，即
  `image` 和 `bbox`。`image` 输入对应文本标记出现的原始文档图像。模型期望每个文档图像的大小为 224x224。这意味着，如果你有一批文档图像，则 `image` 应该是形状为 (batch_size, 3, 224, 224) 的张量。这可以是一个 `torch.Tensor` 或一个 `Detectron2.structures.ImageList`。你无需对通道进行归一化处理，因为模型会执行此操作。重要的是要注意，视觉骨干期望 BGR 通道而不是 RGB 通道，因为 Detectron2 中的所有模型都使用 BGR 格式进行了预训练。`bbox` 输入是输入文本标记的边界框（即 2D位置）。这与 [`LayoutLMModel`] 中的相同。可以使用外部 OCR 引擎（例如 Google 的 [Tesseract](https://github.com/tesseract-ocr/tesseract)）将其检索出来（有一个可用的 [Python 封装](https://pypi.org/project/pytesseract/)）。每个边界框的格式应为 (x0, y0, x1, y1)，其中 (x0, y0) 对应于边界框左上角的位置，(x1, y1) 表示边界框右下角的位置。请注意，首先需要将边界框进行正则化，使其位于 0-1000 的范围内。要进行正则化，可以使用以下函数：

```python
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

这里，`width` 和 `height` 对应于文本标记出现的原始文档的宽度和高度（调整图像大小之前）。例如，可以使用 Python 图像库（PIL）库获取这些值，例如：

```python
from PIL import Image

image = Image.open(
    "name_of_your_document - 可以是你要处理的文档的 png、jpg 等（PDF 必须转换为图像）。"
)

width, height = image.size
```

不过，该模型包括全新的 [`~transformers.LayoutLMv2Processor`]，可用于直接为模型准备数据（在幕后应用 OCR）。更多信息可以在下面的“用法”部分中找到。

- 在内部，[`~transformers.LayoutLMv2Model`]会将 `image` 输入通过其视觉骨干传递，以获取低分辨率特征图，其形状等于 [`~transformers.LayoutLMv2Config`] 的 `image_feature_pool_shape` 属性。然后，此特征图被展平以获取图像标记的序列。由于默认情况下特征图的大小为 7x7，因此一共会得到 49 个图像标记。然后，这些图像标记与文本标记连接，并通过 Transformer 编码器。这意味着，如果你将文本标记扩展到最大长度，则模型的最后隐藏状态的长度将为 512 + 49 = 561。更一般地，最后的隐藏状态将具有形状 `seq_length` + `image_feature_pool_shape[0]` *
  `config.image_feature_pool_shape[1]`。
- 在调用 [`~transformers.LayoutLMv2Model.from_pretrained`] 时，将显示一个警告，其中列出了未初始化的一长串参数名称。这并不是问题，因为这些参数是批量归一化统计数据，在自定义数据集上进行微调时将具有值。
- 如果你想在分布式环境中训练模型，请确保在调用中的模型上调用 [`synchronize_batch_norm`]，以便正确同步可视化骨干的批量归一化层。

此外，还有 LayoutXLM，它是 LayoutLMv2 的多语言版本。更多信息可以在[LayoutXLM 的文档页面](layoutxlm)找到。

## 资源

LayoutLMv2 入门的官方 Hugging Face 和社区资源列表（由 🌎 表示）。如果你有兴趣提交要包含在此处的资源，请随时提出合并请求，我们将对其进行审核！该资源应最好展示出新的内容，而不是重复现有资源。

<PipelineTag pipeline="text-classification"/>

- 有关[如何在 RVL-CDIP 数据集上对 LayoutLMv2 进行文本分类的微调的笔记本](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb)。
- 另请参阅：[文本分类任务指南](../tasks/sequence_classification)

<PipelineTag pipeline="question-answering"/>

- 有关[如何在 DocVQA 数据集上对 LayoutLMv2 进行问答的微调的笔记本](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb)。
- 另请参阅：[问题回答任务指南](../tasks/question_answering)
- 另请参阅：[文档问答任务指南](../tasks/document_question_answering)


<PipelineTag pipeline="token-classification"/>

- 有关[如何在 CORD 数据集上对 LayoutLMv2 进行令牌分类的微调的笔记本](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/CORD/Fine_tuning_LayoutLMv2ForTokenClassification_on_CORD.ipynb)。
- 有关[如何在 FUNSD 数据集上对 LayoutLMv2 进行令牌分类的微调的笔记本](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb)。
- 另请参阅：[令牌分类任务指南](../tasks/token_classification)

## 用法：LayoutLMv2Processor

准备模型数据的最简单方法是使用 [`LayoutLMv2Processor`]，它内部组合了图像处理器（[`LayoutLMv2ImageProcessor`]）和分词器（[`LayoutLMv2Tokenizer`] 或 [`LayoutLMv2TokenizerFast`]）。图像处理器处理图像模态，而分词器处理文本模态。处理器将两者结合起来，这对于像 LayoutLMv2 这样的多模态模型非常理想。注意，你仍然可以单独使用它们，如果只想处理一种模态。

```python
from transformers import LayoutLMv2ImageProcessor, LayoutLMv2TokenizerFast, LayoutLMv2Processor

image_processor = LayoutLMv2ImageProcessor()  # apply_ocr 默认为 True
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor(image_processor, tokenizer)
```

简而言之，一个可以提供文档图像（以及可能的其他数据）给 [`LayoutLMv2Processor`]，它将创建模型所需的输入。在内部，处理器首先使用 [`LayoutLMv2ImageProcessor`] 在图像上应用 OCR 以获取单词和规范化的边界框列表，并将图像调整为给定大小以获取 `image` 输入。然后，将这些单词和规范化的边界框提供给 [`LayoutLMv2Tokenizer`] 或 [`LayoutLMv2TokenizerFast`]，将其转换为标记级别的 `input_ids`、`attention_mask`、`token_type_ids` 和 `bbox`。可选地，还可以将单词标签提供给处理器，它们将转换为标记级别的 `labels`。

[`LayoutLMv2Processor`] 在幕后使用 [PyTesseract](https://pypi.org/project/pytesseract/)，这是一个围绕 Google Tesseract OCR 引擎的 Python 封装。请注意，你仍然可以使用自己选择的 OCR 引擎，并将单词和规范化的边界框提供给处理器。这要求使用 `apply_ocr` 设置为 `False` 初始化 [`LayoutLMv2ImageProcessor`]。

总的来说，处理器支持以下 5 种用例。下面列出了所有这些用例。请注意，每个这些用例都适用于批量和非批量输入（我们以非批量输入为例进行说明）。

**用例 1：文档图像分类（训练、推断）+ 令牌分类（推断），apply_ocr = True**

这是最简单的用例，在该用例中处理器（实际上是图像处理器）将对图像执行 OCR 以获取单词和规范化的边界框。

```python
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

image = Image.open(
    "name_of_your_document - 可以是你要处理的文档的 png、jpg 等（PDF 必须转换为图像）。"
).convert("RGB")
encoding = processor(
    image, return_tensors="pt"
)  # 你也可以在此处添加所有分词器参数，如 padding、截断等
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

**用例 2：文档图像分类（训练、推断）+ 令牌分类（推断），apply_ocr=False**

如果你想自行执行 OCR，可以使用 `apply_ocr` 设置为 `False` 初始化图像处理器。在这种情况下，你应该自己提供单词和相应的（规范化的）边界框，以供处理器使用。

```python
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

image = Image.open(
    "name_of_your_document - 可以是你要处理的文档的 png、jpg 等（PDF 必须转换为图像）。"
).convert("RGB")
words = ["hello", "world"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 请确保正规化你的边界框
encoding = processor(image, words, boxes=boxes, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

**用例 3：令牌分类（训练），apply_ocr=False**

对于令牌分类任务（如 FUNSD、CORD、SROIE、Kleister-NDA），还可以提供相应的单词标签以便训练模型。处理器将这些转换为标记级别的 `labels`。默认情况下，它仅标记单词的第一个词片，然后使用 -100 标记剩余词片，这是 PyTorch 的 CrossEntropyLoss 的 `ignore_index`。如果要对单词的所有词片进行标记，可以将分词器的 `only_label_first_subword` 设置为 `False`。

```python
from transformers import LayoutLMv2Processor
from PIL import Image

```markdown
将下面这句话翻译成中文，格式是markdown，<>里面的保留原文，也不要添加额外的内容：

```python
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
words = ["hello", "world"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
word_labels = [1, 2]
encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'labels', 'image'])
```

**用例 4：视觉问答（推理），apply_ocr=True**

对于视觉问答任务（如DocVQA），你可以向处理器提供一个问题。默认情况下，处理器将在图像上应用OCR，并创建[CLS]问题标记[SEP]单词标记[SEP]。

```python
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
question = "What's his name?"
encoding = processor(image, question, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

**用例 5：视觉问答（推理），apply_ocr=False**

对于视觉问答任务（如DocVQA），你可以向处理器提供一个问题。如果你想自己执行OCR，可以向处理器提供你自己的单词和（标准化的）边界框。

```python
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
question = "What's his name?"
words = ["hello", "world"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

## LayoutLMv2Config

[[autodoc]] LayoutLMv2Config

## LayoutLMv2FeatureExtractor

[[autodoc]] LayoutLMv2FeatureExtractor
    - __call__

## LayoutLMv2ImageProcessor

[[autodoc]] LayoutLMv2ImageProcessor
    - preprocess

## LayoutLMv2Tokenizer

[[autodoc]] LayoutLMv2Tokenizer
    - __call__
    - save_vocabulary

## LayoutLMv2TokenizerFast

[[autodoc]] LayoutLMv2TokenizerFast
    - __call__

## LayoutLMv2Processor

[[autodoc]] LayoutLMv2Processor
    - __call__

## LayoutLMv2Model

[[autodoc]] LayoutLMv2Model
    - forward

## LayoutLMv2ForSequenceClassification

[[autodoc]] LayoutLMv2ForSequenceClassification

## LayoutLMv2ForTokenClassification

[[autodoc]] LayoutLMv2ForTokenClassification

## LayoutLMv2ForQuestionAnswering

[[autodoc]] LayoutLMv2ForQuestionAnswering
```