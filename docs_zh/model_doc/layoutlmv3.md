<!--版权S2022年HuggingFace团队。保留所有权利。

根据Apache许可证第2版（"许可证"）的规定，你不能使用此文件，除非你遵守许可证的规定。你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以其他方式分发的软件根据"AS IS"基础分发，无论明示或暗示，不提供任何形式的保证或条件，所以不是用于特定语言和行为的保证或条件。" --> 

# LayoutLMv3

## 概述

LayoutLMv3模型在[LayoutLMv3：使用统一的文本和图像屏蔽预训练文档AI](https://arxiv.org/abs/2204.08387)中提出，作者为Yupan Huang，Tengchao Lv，Lei Cui，Yutong Lu，Furu Wei。
LayoutLMv3通过使用调制词块而不是卷积神经网络骨干网络简化了[LayoutLMv2](layoutlmv2)，并且在3个目标上预训练模型：遮蔽语言建模（MLM），遮蔽图像建模（MIM）和字块对齐（WPA）。

论文的摘要如下：

*自我监督的预训练技术在文档AI中取得了显著的进展。大多数多模态预训练模型使用遮蔽语言建模目标来学习文本模态上的双向表示，但它们在图像模态的预训练目标上有所不同。这种差异增加了多模态表示学习的难度。在本文中，我们提出了LayoutLMv3，用于预训练文档AI的多模态Transformer，其中结合了文本和图像屏蔽。此外，LayoutLMv3还通过预测文本单词对应的图像词块是否被遮蔽来预训练字块对齐目标，从而学习跨模态对齐。简单的统一体系结构和训练目标使LayoutLMv3成为旨在应用于文本中心和图像中心的文档AI任务的通用预训练模型。实验结果表明，LayoutLMv3在文本中心任务（包括表单理解、收据理解和文档视觉问答）以及图像中心任务（如文档图像分类和文档布局分析）方面均取得了最先进的性能。*

提示：

- 在数据处理方面，LayoutLMv3与其前身[LayoutLMv2](layoutlmv2)几乎相同，不同之处在于：
    - 图像需要用正常的RGB格式调整大小和归一化。另一方面，LayoutLMv2在内部规范化图像，并期望通道为BGR格式。
    - 文本使用字节对编码（BPE）进行标记化，而不是WordPiece。
  由于这些数据预处理的差异，可以使用[`LayoutLMv3Processor`]，它在内部结合了[`LayoutLMv3ImageProcessor`]（用于图像模态）以及[`LayoutLMv3Tokenizer`] / [`LayoutLMv3TokenizerFast`]（用于文本模态），以准备模型的所有数据。
- 关于[`LayoutLMv3Processor`]的使用，我们建议参考其前身的[使用指南](layoutlmv2#usage-layoutlmv2processor)。
- LayoutLMv3的演示笔记本可以在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3)找到。
- 演示脚本可以在[此处](https://github.com/huggingface/transformers/tree/main/examples/research_projects/layoutlmv3)找到。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/layoutlmv3_architecture.png"
alt="drawing" width="600"/>

<small>LayoutLMv3架构。来自<a href="https://arxiv.org/abs/2204.08387">原始论文</a>。</small>

此模型由[nielsr](https://huggingface.co/nielsr)贡献。[chriskoo](https://huggingface.co/chriskoo)，[tokec](https://huggingface.co/tokec)和[lre](https://huggingface.co/lre)添加了此模型的TensorFlow版本。原始代码可以在[此处](https://github.com/microsoft/unilm/tree/master/layoutlmv3)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）提供的资源列表，可帮助你开始使用LayoutLMv3。如果你有兴趣提交资源以包含在此处，请随时提交Pull Request，我们将进行审查！资源应该最好展示出新东西，而不是重复现有资源。

<Tip>

LayoutLMv3与LayoutLMv2几乎相同，因此我们还包括了可以为LayoutLMv3任务进行调整的LayoutLMv2资源。对于这些笔记本，请务必在准备模型的数据时使用[`LayoutLMv2Processor`]。

</Tip>

<PipelineTag pipeline="text-classification"/>

- 此[notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb)支持[`LayoutLMv2ForSequenceClassification`]。
- [文本分类任务指南](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/research_projects/layoutlmv3)和[notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb)支持 [`LayoutLMv3ForTokenClassification`]。
- 用于如何使用[`LayoutLMv2ForTokenClassification`]进行推理的[笔记本](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Inference_with_LayoutLMv2ForTokenClassification.ipynb)，以及无标签情况下如何使用[`LayoutLMv2ForTokenClassification`]进行推理的[笔记本](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/True_inference_with_LayoutLMv2ForTokenClassification_%2B_Gradio_demo.ipynb)。
- 如何使用🤗训练器进行微调[`LayoutLMv2ForTokenClassification`]的[笔记本](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb)。
- [令牌分类任务指南](../tasks/token_classification)

<PipelineTag pipeline="question-answering"/>

- 此[notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb)支持[`LayoutLMv2ForQuestionAnswering`]。
- [问答任务指南](../tasks/question_answering)

**文档问答**
- [文档问答任务指南](../tasks/document_question_answering)

## LayoutLMv3Config

[[autodoc]] LayoutLMv3Config

## LayoutLMv3FeatureExtractor

[[autodoc]] LayoutLMv3FeatureExtractor
    - __call__

## LayoutLMv3ImageProcessor

[[autodoc]] LayoutLMv3ImageProcessor
    - preprocess

## LayoutLMv3Tokenizer

[[autodoc]] LayoutLMv3Tokenizer
    - __call__
    - save_vocabulary

## LayoutLMv3TokenizerFast

[[autodoc]] LayoutLMv3TokenizerFast
    - __call__

## LayoutLMv3Processor

[[autodoc]] LayoutLMv3Processor
    - __call__

## LayoutLMv3Model

[[autodoc]] LayoutLMv3Model
    - forward

## LayoutLMv3ForSequenceClassification

[[autodoc]] LayoutLMv3ForSequenceClassification
    - forward

## LayoutLMv3ForTokenClassification

[[autodoc]] LayoutLMv3ForTokenClassification
    - forward

## LayoutLMv3ForQuestionAnswering

[[autodoc]] LayoutLMv3ForQuestionAnswering
    - forward

## TFLayoutLMv3Model

[[autodoc]] TFLayoutLMv3Model
    - call

## TFLayoutLMv3ForSequenceClassification

[[autodoc]] TFLayoutLMv3ForSequenceClassification
    - call

## TFLayoutLMv3ForTokenClassification

[[autodoc]] TFLayoutLMv3ForTokenClassification
    - call

## TFLayoutLMv3ForQuestionAnswering

[[autodoc]] TFLayoutLMv3ForQuestionAnswering
    - call