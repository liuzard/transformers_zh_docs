<!--版权 2022 HuggingFace团队。版权所有。

根据Apache许可证，版本2.0（“许可证”），除非符合
许可证。您可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，在许可证下分发的软件是根据
“原样” BASIS，无论是明示还是暗示的，也没有任何担保或条件。请参阅许可证
特定语言对于特定语言的特定语言和限制的隐含，无论是明示还是暗示的语言，该语言用于
适用许可证。

⚠️请注意，此文件是Markdown格式，但包含适用于我们的文档生成器（类似于MDX）的特定语法，可能不会
在您的Markdown查看器中正常显示。

-->

# 表格变换器

## 概述

Brandon Smock，Rohith Pesala，Robin Abraham在[PubTables-1M: Towards comprehensive table extraction from unstructured documents](https://arxiv.org/abs/2110.00061)中提出了表格变换器模型。作者引入了一个新的数据集PubTables-1M，以对比不结构化文档中的表格提取、表格结构识别和功能分析的进展。作者训练了2个[DETR](detr)模型，一个用于表格检测，一个用于表格结构识别，被称为表格变换器。

论文的摘要如下：

*最近，在将机器学习应用于从不结构化文档中推断和提取表格结构方面取得了重要进展。然而，最大的挑战之一仍然是创建具有完整、明确的大规模真实数据集。为了解决这个问题，我们开发了一个新的更全面的表格提取数据集，称为PubTables-1M。PubTables-1M包含了来自科学文章的近百万个表格，支持多种输入模态，并包含了详细的表头和位置信息，对于各种建模方法都是有用的。它还通过一种新颖的规范化过程解决了先前数据集中观察到的一个显著的真实性不一致性问题，被称为过分分割。我们证明，这些改进在训练性能上显著提高，使得对表格结构识别的模型性能评估更加可靠。此外，我们还展示了在不需要这些任务的任何特殊定制的情况下，在PubTables-1M上训练的基于transformer的目标检测模型在检测、结构识别和功能分析等三个任务上都能取得出色的结果。*

提示：

- 作者发布了两个模型，一个用于文档中的[表格检测](https://huggingface.co/microsoft/table-transformer-detection)，一个用于[表格结构识别](https://huggingface.co/microsoft/table-transformer-structure-recognition)(即识别表格中的行、列等)。
- 您可以使用[`AutoImageProcessor`] API来为模型准备图像和可选目标。这将在幕后加载一个[`DetrImageProcessor`]。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/table_transformer_architecture.jpeg"
alt="drawing" width="600"/>

<small> 明确表格检测和表格结构识别。引用自<a href="https://arxiv.org/abs/2110.00061">原始论文</a>。 </small>

此模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可以在[这里](https://github.com/microsoft/table-transformer)找到。

## 资源

<PipelineTag pipeline="object-detection"/>

- 可在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Table%20Transformer)找到用于表格变换器的演示笔记本。
- 原来图像的填充对于检测非常重要。一个有趣的Github讨论线程及作者的回复可以在[这里](https://github.com/microsoft/table-transformer/issues/68)找到。

## TableTransformerConfig

[[autodoc]] TableTransformerConfig

## TableTransformerModel

[[autodoc]] TableTransformerModel
    - forward

## TableTransformerForObjectDetection

[[autodoc]] TableTransformerForObjectDetection
    - forward