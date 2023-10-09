<!--版权所有2020年The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；在符合许可证的情况下，你可能不使用此文件。
你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则按“原样”方式分发的软件仅基于许可证分发
无论是明示或暗示的，也没有任何形式的保证或条件。请参阅许可证以获取
许可下的限制和语言的特定语法
渲染你的Markdown查看器中的说明： />

# I-BERT

## 概述

I-BERT模型在[《I-BERT：仅整型BERT量化》](https://arxiv.org/abs/2101.01321)中提出，作者为Sehoon Kim，Amir Gholami，Zhewei Yao，Michael W. Mahoney和Kurt Keutzer。它是RoBERTa的量化版本，推理速度快了四倍。

摘要如下：

*基于Transformer的模型（如BERT和RoBERTa）在许多自然语言处理任务中取得了最先进的结果。然而，它们的内存占用量，推理延迟和功耗对于边缘的有效推理，甚至对于数据中心来说是禁止的。虽然量化可以解决这个问题，但是之前在量化基于Transformer的模型方面的工作在推理过程中使用浮点运算，这种运算无法有效利用最近的Turing Tensor Cores等仅整型逻辑单元，或者传统的仅整型ARM处理器。在这项工作中，我们提出了I-BERT，这是一种新的Transformer基础模型的量化方案，它使用了只有整数运算的完整推理。基于轻量级仅整型近似方法用于非线性操作，例如GELU，Softmax和Layer Normalization，I-BERT执行端到端的只有整数的BERT推理，没有任何浮点计算。我们使用RoBERTa-Base/Large在GLUE的下游任务上评估了我们的方法。我们表明，对于两种情况，I-BERT的准确性与全精度基准相比达到了相似的（稍微更高的）准确性。此外，我们对在T4 GPU系统上进行INT8推理的I-BERT的初步实现展示了相对于FP32推理的2.4-4.0倍加速。该框架已在PyTorch中开发并开源。*

此模型由[kssteven](https://huggingface.co/kssteven)贡献。原始代码可以在[这里](https://github.com/kssteven418/I-BERT)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/masked_language_modeling)

## IBertConfig

[[autodoc]] IBertConfig

## IBertModel

[[autodoc]] IBertModel
    - forward

## IBertForMaskedLM

[[autodoc]] IBertForMaskedLM
    - forward

## IBertForSequenceClassification

[[autodoc]] IBertForSequenceClassification
    - forward

## IBertForMultipleChoice

[[autodoc]] IBertForMultipleChoice
    - forward

## IBertForTokenClassification

[[autodoc]] IBertForTokenClassification
    - forward

## IBertForQuestionAnswering

[[autodoc]] IBertForQuestionAnswering
    - forward