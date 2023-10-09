<!--
版权所有2021年The HuggingFace团队。保留所有权利。

根据Apache 2.0许可证（"许可证"）许可的；除非符合许可证，否则不能使用此文件；您可以在以下获取许可证副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律另有规定或以书面形式达成协议，否则根据许可证分发的软件均以"按原样"的方式分发，不附带任何明示或暗示的保证或条件。请参阅许可证以了解许可证下的特定语言和限制条件。

⚠️请注意，此文件以Markdown格式编写，但包含我们文档构建器的特定语法（类似于MDX），可能在您的Markdown查看器中无法正常显示。

-->

# LayoutXLM

## 概述

LayoutXLM是由Yiheng Xu，Tengchao Lv，Lei Cui，Guoxin Wang，Yijuan Lu，Dinei Florencio，Cha Zhang，Furu Wei在[LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836)中提出的，是在53种语言上训练的[LayoutLMv2模型](https://arxiv.org/abs/2012.14740)的多语言扩展。

论文中的摘要如下：

*最近，利用文本、布局和图像进行多模态预训练在视觉丰富的文档理解任务中取得了SOTA性能，这表明在不同模态之间进行联合学习具有很大的潜力。在本文中，我们提出了LayoutXLM，这是一个多模态的预训练模型，用于多语言文档理解，旨在消除视觉丰富的文档理解中的语言壁垒。为了准确评估LayoutXLM，在此还介绍了一个名为XFUN的多语言表单理解基准数据集，其中包括7种语言（中文、日文、西班牙文、法文、意大利文、德文、葡萄牙文）的表单理解样本，并为每种语言手动标记了键值对。实验结果表明，LayoutXLM模型在XFUN数据集上显著优于现有的SOTA跨语言预训练模型。*

可以直接将LayoutXLM的权重插入到LayoutLMv2模型中，如下所示：

```python
from transformers import LayoutLMv2Model

model = LayoutLMv2Model.from_pretrained("microsoft/layoutxlm-base")
```

请注意，LayoutXLM有自己的标记器，基于
[`LayoutXLMTokenizer`]/[`LayoutXLMTokenizerFast`]。可以如下初始化：

```python
from transformers import LayoutXLMTokenizer

tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
```

与LayoutLMv2类似，可以使用[`LayoutXLMProcessor`]（在内部依次应用
[`LayoutLMv2ImageProcessor`]和
[`LayoutXLMTokenizer`]/[`LayoutXLMTokenizerFast`]）来准备模型的所有数据。

由于LayoutXLM的架构与LayoutLMv2的架构相同，可以参考[LayoutLMv2的文档页面](layoutlmv2)获取所有提示、代码示例和笔记本。

此模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可以在[此处](https://github.com/microsoft/unilm)找到。


## LayoutXLMTokenizer

[[autodoc]] LayoutXLMTokenizer
    - __call__
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LayoutXLMTokenizerFast

[[autodoc]] LayoutXLMTokenizerFast
    - __call__

## LayoutXLMProcessor

[[autodoc]] LayoutXLMProcessor
    - __call__
