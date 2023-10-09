<!--版权2020由拥抱面团小组所有。

根据Apache许可证2.0版（“许可证”）许可；除非符合许可证规定，否则您不得使用此文件。您可以在下面的链接处获取许可证副本。

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，否则根据许可证分发的软件是按“原样”分发的，不附带任何形式的明示或暗示担保。请参阅许可证了解许可下语言的特定权限和限制。

⚠️注意，此文件为Markdown格式，但包含了我们文档构建程序（类似于MDX）的特定语法，可能无法在Markdown查看器中正确显示。

-->

# XLM-ProphetNet

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=xprophetnet">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-xprophetnet-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/xprophetnet-large-wiki100-cased-xglue-ntg">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**免责声明：**如果你看到任何奇怪的东西，请提出[a Github问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)并指派
@patrickvonplaten


## 概述

XLM-ProphetNet模型是由Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang, Ming Zhou在2020年1月13日提出的[ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063)。

XLM-ProphetNet是一个编码器-解码器模型，可以预测“ngram”语言建模中n个未来的token，而不仅仅是下一个token。它的架构与ProphetNet相同，但该模型是在多语言“wiki100”维基百科数据集上进行训练的。

根据论文摘要：

*在本文中，我们介绍了一种新的序列到序列预训练模型，称为ProphetNet，它引入了一种名为future n-gram预测的新型自监督目标和提出的n流自注意机制。ProphetNet的优化不再像传统序列到序列模型中的一步预测那样，而是通过在每个时间步上基于上下文token预测下一个n个token。未来的n-gram预测明确鼓励模型为未来的token进行规划，并防止在强大的本地相关性上过拟合。我们分别使用基础规模数据集（16GB）和大规模数据集（160GB）对ProphetNet进行预训练。然后，我们在CNN/DailyMail、Gigaword和SQuAD 1.1基准上进行抽象摘要和问题生成任务的实验。实验结果表明，与使用相同规模的预训练语料库的模型相比，ProphetNet在所有这些数据集上都取得了新的最先进结果。*

作者的代码可以在[这里](https://github.com/microsoft/ProphetNet)找到。

提示：

- XLM-ProphetNet的模型架构和预训练目标与ProphetNet相同，但XLM-ProphetNet是在跨语言数据集XGLUE上进行了预训练。

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## XLMProphetNetConfig

[[autodoc]] XLMProphetNetConfig

## XLMProphetNetTokenizer

[[autodoc]] XLMProphetNetTokenizer

## XLMProphetNetModel

[[autodoc]] XLMProphetNetModel

## XLMProphetNetEncoder

[[autodoc]] XLMProphetNetEncoder

## XLMProphetNetDecoder

[[autodoc]] XLMProphetNetDecoder

## XLMProphetNetForConditionalGeneration

[[autodoc]] XLMProphetNetForConditionalGeneration

## XLMProphetNetForCausalLM

[[autodoc]] XLMProphetNetForCausalLM