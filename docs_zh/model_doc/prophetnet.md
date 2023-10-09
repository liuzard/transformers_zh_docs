<!--2020年版权归HuggingFace团队所有。

根据Apache许可证第2版（“许可证”），您不得使用此文件，除非符合许可证的要求。您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按"原样"基础分发，不附带任何明示或暗示的保证或条件。有关更多信息，请参阅许可证中的特定语言。

⚠️ 请注意，此文件采用Markdown格式，但包含我们文档生成器的特定语法（类似于MDX），可能在您的Markdown查看器中无法正确显示。-->

# ProphetNet

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=prophetnet">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-prophetnet-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/prophetnet-large-uncased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**免责声明：** 如果您看到任何异常，可以在[GitHub问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)上提出并指定@patrickvonplaten。

## 概述

ProphetNet模型于2020年1月13日由Yu Yan，Weizhen Qi，Yeyun Gong，Dayiheng Liu，Nan Duan，Jiusheng Chen，Ruofei Zhang和Ming Zhou在论文《ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training》中提出。

ProphetNet是一种编码器-解码器模型，可以对“ngram”语言模型进行n个未来标记的预测，而不仅仅是下一个标记。

论文中的摘要如下：

*在本文中，我们提出了一种新的序列到序列预训练模型，称为ProphetNet，该模型引入了一种新颖的自监督目标，即未来的n-gram预测和提出的n-stream自注意机制。ProphetNet通过对传统序列到序列模型中的一步预测进行优化，而不是优化n步预测，即在每个时间步基于先前的上下文标记同时预测下一个n个标记。未来的n-gram预测明确鼓励模型规划未来的标记，并防止在强相关局部上过度拟合。我们使用基本规模数据集（16GB）和大规模数据集（160GB）分别预训练ProphetNet。然后，我们在CNN / DailyMail，Gigaword和SQuAD 1.1基准测试上进行自动摘要和问答生成任务的实验。实验结果表明，与使用相同规模预训练语料的模型相比，ProphetNet在所有这些数据集上均取得了新的最先进结果。*

提示：

- ProphetNet是一个具有绝对位置嵌入的模型，所以建议在右侧而不是左侧填充输入。
- 模型架构基于原始Transformer，但将解码器中的“标准”自注意机制替换为主要自注意机制和自注意机制和n流（预测）自注意机制。

作者的代码可以在[这里](https://github.com/microsoft/ProphetNet)找到。

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## ProphetNetConfig

[[autodoc]] ProphetNetConfig

## ProphetNetTokenizer

[[autodoc]] ProphetNetTokenizer

## ProphetNet特定输出

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput

## ProphetNetModel

[[autodoc]] ProphetNetModel
    - forward

## ProphetNetEncoder

[[autodoc]] ProphetNetEncoder
    - forward

## ProphetNetDecoder

[[autodoc]] ProphetNetDecoder
    - forward

## ProphetNetForConditionalGeneration

[[autodoc]] ProphetNetForConditionalGeneration
    - forward

## ProphetNetForCausalLM

[[autodoc]] ProphetNetForCausalLM
    - forward