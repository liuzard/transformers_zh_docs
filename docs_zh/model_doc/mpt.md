<!--版权所有2023年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（"许可证"）授权；除非遵守许可证，否则你不得使用此文件。你可以获取许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证，分发的软件基于"原样"的基础上，没有任何形式的明示或暗示担保和条件。请参阅许可证以了解许可证下的特定语言、权限和限制。

⚠️请注意，此文件采用Markdown格式，但包含我们的文档构建器（类似于MDX）的特定语法，可能无法在你的Markdown查看器中正确呈现。-->

# MPT

## 概述

MPT模型是由[MosaicML](https://www.mosaicml.com/)团队提出的，并以多个尺寸和微调变体发布。MPT模型是一系列在1T标记上进行预训练的开源和商业可用LLM。

MPT模型是一种GPT风格的仅解码器的transformer，具有以下改进：性能优化的层实现、提供更大训练稳定性的架构变化以及通过用ALiBi替换位置嵌入来消除上下文长度限制。

- MPT base：MPT base是在下一个标记预测上对预训练模型进行了微调
- MPT instruct：MPT base模型针对基于指令的任务进行了微调
- MPT storywriter：MPT base模型在books3语料库中针对65000标记的小说摘录进行了2500步的微调，从而使模型能够处理非常长的序列

原始代码可在[`llm-foundry`](https://github.com/mosaicml/llm-foundry/tree/main)存储库中找到。

在[发布博文](https://www.mosaicml.com/blog/mpt-7b)中了解更多信息。

提示：

- 了解模型训练背后的一些技术的更多信息，请阅读`llm-foundry`存储库中的[本章节](https://github.com/mosaicml/llm-foundry/blob/main/TUTORIAL.md#faqs)。
- 如果你想使用模型的高级版本（triton内核、直接闪存注意力集成），你仍然可以通过在调用`from_pretrained`时添加`trust_remote_code=True`来使用原始的模型实现。

- [微调笔记本](https://colab.research.google.com/drive/1HCpQkLL7UXW8xJUJJ29X7QAeNJKO0frZ?usp=sharing)：详细了解如何在免费的Google Colab实例上对MPT-7B进行微调，将模型转化为聊天机器人。


## MptConfig

[[autodoc]] MptConfig
    - 全部

## MptModel

[[autodoc]] MptModel
    - forward

## MptForCausalLM

[[autodoc]] MptForCausalLM
    - forward

## MptForSequenceClassification

[[autodoc]] MptForSequenceClassification
    - forward

## MptForTokenClassification

[[autodoc]] MptForTokenClassification
    - forward

## MptForQuestionAnswering

[[autodoc]] MptForQuestionAnswering
    - forward