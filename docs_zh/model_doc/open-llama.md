<!--
版权所有 2023 The HuggingFace团队。

根据Apache许可证第2版（“许可证”）许可；除非符合许可证，否则不得使用此文件。

您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件以"原样"分发，不附带任何明示或暗示的保证或条件。请参阅许可证以获取详细的管理权限和限制。

⚠️请注意，此文件使用Markdown格式，但包含我们的文档构建器的特定语法（类似于MDX），这可能无法在您的Markdown查看器中正常显示。
-->

# Open-Llama

<Tip warning={true}>

此模型仅处于维护模式，因此我们不会接受任何修改其代码的新PR。

如果您在运行此模型时遇到任何问题，请重新安装支持此模型的最后一个版本：v4.31.0。
您可以通过运行以下命令进行安装：`pip install -U transformers==4.31.0`。

</Tip>

<Tip warning={true}>

该模型与Hugging Face Hub上的[OpenLLaMA模型](https://huggingface.co/models?search=openllama)不同，Hugging Face Hub上的模型主要使用[LLaMA](llama)架构。

</Tip>

## 概述

Open-Llama模型是由社区开发者s-JoL在[Open-Llama项目](https://github.com/s-JoL/Open-Llama)中提出的。

该模型主要基于LLaMA进行了一些修改，包括了Xformers的内存高效注意力，Bloom的稳定嵌入和PaLM的共享输入-输出嵌入。
此外，该模型在中文和英文上都进行了预训练，从而在中文任务上具有更好的性能。

此模型由[s-JoL](https://huggingface.co/s-JoL)贡献。
原始代码可在[Open-Llama](https://github.com/s-JoL/Open-Llama)中找到。
检查点和用法可以在[s-JoL/Open-Llama-V1](https://huggingface.co/s-JoL/Open-Llama-V1)中找到。

## OpenLlamaConfig

[[autodoc]] OpenLlamaConfig

## OpenLlamaModel

[[autodoc]] OpenLlamaModel
    - forward

## OpenLlamaForCausalLM

[[autodoc]] OpenLlamaForCausalLM
    - forward

## OpenLlamaForSequenceClassification

[[autodoc]] OpenLlamaForSequenceClassification
    - forward
