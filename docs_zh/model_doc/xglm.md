<!--版权所有2021年HuggingFace Team。保留所有权利。

根据Apache许可证，版本2.0（"许可证"），除非符合许可证的规定，否则您不得使用此文件。您可以获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于"原样"的基础分发的，
不附带任何明示或暗示的担保或条件。请查看许可证以了解特定语言下的权限和限制。

请注意，此文件是Markdown格式，但包含了我们doc-builder（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正确显示。-->

# XGLM

## 概述

XGLM模型是由Xi Victoria Lin，Todor Mihaylov，Mikel Artetxe，Tianlu Wang，Shuohui Chen，Daniel Simig，
Myle Ott，Naman Goyal，Shruti Bhosale，Jingfei Du，Ramakanth Pasunuru，Sam Shleifer，Punit Singh Koura，
Vishrav Chaudhary，Brian O'Horo，Jeff Wang，Luke Zettlemoyer，Zornitsa Kozareva，Mona Diab，Veselin Stoyanov，
Xian Li在[《Few-shot Learning with Multilingual Language Models》](https://arxiv.org/abs/2112.10668)中提出的。

论文中的摘要如下：

*大规模自回归语言模型（如GPT-3）是一种能够在不进行微调的情况下执行各种语言任务的少样本学习器。
虽然已知这些模型能够共同表示许多不同的语言，但它们的训练数据主要由英语占据，可能限制了它们的跨语言泛化能力。
在这项工作中，我们在一个覆盖多种语言的平衡语料库上训练了多语言自回归语言模型，并研究了它们在各种任务的少样本和零样本学习能力。
我们最大的模型拥有75亿个参数，在20多种代表性语言的少样本学习中取得了新的最佳结果，
在多语言常识推理方面超过了具有相似规模的GPT-3（在0样本设置中准确率提高了7.4%，在4样本设置中提高了9.4%），
在自然语言推理方面分别提高了0样本和4样本设置的准确率（分别提高了5.4%）。
在FLORES-101机器翻译基准测试中，我们的模型在32个训练示例下在182个翻译方向中有171个超过了GPT-3，
同时在45个方向上超过了官方的监督基线。我们对模型的成功和失败进行了详细分析，
特别是展示了它在某些任务上实现了跨语言的上下文学习，同时还有改进的空间，
例如在表面形式的健壮性和适应没有自然填充形式的任务方面。最后，我们在五种语言中评估了我们的模型在社交价值任务中的性能，
发现它与类似规模的GPT-3模型存在类似的局限性。*

此模型由[Suraj](https://huggingface.co/valhalla)贡献。原始代码可在[此处](https://github.com/pytorch/fairseq/tree/main/examples/xglm)找到。

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)

## XGLMConfig

[[autodoc]] XGLMConfig

## XGLMTokenizer

[[autodoc]] XGLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XGLMTokenizerFast

[[autodoc]] XGLMTokenizerFast

## XGLMModel

[[autodoc]] XGLMModel
    - forward

## XGLMForCausalLM

[[autodoc]] XGLMForCausalLM
    - forward

## TFXGLMModel

[[autodoc]] TFXGLMModel
    - call

## TFXGLMForCausalLM

[[autodoc]] TFXGLMForCausalLM
    - call

## FlaxXGLMModel

[[autodoc]] FlaxXGLMModel
    - __call__

## FlaxXGLMForCausalLM

[[autodoc]] FlaxXGLMForCausalLM
    - __call__