<!--
版权归2022年HuggingFace团队所有。

根据Apache许可证第2.0版（“许可证”），你不得使用此文件，除非符合许可证的规定。你可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是在“原样”基础上分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以获取特定语言下的权限和限制。

⚠️ 请注意，此文件是Markdown格式，但包含了我们的文档构建器（类似于MDX）的特定语法，可能无法在你的Markdown查看器中正确呈现。

-->

# CodeGen

## 概述

CodeGen模型是由Erik Nijkamp、Bo Pang、Hiroaki Hayashi、Lifu Tu、Huan Wang、Yingbo Zhou、Silvio Savarese和Caiming Xiong在《A Conversational Paradigm for Program Synthesis》（https://arxiv.org/abs/2203.13474）中提出的。

CodeGen是一种自回归语言模型，用于在[The Pile](https://pile.eleuther.ai/)、BigQuery和BigPython上进行程序合成的顺序训练。

论文中的摘要如下：

*程序合成旨在生成一个计算机程序，作为给定问题规范的解决方案。我们提出了一种通过大型语言模型进行对话式程序合成的方法，以解决以前方法中的程序空间和用户意图规范搜索的挑战。我们的新方法将写规范和程序的过程视为用户和系统之间的多轮对话。它将程序合成视为一个序列预测问题，在其中规范以自然语言表达，并有条件地采样所需的程序。我们在自然语言和编程语言数据上训练了一系列大型语言模型，称为CodeGen。通过在数据中弱监督和数据大小和模型大小的扩大，从简单的自回归语言模型中出现了对话能力。为了研究对话式程序合成的模型行为，我们开发了一个多轮编程基准（MTPB），其中解决每个问题都需要通过用户和模型之间的多轮对话进行多步合成。我们的研究结果表明，会话能力的出现以及所提议的对话式程序合成范例的有效性。此外，我们的CodeG en模型（在TPU-v4上训练的最多16B参数）在HumanEval基准测试中胜过了OpenAI的Codex。我们将包括检查点在内的训练库JaxFormer作为开源贡献提供：[https://github.com/salesforce/codegen]（https://github.com/salesforce/codegen）。

此模型由[Hiroaki Hayashi]（https://huggingface.co/rooa）贡献。
原始代码可以在[此处](https://github.com/salesforce/codegen)找到。

## 检查点命名

* CodeGen模型的[检查点](https://huggingface.co/models?other=codegen)可用于不同的预训练数据并具有可变大小。
* 格式为：`Salesforce/codegen-{size}-{data}`，其中
  * `size`：`350M`、`2B`、`6B`、`16B`
  * `data`：
    * `nl`：在The Pile上预训练
    * `multi`：以`nl`为初始值，然后在多个编程语言数据上进行进一步预训练
    * `mono`：以`multi`为初始值，然后在Python数据上进行进一步预训练
* 例如，`Salesforce/codegen-350M-mono`是在The Pile、多个编程语言和Python上连续预训练的3.5亿参数检查点。

## 如何使用

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> checkpoint = "Salesforce/codegen-350M-mono"
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)

>>> text = "def hello_world():"

>>> completion = model.generate(**tokenizer(text, return_tensors="pt"))

>>> print(tokenizer.decode(completion[0]))
def hello_world():
    print("Hello World")

hello_world()
```

## 文档资源

- [因果语言模型任务指南](../tasks/language_modeling)

## CodeGenConfig

[[autodoc]] CodeGenConfig
- all

## CodeGenTokenizer

[[autodoc]] CodeGenTokenizer
- save_vocabulary

## CodeGenTokenizerFast

[[autodoc]] CodeGenTokenizerFast

## CodeGenModel

[[autodoc]] CodeGenModel
- forward

## CodeGenForCausalLM

[[autodoc]] CodeGenForCausalLM
- forward
