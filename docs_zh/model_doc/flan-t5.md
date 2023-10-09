<!--版权所有2022年贴心面团小组。版权所有。

根据Apache License，Version 2.0（“许可证”）许可；除非符合许可证的规定，否则不得使用此文件。
您可以获取该许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证以“按原样”方式分发，
没有任何明示或暗示的担保或条件。有关的具体语言，请参见许可证
中的特定语法。

⚠️请注意，此文件是Markdown格式，但包含我们文档生成器的特定语法（类似于MDX），在您的Markdown查看器中可能无法正确渲染。

-->

# FLAN-T5

## 概述

FLAN-T5在论文[《Scaling Instruction-Finetuned Language Models》](https://arxiv.org/pdf/2210.11416.pdf)中发布-它是T5的增强版本，在多个任务的混合中进行了微调。

一个可以直接使用FLAN-T5模型权重而不需要微调模型的例子：

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

>>> inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Pour a cup of bolognese into a large bowl and add the pasta']
```

FLAN-T5包含与T5版本1.1相同的改进（有关模型改进的完整细节请参见[此处](https://huggingface.co/docs/transformers/model_doc/t5v1.1)）。

Google发布了以下变体：

- [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)

- [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)

- [google/flan-t5-large](https://huggingface.co/google/flan-t5-large)

- [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)

- [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl).

您可以参考[T5的文档页面](t5)获取所有提示、代码示例和笔记本文件。以及有关模型训练和评估的FLAN-T5模型卡片的详细信息。

原始检查点可在[此处](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)找到。