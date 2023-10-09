版权所有© 2023 HuggingFace团队。"

根据Apache许可证，第2.0版（"许可证"），您不得使用此文件，除非符合许可证的规定。您可在以下网址获取许可证副本：

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

除非适用法律另有规定或书面同意，根据许可证分发的软件是基于"按原样"的基础分发的，没有任何明示或暗示的保证或条件。请参阅许可证了解许可证下的特定语言表示和限制。


⚠️ 请注意，此文件是Markdown格式，但包含类似MDX的特定语法，可能无法在您的Markdown查看器中正确渲染。

# FLAN-UL2

## 概述

Flan-UL2是基于T5架构的编码解码模型。它使用与去年年初发布的UL2模型相同的配置。
它是通过"Flan"提示调优和数据集收集来微调的。与`Flan-T5`类似，可以直接使用FLAN-UL2的权重而无需微调模型：

根据原始博客文章，以下是明显的改进：

- 原始的UL2模型只使用512的感受野进行训练，这使得它在大量N-shot提示的情况下效果不佳。
- Flan-UL2检查点使用2048的感受野，可以更好地用于few-shot上下文学习。
- 原始的UL2模型还有模式切换令牌，这对于获得良好性能是强制要求的。然而，它们在推断或微调过程中有时会有点麻烦。在此次更新/更改中，我们在应用Flan指令调优之前，继续训练UL2 20B额外100k步（使用小批量）以遗忘"模式令牌"。这个Flan-UL2检查点不再需要模式令牌。
Google已经发布了以下变种：

可以参考[T5的文档页面](t5)获取所有提示、代码示例和笔记本。以及有关模型训练和评估更多细节的FLAN-T5模型卡。

原始检查点可以在[这里](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-ul2-checkpoints)找到。


## 在低资源设备上运行

该模型非常庞大（~40GB的半精度），所以如果您只想运行模型，请确保以8位加载您的模型，并使用`device_map="auto"`以确保您没有任何内存不足的问题！ 

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", load_in_8bit=True, device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

>>> inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['在一个大锅中，煎炒牛肉和洋葱，中火煮至熟。添加大蒜']
```

## 推断

推理协议与任何`T5`模型完全相同，请参阅[T5的文档页面](t5)以获取更多细节。