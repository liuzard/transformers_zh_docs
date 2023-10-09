<!--版权所有2021年The HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”）获得许可；在遵守许可证的情况下，你不得使用此文件。
你可以获得许可证的副本通过下面的链接：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件在许可证下分发
基于“AS IS”原则，不提供任何明示或暗示的担保或条件。有关许可的更多信息，请参阅许可证
特定语言限制和限制。

⚠️请注意，此文件是Markdown的，但包含特定于我们文档生成器（类似于MDX）的语法，可能
在你的Markdown查看器中无法正确渲染。

-->

# BARTpho

## 概述

BARTpho模型由Nguyen Luong Tran, Duong Minh Le和Dat Quoc Nguyen在[《BARTpho：用于越南语的预训练序列到序列模型》](https://arxiv.org/abs/2109.09701)中提出。

摘要如下：

*我们提出了两个版本的BARTpho - BARTpho_word和BARTpho_syllable - 这是针对越南语进行的第一批公开大规模单语序列到序列模型。我们的BARTpho使用了"large"架构和序列到序列去噪模型BART的预训练方案，因此特别适用于生成型自然语言处理任务。在越南文本摘要的下游任务上的实验证明，无论是自动评估还是人工评估，我们的BARTpho都优于强基准mBART，并改进了最新技术水平。我们发布了BARTpho以促进未来的研究和生成型越南自然语言处理任务的应用。*

使用示例：

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bartpho = AutoModel.from_pretrained("vinai/bartpho-syllable")

>>> tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

>>> line = "Chúng tôi là những nghiên cứu viên."

>>> input_ids = tokenizer(line, return_tensors="pt")

>>> with torch.no_grad():
...     features = bartpho(**input_ids)  # Models outputs are now tuples

>>> # With TensorFlow 2.0+:
>>> from transformers import TFAutoModel

>>> bartpho = TFAutoModel.from_pretrained("vinai/bartpho-syllable")
>>> input_ids = tokenizer(line, return_tensors="tf")
>>> features = bartpho(**input_ids)
```

提示：

- 与mBART类似，BARTpho使用BART的"large"架构，并在编码器和解码器之上添加了额外的层归一化层。因此，在BART的[文档](bart)中进行调整以适应BARTpho时，应将BART特定的类替换为mBART特定的类。例如：

```python
>>> from transformers import MBartForConditionalGeneration

>>> bartpho = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable")
>>> TXT = "Chúng tôi là <mask> nghiên cứu viên."
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
>>> logits = bartpho(input_ids).logits
>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)
>>> print(tokenizer.decode(predictions).split())
```

- 此实现仅用于分词：“monolingual_vocab_file”包含从预训练的SentencePiece模型“vocab_file”中提取出的越南语专用类型。如果其他语言将此预训练的多语言SentencePiece模型“vocab_file”用于子词划分，则可以使用它们自己的语言专用“monolingual_vocab_file”重用BartphoTokenizer。

此模型由[dqnguyen](https://huggingface.co/dqnguyen)提供。原始代码可在[此处](https://github.com/VinAIResearch/BARTpho)找到。

## BartphoTokenizer

[[autodoc]] BartphoTokenizer
