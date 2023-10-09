<!--版权所有2020年The HuggingFace团队。保留所有权利。

根据Apache License, Version 2.0进行许可（"许可证"）; 除非符合
许可证，否则不得使用此文件。您可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是分发在
"AS IS" BASIS， WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND，潜在风险
无论是明示还是暗示。有关的特定语言请看许可证
详细信息和限制。

⚠️请注意，此文件是Markdown格式的，但包含我们文档生成器的特定语法（类似于MDX），可能不会在您的Markdown查看器中正确显示。

-->

# PhoBERT

## 概述

PhoBERT模型是由Dat Quoc Nguyen和Anh Tuan Nguyen在[PhoBERT: Pre-trained language models for Vietnamese](https://www.aclweb.org/anthology/2020.findings-emnlp.92.pdf)中提出的。

论文的摘要如下：

*我们提出的PhoBERT有两个版本，即PhoBERT-base和PhoBERT-large，是首个用于越南语的公开大规模单语
预训练语言模型。实验证明，PhoBERT在多个越南语特定的自然语言处理任务中，包括词性标注、依存分析、命名实体识别和
自然语言推理，与最近最优秀的预训练多语言模型XLM-R（Conneau et al., 2020）相比表现出一致的优势。*

使用示例：

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> phobert = AutoModel.from_pretrained("vinai/phobert-base")
>>> tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

>>> # 输入文本必须已被分词！
>>> line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

>>> input_ids = torch.tensor([tokenizer.encode(line)])

>>> with torch.no_grad():
...     features = phobert(input_ids)  # Models outputs are now tuples

>>> # 使用 TensorFlow 2.0+:
>>> # from transformers import TFAutoModel
>>> # phobert = TFAutoModel.from_pretrained("vinai/phobert-base")
```

此模型由[dqnguyen](https://huggingface.co/dqnguyen)提供。原始代码可以在此处找到[here](https://github.com/VinAIResearch/PhoBERT)。

## PhobertTokenizer

[[autodoc]] PhobertTokenizer