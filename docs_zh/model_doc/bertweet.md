<!--版权2020年HuggingFace团队。版权所有。

根据Apache许可证2.0版（“许可证”），您除非符合许可证的规定，否则不得使用此文件。
您可以获得许可证副本，请参阅

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是按照“原样” BASIS分发的，不附有任何明示或暗示的保证或条件。
有关许可证下的特定语言的适用性的详细信息和限制，请参阅许可证。

⚠️ 请注意，此文件采用Markdown格式，但包含用于我们的doc-builder的特定语法（类似于MDX），可能无法在Markdown查看器中正确显示。

-->

# BERTweet

## 概述

BERTweet模型由Dat Quoc Nguyen，Thanh Vu和Anh Tuan Nguyen在[《BERTweet: A pre-trained language model for English Tweets》](https://www.aclweb.org/anthology/2020.emnlp-demos.2.pdf)中提出。

来自论文的摘要如下：

*我们提出了BERTweet，这是首个用于英文推文的公开大规模预训练语言模型。我们的BERTweet具有与BERT-base（Devlin et al., 2019）相同的架构，采用RoBERTa预训练过程进行训练（Liu et
al., 2019）。实验表明，BERTweet在三个推文NLP任务（词性标注、命名实体识别和文本分类）上优于强基线RoBERTa-base和XLM-R-base（Conneau et al.,
2020），产生了比之前最先进模型更好的性能结果。*

使用示例：

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

>>> # 对于transformers v4.x+版本:
>>> tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

>>> # 对于transformers v3.x版本:
>>> # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

>>> # 输入推文已经被规范化！
>>> line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

>>> input_ids = torch.tensor([tokenizer.encode(line)])

>>> with torch.no_grad():
...     features = bertweet(input_ids)  # Models outputs are now tuples

>>> # 对于TensorFlow 2.0+版本:
>>> # from transformers import TFAutoModel
>>> # bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")
```

此模型由[dqnguyen](https://huggingface.co/dqnguyen)贡献。原始代码可以在[这里](https://github.com/VinAIResearch/BERTweet)找到。

## BertweetTokenizer

[[autodoc]] BertweetTokenizer