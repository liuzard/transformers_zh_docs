<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”），您不得使用此文件，除非符合许可证的要求。您可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于“按原样”分发的基础上，不附带任何明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言、条件和限制。

⚠️请注意，此文件是Markdown格式的，但包含了我们的文档生成器（类似MDX）的特定语法，可能无法在您的Markdown查看器中正确渲染。-->

# BertGeneration

## 概述

BertGeneration模型是一种BERT模型，可以在[`EncoderDecoderModel`]中利用它进行序列到序列任务，如[Sascha Rothe, Shashi Narayan, Aliaksei Severyn](https://arxiv.org/abs/1907.12461)提出的“利用预训练检查点进行序列生成任务”。

论文的摘要如下：

*大规模神经模型的无监督预训练最近彻底改变了自然语言处理。通过从公开发布的检查点开始，自然语言处理专家在多个基准测试中推动了最新的技术发展，同时节省了大量的计算时间。到目前为止，主要关注的是自然语言理解任务。在本文中，我们证明了预训练检查点对于序列生成任务的有效性。我们开发了一种基于Transformer的序列到序列模型，兼容公开可用的预训练BERT、GPT-2和RoBERTa检查点，并对使用这些检查点初始化我们的模型（编码器和解码器）的实用性进行了广泛的实证研究。我们的模型在机器翻译、文本摘要、句子拆分和句子融合等任务中取得了最新的技术成果。*

用法：

- 该模型可以与[`EncoderDecoderModel`]结合使用，以利用两个预训练BERT检查点进行后续微调。

```python
>>> # 利用Bert2Bert模型的检查点...
>>> # 使用BERT的cls标记作为BOS标记，使用sep标记作为EOS标记
>>> encoder = BertGenerationEncoder.from_pretrained("bert-large-uncased", bos_token_id=101, eos_token_id=102)
>>> # 添加互注意层，并使用BERT的cls标记作为BOS标记，使用sep标记作为EOS标记
>>> decoder = BertGenerationDecoder.from_pretrained(
...     "bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
... )
>>> bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

>>> # 创建分词器...
>>> tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

>>> input_ids = tokenizer(
...     "This is a long article to summarize", add_special_tokens=False, return_tensors="pt"
... ).input_ids
>>> labels = tokenizer("This is a short summary", return_tensors="pt").input_ids

>>> # 训练...
>>> loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
>>> loss.backward()
```

- 预训练的[`EncoderDecoderModel`]也可以直接在模型中心获得，例如：

```python
>>> # 实例化句子融合模型
>>> sentence_fuser = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
>>> tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

>>> input_ids = tokenizer(
...     "This is the first sentence. This is the second sentence.", add_special_tokens=False, return_tensors="pt"
... ).input_ids

>>> outputs = sentence_fuser.generate(input_ids)

>>> print(tokenizer.decode(outputs[0]))
```

提示：

- [`BertGenerationEncoder`]和[`BertGenerationDecoder`]应与[`EncoderDecoder`]结合使用。
- 对于摘要、句子拆分、句子融合和翻译，输入不需要特殊的标记。因此，输入末尾不应添加EOS标记。

此模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。原始代码可以在[此处](https://tfhub.dev/s?module-type=text-generation&subtype=module,placeholder)找到。

## BertGenerationConfig

[[autodoc]] BertGenerationConfig

## BertGenerationTokenizer

[[autodoc]] BertGenerationTokenizer
    - save_vocabulary

## BertGenerationEncoder

[[autodoc]] BertGenerationEncoder
    - forward

## BertGenerationDecoder

[[autodoc]] BertGenerationDecoder
    - forward