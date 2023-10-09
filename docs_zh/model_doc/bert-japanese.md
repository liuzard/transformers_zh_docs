<!--版权所有2020年The HuggingFace Team。
根据Apache许可证，第2版（"许可证"）获得许可；在遵守许可证的情况下，您不得使用本文件。
您可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，否则依据许可证分发的软件是基于"AS IS"的基础，不附带任何形式的保证或条件，无论是明示还是暗示。请参阅许可证获取特定语言下的更多限制和条件。

请注意，此文件是Markdown格式，但包含我们的文档生成器（类似MDX）的特定语法，可能在您的Markdown查看器中无法正确呈现。

-->

# BertJapanese

## 概览

在日本文本上训练的BERT模型。

这里有两种不同的分词方法的模型：

- 用MeCab和WordPiece进行分词。这需要一些额外的依赖项，[fugashi](https://github.com/polm/fugashi) 是[MeCab](https://taku910.github.io/mecab/)的一个包装器。
- 按字符进行分词。

要使用*MecabTokenizer*，您应该安装依赖项 `pip install transformers["ja"]`（或者如果您从源代码安装，可以使用 `pip install -e .["ja"]`）。

请参阅 [details on cl-tohoku repository](https://github.com/cl-tohoku/bert-japanese)。

使用采用MeCab和WordPiece分词的模型的示例：

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

>>> ## 输入的日文文本
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾輩 は 猫 で ある 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

使用按字符分词的模型的示例：

```python
>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

>>> ## 输入的日文文本
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾 輩 は 猫 で あ る 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

提示：

- 这个实现与BERT相同，除了分词方法。有关更多用法示例，请参阅 [BERT文档](bert)。

此模型由 [cl-tohoku](https://huggingface.co/cl-tohoku) 贡献。

## BertJapaneseTokenizer

[[autodoc]] BertJapaneseTokenizer