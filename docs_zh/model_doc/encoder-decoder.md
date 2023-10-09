<!--版权保留 2020 The HuggingFace Team. All rights reserved.

根据 Apache License, Version 2.0 进行许可（“许可证”）；除非符合许可证的规定，
否则你不得使用此文件。你可以在http://www.apache.org/licenses/LICENSE-2.0获得许可证副本。

请注意，此文件是Markdown格式的，但包含我们的文档生成器的特定语法（类似于MDX），可能无法正常呈现在你的Markdown查看器中。

-->

# 编码器解码器模型

## 概述

[`EncoderDecoderModel`]可用于初始化序列到序列模型，其中编码器使用任何预训练的自编码模型，
解码器使用任何预训练的自回归模型。

在Sascha Rothe、Shashi Narayan和Aliaksei Severyn的[《Leveraging Pre-trained Checkpoints for Sequence Generation Tasks》](https://arxiv.org/abs/1907.12461)一文中，
展示了使用预训练检查点初始化序列到序列模型对于序列生成任务的有效性。

在训练/微调了此类[`EncoderDecoderModel`]之后，可以像使用其他模型一样将其保存/加载（有关更多信息，请参见示例）。

这种架构的应用可以是利用两个预训练的[`BertModel`]作为编码器和解码器，用于摘要模型，如[Yang Liu和Mirella Lapata的《Text Summarization with Pretrained Encoders》](https://arxiv.org/abs/1908.08345)一文中所示。

## 从模型配置随机初始化`EncoderDecoderModel`。

可以从编码器和解码器配置随机初始化[`EncoderDecoderModel`]。
在下面的示例中，我们展示了如何使用默认的[`BertModel`]配置对编码器进行初始化，
并使用默认的[`BertForCausalLM`]配置对解码器进行初始化。

```python
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = EncoderDecoderModel(config=config)
```

## 从预训练的编码器和解码器初始化`EncoderDecoderModel`。

可以从预训练的编码器检查点和预训练的解码器检查点初始化[`EncoderDecoderModel`]。
请注意，任何预训练的自编码模型（例如BERT）都可以作为编码器，
而预训练的自编码模型（例如BERT），预训练的因果语言模型（例如GPT2），以及序列到序列模型的预训练解码器部分（例如BART的解码器）都可以作为解码器。
根据你选择的解码器架构不同，交叉注意层的初始化方式可能是随机的。
从预训练的编码器和解码器检查点初始化[`EncoderDecoderModel`]需要对下游任务进行微调，
这在[《*Warm-starting-encoder-decoder blog post*》](https://huggingface.co/blog/warm-starting-encoder-decoder)中已经证明过。
为此，[`EncoderDecoderModel`]类提供了[`EncoderDecoderModel.from_encoder_decoder_pretrained`]方法。

```python
>>> from transformers import EncoderDecoderModel, BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
```

## 加载现有的`EncoderDecoderModel`检查点并进行推理。

要加载[`EncoderDecoderModel`]类的微调检查点，[`EncoderDecoderModel`]提供了与Transformers中的任何其他模型架构相同的`from_pretrained(...)`方法。

要执行推理，可以使用[`generate`]方法，该方法允许自回归地生成文本。此方法支持多种解码方式，例如贪婪解码、束搜索和多项式采样。

```python
>>> from transformers import AutoTokenizer, EncoderDecoderModel

>>> # 加载微调的seq2seq模型和对应的分词器
>>> model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

>>> # 对一段长文本执行推理
>>> ARTICLE_TO_SUMMARIZE = (
...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
... )
>>> input_ids = tokenizer(ARTICLE_TO_SUMMARIZE, return_tensors="pt").input_ids

>>> # 自回归生成摘要（默认使用贪婪解码）
>>> generated_ids = model.generate(input_ids)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
nearly 800 thousand customers were affected by the shutoffs. the aim is to reduce the risk of wildfires. nearly 800, 000 customers were expected to be affected by high winds amid dry conditions. pg & e said it scheduled the blackouts to last through at least midday tomorrow.
```

## 将PyTorch检查点加载到`TFEncoderDecoderModel`中。

[`TFEncoderDecoderModel.from_pretrained`]目前不支持从pytorch检查点初始化模型。
将`from_pt=True`传递给此方法将抛出异常。如果只有特定编码器-解码器模型的pytorch检查点，可以使用以下解决办法：

```python
>>> # 从pytorch检查点加载的解决办法
>>> from transformers import EncoderDecoderModel, TFEncoderDecoderModel

>>> _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> # 这仅用于复制此特定模型的某些特定属性。
>>> model.config = _model.config
```

## 训练

创建了模型之后，可以进行与BART、T5或任何其他编码器解码器模型类似的微调。
如你所见，为了计算损失，模型只需要2个输入：`input_ids`（已编码输入序列的`input_ids`）和`labels`（已编码目标序列的`input_ids`）。

```python
>>> from transformers import BertTokenizer, EncoderDecoderModel

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> input_ids = tokenizer(
...     "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
...     return_tensors="pt",
... ).input_ids

>>> labels = tokenizer(
...     "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
...     return_tensors="pt",
... ).input_ids

>>> # forward函数会自动创建正确的decoder_input_ids
>>> loss = model(input_ids=input_ids, labels=labels).loss
```

有关训练的详细信息，请参见[colab](https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=ZwQIEhKOrJpl)。

此模型由[thomwolf](https://github.com/thomwolf)贡献。此模型的TensorFlow和Flax版本由[ydshieh](https://github.com/ydshieh)贡献。

## EncoderDecoderConfig

[[autodoc]] EncoderDecoderConfig

## EncoderDecoderModel

[[autodoc]] EncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained

## TFEncoderDecoderModel

[[autodoc]] TFEncoderDecoderModel
    - call
    - from_encoder_decoder_pretrained

## FlaxEncoderDecoderModel

[[autodoc]] FlaxEncoderDecoderModel
    - __call__
    - from_encoder_decoder_pretrained