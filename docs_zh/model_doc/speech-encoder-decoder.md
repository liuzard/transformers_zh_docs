<!--
版权所有©2021 HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），你不得使用此文件，除非符合许可证的规定。
你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

请注意，虽然此文件是使用Markdown格式的，但包含我们的文档生成器的特定语法（类似于MDX），可能无法在你的Markdown查看器中正确显示。
-->

# 语音编码解码模型

[`SpeechEncoderDecoderModel`]可以用于初始化语音到文本模型，其中编码器可以是任何预训练的语音自编码模型（例如[Wav2Vec2](wav2vec2)、[Hubert](hubert)），而解码器可以是任何预训练的自回归模型。

初始化语音序列到文本序列模型时，使用预训练的语音自编码器和解码器检查点可以提高其性能，例如在[大规模自监督学习用于语音翻译](https://arxiv.org/abs/2104.06678)一文中，作者Changhan Wang、Anne Wu、Juan Pino、Alexei Baevski、Michael Auli、Alexis Conneau展示了预训练检查点在语音识别和语音翻译中的效果。

使用[`SpeechEncoderDecoderModel`]进行推理的示例可以在[Speech2Text2](speech_to_text_2)中看到。

## 从模型配置随机初始化`SpeechEncoderDecoderModel`。

[`SpeechEncoderDecoderModel`]可以从编码器和解码器的配置随机初始化。在以下示例中，我们演示如何使用默认的[`Wav2Vec2Model`]编码器配置和默认的[`BertForCausalLM`]解码器配置进行初始化。

```python
>>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

>>> config_encoder = Wav2Vec2Config()
>>> config_decoder = BertConfig()

>>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = SpeechEncoderDecoderModel(config=config)
```

## 从预训练的编码器和预训练的解码器初始化`SpeechEncoderDecoderModel`。

[`SpeechEncoderDecoderModel`]可以从预训练的编码器和预训练的解码器检查点进行初始化。请注意，任何预训练的基于Transformer的语音模型，例如[Wav2Vec2](wav2vec2)、[Hubert](hubert)都可以作为编码器，而预训练的自编码模型（例如BERT）以及预训练的序列到序列模型（例如BART的解码器）也可以作为解码器。根据你选择的解码器架构不同，交叉注意力层可能会被随机初始化。[`SpeechEncoderDecoderModel`]的初始化需要在下游任务上进行微调，就像[解码器的温启动博文](https://huggingface.co/blog/warm-starting-encoder-decoder)中所示。为此，[`SpeechEncoderDecoderModel`]类提供了一个[`SpeechEncoderDecoderModel.from_encoder_decoder_pretrained`]方法。

```python
>>> from transformers import SpeechEncoderDecoderModel

>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "facebook/hubert-large-ll60k", "bert-base-uncased"
... )
```

## 加载现有的`SpeechEncoderDecoderModel`检查点并进行推理。

要加载`SpeechEncoderDecoderModel`类的微调检查点，[`SpeechEncoderDecoderModel`]提供了与Transformers中的其他模型架构一样的`from_pretrained(...)`方法。

要执行推理，可以使用[`generate`]方法，该方法允许自回归生成文本。该方法支持各种解码方式，例如贪心解码、波束搜索和多项式抽样。

```python
>>> from transformers import Wav2Vec2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> # 加载微调的语音翻译模型和相应的处理器
>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
>>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

>>> # 对一段英文语音进行推理（翻译为德文）
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> # 自回归生成转录文本（默认使用贪心解码）
>>> generated_ids = model.generate(input_values)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
Mr. Quilter ist der Apostel der Mittelschicht und wir freuen uns, sein Evangelium willkommen heißen zu können.
```

## 训练

模型创建后，可以像BART、T5或任何其他编码器-解码器模型一样对其进行微调，使用一组（语音，文本）对的数据集。
正如你所看到的，该模型只需两个输入即可计算损失：`input_values`（即语音输入）和`labels`（即目标序列的`input_ids`）。

```python
>>> from transformers import AutoTokenizer, AutoFeatureExtractor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset

>>> encoder_id = "facebook/wav2vec2-base-960h"  # 声学模型的编码器
>>> decoder_id = "bert-base-uncased"  # 文本解码器

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
>>> tokenizer = AutoTokenizer.from_pretrained(decoder_id)
>>> # 将预训练的编码器和预训练的解码器结合起来形成一个Seq2Seq模型
>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id)

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> # 加载音频输入并进行预处理（将均值/标准差标准化为0/1）
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> # 加载相应的转录文本并进行分词以生成标签
>>> labels = tokenizer(ds[0]["text"], return_tensors="pt").input_ids

>>> # 前向函数会自动创建正确的decoder_input_ids
>>> loss = model(input_values=input_values, labels=labels).loss
>>> loss.backward()
```

## SpeechEncoderDecoderConfig

[[autodoc]] SpeechEncoderDecoderConfig

## SpeechEncoderDecoderModel

[[autodoc]] SpeechEncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained

## FlaxSpeechEncoderDecoderModel

[[autodoc]] FlaxSpeechEncoderDecoderModel
    - __call__
    - from_encoder_decoder_pretrained