<!--版权 2021年HuggingFace团队。版权所有。

根据Apache许可证第2.0版（"许可证"），除非符合许可证规定，
否则不得使用此文件。您可以获取许可证的副本，网址为

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，否则按"原样"分发软件，无论是明示还是暗示。
请参阅许可证了解许可证下特定语言的权限和限制。

⚠️ 请注意，此文件采用Markdown格式，但包含我们的文档生成器（类似于MDX）的特定语法，
可能无法在您的Markdown查看器中正确呈现。

-->

# 视觉编码器解码器模型

## 概述

[`VisionEncoderDecoderModel`]可以用于使用任何预训练的基于Transformer的视觉模型作为编码器（如[Vit]，[BEiT]，[DeiT]，[Swin]）
和任何预训练的语言模型作为解码器（如[RoBERTa]，[GPT2]，[BERT]，[DistilBERT]）初始化图像到文本模型。

在[TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282)中，
Minghao Li、Tengchao Lv、Lei Cui、Yijuan Lu、Dinei Florencio、Cha Zhang、Zhoujun Li和Furu Wei展示了使用预训练检查点初始化图像到文本序列模型的有效性。

在训练/微调[`VisionEncoderDecoderModel`]之后，它可以像其他模型一样保存/加载（有关更多信息，请参见下面的示例）。

一个示例应用是图像字幕，在这种情况下，编码器用于编码图像，之后一个自回归语言模型生成字幕。另一个示例是光学字符识别。参考[TrOCR](trocr)，
它是[`VisionEncoderDecoderModel`]的一个实例。

## 从模型配置随机初始化`VisionEncoderDecoderModel`。

可以使用编码器和解码器配置随机初始化[`VisionEncoderDecoderModel`]。在下面的例子中，我们展示了如何使用编码器的默认[`ViTModel`]配置
和解码器的默认[`BertForCausalLM`]配置进行这样做。

```python
>>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

>>> config_encoder = ViTConfig()
>>> config_decoder = BertConfig()

>>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = VisionEncoderDecoderModel(config=config)
```

## 从预训练的编码器和预训练的解码器初始化`VisionEncoderDecoderModel`。

[`VisionEncoderDecoderModel`]可以从预训练的编码器检查点和预训练的解码器检查点进行初始化。请注意，任何预训练的基于Transformer的视觉模型，
如[Swin]，都可以作为编码器，同时预训练的自编码模型，如BERT，预训练的因果语言模型，如GPT2，以及序列到序列模型的预训练解码器部分，
如BART的解码器，都可以作为解码器。根据您选择的解码器架构，交叉注意力层可能会随机初始化。
从预训练的编码器和解码器检查点初始化[`VisionEncoderDecoderModel`]要求对模型进行下游任务的微调，就像在[Warm-starting-encoder-decoder博客文章](https://huggingface.co/blog/warm-starting-encoder-decoder)中所展示的那样。
为此，`VisionEncoderDecoderModel`类提供了一个[`VisionEncoderDecoderModel.from_encoder_decoder_pretrained`]方法。

```python
>>> from transformers import VisionEncoderDecoderModel

>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "microsoft/swin-base-patch4-window7-224-in22k", "bert-base-uncased"
... )
```

## 加载现有的`VisionEncoderDecoderModel`检查点并进行推理。

要加载`VisionEncoderDecoderModel`类的微调检查点，[`VisionEncoderDecoderModel`]提供了与Transformers中的其他模型架构一样的`from_pretrained(...)`方法。

要执行推理，可以使用[`generate`]方法，该方法允许自动回归生成文本。该方法支持多种解码形式，如贪婪、束搜索和多项式采样。

```python
>>> import requests
>>> from PIL import Image

>>> from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

>>> # 加载微调后的图像字幕模型及其相应的标记器和图像处理器
>>> model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> # 在图像上执行推理
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> # 自动回归生成字幕（默认使用贪婪解码）
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
一个猫躺在毯子上，旁边有一个猫躺在床上
```

## 将PyTorch检查点加载到`TFVisionEncoderDecoderModel`中。

`TFVisionEncoderDecoderModel.from_pretrained`目前不支持从PyTorch检查点初始化模型。
将`from_pt=True`传递给此方法将引发异常。如果针对特定视觉编码器-解码器模型仅有PyTorch检查点，则可以使用以下解决方法：

```python
>>> from transformers import VisionEncoderDecoderModel, TFVisionEncoderDecoderModel

>>> _model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> # 此代码仅用于复制此特定模型的某些特定属性。
>>> model.config = _model.config
```

## 训练

一旦创建了模型，可以在BART、T5或任何其他编码器-解码器模型的数据集上进行微调。
可以看到，为了计算损失，模型只需要两个输入项：`pixel_values`（即图像）和`labels`（即目标序列的`input_ids`）。

```python
>>> from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
>>> from datasets import load_dataset

>>> image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "bert-base-uncased"
... )

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> labels = tokenizer(
...     "an image of two cats chilling on a couch",
...     return_tensors="pt",
... ).input_ids

>>> # 前向函数会自动创建正确的decoder_input_ids
>>> loss = model(pixel_values=pixel_values, labels=labels).loss
```

本模型的贡献者为[nielsr](https://github.com/nielsrogge)。该模型的TensorFlow版本和Flax版本的贡献者是[ydshieh](https://github.com/ydshieh)。

## VisionEncoderDecoderConfig

[[autodoc]] VisionEncoderDecoderConfig

## VisionEncoderDecoderModel

[[autodoc]] VisionEncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained

## TFVisionEncoderDecoderModel

[[autodoc]] TFVisionEncoderDecoderModel
    - call
    - from_encoder_decoder_pretrained

## FlaxVisionEncoderDecoderModel

[[autodoc]] FlaxVisionEncoderDecoderModel
    - __call__
    - from_encoder_decoder_pretrained