<!--版权所有 2023 年 HuggingFace 团队。保留所有权利。

根据 Apache License, Version 2.0 许可证（以下简称“许可证”），除非遵守许可证，否则你不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，不附带任何明示或暗示的担保或条件。查看许可证以获取许可证下的特定语言和限制事项。

⚠️请注意，此文件为 Markdown 格式，但包含我们的文档生成器的特定语法（类似于 MDX），在你的 Markdown 查看器中可能无法正确显示。

-->

# EnCodec

## 概述

EnCodec 神经编解码模型在 《High Fidelity Neural Audio Compression》 中由 Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi 提出。

论文摘要如下：

*我们引入了一种基于神经网络的实时高保真音频编解码器。它采用一种流式编码器-解码器架构，带有量化的潜在空间，并以端到端方式进行训练。我们通过使用一个多尺度频谱敌对者来简化和加快训练，以有效减少伪像并生成高质量样本。我们引入了一种新颖的损失平衡机制来稳定训练：现在，一个损失的权重定义了它应该表示的总梯度的比例，从而将这个超参数的选择与损失的典型规模解耦。最后，我们研究了如何使用轻量级 Transformer 模型进一步压缩所获得的表示，同时保持超过 40% 的压缩比率，而速度快于实时。我们详细描述了所提出模型的关键设计选择，包括：训练目标、结构变化和各种感知损失函数的研究。我们提供了广泛的主观评估（MUSHRA 测试），并对一系列的带宽和音频领域进行了消融研究，包括语音、有噪音混响的语音和音乐。我们的方法在所有评估设置中均优于基线方法，考虑到 24 kHz 单声道和 48 kHz 立体声音频。*

该模型由 [Matthijs](https://huggingface.co/Matthijs), [Patrick Von Platen](https://huggingface.co/patrickvonplaten) 和 [Arthur Zucker](https://huggingface.co/ArthurZ) 提供。原始代码可以在 [这里](https://github.com/facebookresearch/encodec) 找到。以下是使用该模型对音频进行编码和解码的示例：

```python 
>>> from datasets import load_dataset, Audio
>>> from transformers import EncodecModel, AutoProcessor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> model = EncodecModel.from_pretrained("facebook/encodec_24khz")
>>> processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
>>> audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
>>> # 或者使用前向传递进行等效操作
>>> audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```


## EncodecConfig

[[autodoc]] EncodecConfig

## EncodecFeatureExtractor

[[autodoc]] EncodecFeatureExtractor
    - __call__

## EncodecModel

[[autodoc]] EncodecModel
    - decode
    - encode
    - forward
-->