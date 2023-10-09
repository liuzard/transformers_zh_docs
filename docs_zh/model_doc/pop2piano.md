<!--版权所有2023 The HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”），你将无法使用此文件，除非符合许可证的规定。你可以在

http://www.apache.org/license/LICENSE-2.0

从该许可证获取许可证的副本。

除非适用法律要求或书面同意，软件根据

“按原样”分发，不附带任何明示或暗示的担保或条件。请查看有关

特定语言限制和许可证下限制的许可证。-->

# Pop2Piano

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/spaces/sweetcocoa/pop2piano">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概览

Pop2Piano模型由Jongho Choi和Kyogu Lee在[Pop2Piano：基于流行音频的钢琴翻唱曲生成](https://arxiv.org/abs/2211.00895)中提出。

流行音乐的钢琴翻唱很受欢迎，但从音乐中生成它们并不是一项简单的任务。它需要对弹奏钢琴有很高的专业知识，以及了解歌曲的不同特点和旋律。使用Pop2Piano，你可以直接从歌曲的音频波形生成翻唱曲。它是首个不需要旋律和和弦提取模块直接从流行音频生成钢琴翻唱曲的模型。

Pop2Piano是基于[T5](https://arxiv.org/pdf/1910.10683.pdf)的编码器-解码器Transformer模型。输入音频被转换为其波形，并传递给编码器，该编码器将其转换为潜在表示。解码器使用这些潜在表示以自回归的方式生成令牌id。每个令牌id对应于四种不同的令牌类型：时间，速度，音符和“特殊”。然后将令牌id解码为相应的MIDI文件。

论文中的摘要如下：

*许多人喜欢流行音乐的钢琴翻唱。但是，自动生成流行音乐的钢琴翻唱仍然是一个不太研究的任务。这部分是因为缺乏同步的{流行音乐，钢琴翻唱}数据对，这使得应用最新的数据密集型基于深度学习的方法变得困难。为了利用数据驱动方法的优势，我们使用自动化流程创建了大量配对和同步的{流行音乐，钢琴翻唱}数据。在本文中，我们提出了Pop2Piano，这是一种基于Transformer网络的方法，它在给定流行音乐波形的情况下生成钢琴翻唱。据我们所知，这是首个直接从流行音频生成钢琴翻唱而不使用旋律和和弦提取模块的模型。我们展示了通过我们的数据集训练的Pop2Piano能够生成可信的钢琴翻唱。*

提示：

1. 要使用Pop2Piano，你需要安装🤗 Transformers库以及以下第三方模块：
```python
pip install pretty-midi==0.2.9 essentia==2.1b6.dev1034 librosa scipy
```
请注意，安装后可能需要重新启动运行时。
2. Pop2Piano是基于T5的编码器-解码器模型。
3. Pop2Piano可用于为给定的音频序列生成MIDI音频文件。
4. 在`Pop2PianoForConditionalGeneration.generate()`中选择不同的作曲家会导致不同的结果。
5. 在加载音频文件时将采样率设置为44.1 kHz可以获得良好的性能。
6. 尽管Pop2Piano主要是在韩国流行音乐上训练的，但在其他西方流行音乐或嘻哈音乐上也表现得很好。

此模型由[Susnato Dhar](https://huggingface.co/susnato)贡献。
原始代码可以在[这里](https://github.com/sweetcocoa/pop2piano)找到。

## 示例

- 使用HuggingFace数据集的示例：

```python
>>> from datasets import load_dataset
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
>>> ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

>>> inputs = processor(
...     audio=ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"], return_tensors="pt"
... )
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

- 使用你自己的音频文件的示例：

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> audio, sr = librosa.load("<your_audio_file_here>", sr=44100)  # 随意更改sr为适当的值。
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

- 批处理多个音频文件的示例：

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> # 随意更改sr为适当的值。
>>> audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)  
>>> audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=[audio1, audio2], sampling_rate=[sr1, sr2], return_attention_mask=True, return_tensors="pt")
>>> # 由于现在生成批处理（2个音频），我们必须传递attention_mask
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> # 由于我们现在有2个生成的MIDI文件
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```


- 批处理多个音频文件的示例（使用`Pop2PianoFeatureExtractor`和`Pop2PianoTokenizer`）：

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoFeatureExtractor, Pop2PianoTokenizer

>>> # 随意更改sr为适当的值。
>>> audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)  
>>> audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("sweetcocoa/pop2piano")
>>> tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = feature_extractor(
...     audio=[audio1, audio2], 
...     sampling_rate=[sr1, sr2], 
...     return_attention_mask=True, 
...     return_tensors="pt",
... )
>>> # 由于现在生成批处理（2个音频），我们必须传递attention_mask
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = tokenizer.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> # 由于我们现在有2个生成的MIDI文件
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```


## Pop2PianoConfig

[[autodoc]] Pop2PianoConfig

## Pop2PianoFeatureExtractor

[[autodoc]] Pop2PianoFeatureExtractor
    - __call__

## Pop2PianoForConditionalGeneration

[[autodoc]] Pop2PianoForConditionalGeneration
    - forward
    - generate

## Pop2PianoTokenizer

[[autodoc]] Pop2PianoTokenizer
    - __call__

## Pop2PianoProcessor

[[autodoc]] Pop2PianoProcessor
    - __call__