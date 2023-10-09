<!--版权 2023 The HuggingFace Team。版权所有。

根据Apache许可证第2.0版许可（“许可证”）; 除非符合许可证要求或书面同意，否则你不得使用此文件。
你可以获取许可证的副本位于

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则以“按原样”方式分发的软件分发在没有任何种类的条件或保证, 无论是明示还是暗示。有关许可下限制和禁止的特定语言，请参阅许可证。--->

# Bark

## 概览

Bark是一种基于transformer的文本到语音模型，由Suno AI在[suno-ai/bark](https://github.com/suno-ai/bark) 中提出。

Bark由4个主要模型组成：
- [`BarkSemanticModel`]（也称为“文本”模型）：一种因果自回归transformer模型，它以分词的文本作为输入，预测捕捉文本意义的语义文本标记。
- [`BarkCoarseModel`]（也称为“粗粒度声学”模型）：一种因果自回归transformer，它以[`BarkSemanticModel`]模型的结果作为输入。旨在预测EnCodec所需要的前两个音频码簿。
- [`BarkFineModel`]（“精细声学”模型）：这次是非因果自编码器transformer，它根据先前码簿嵌入的总和迭代地预测最后的码簿。
- 在预测了来自[`EncodecModel`]的所有码簿通道之后，Bark使用它来解码输出音频数组。

应该注意的是，前三个模块中的每一个都可以支持有条件的说话者嵌入，以便根据特定的预定义语音来对输出音频进行条件处理。

### 优化Bark

可以通过几行额外的代码对Bark进行优化，从而**显著减少其内存占用**并**加速推理**。

#### 使用半精度

只需将模型加载为半精度，即可将推理加速并减少内存占用约50%。

```python
from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
```

#### 使用 🤗 Better Transformer

Better Transformer是🤗 Optimum的一种优化功能，可以在底层进行内核融合。你可以获得20%至30%的速度提升，而性能不会下降。将模型导出到🤗 Better Transformer只需一行代码：

```python
model = model.to_bettertransformer()
```

请注意，在使用此功能之前，必须安装🤗 Optimum。[点击这里了解如何安装。](https://huggingface.co/docs/optimum/installation)

#### 使用CPU卸载

如上所述，Bark由4个子模型组成，在生成音频时按顺序调用这些子模型。换句话说，当一个子模型正在使用时，其他子模型处于空闲状态。

如果你使用的是CUDA设备，可以通过将空闲子模型卸载到CPU来实现内存占用减少80%的简单解决方案。这个操作称为CPU卸载，你只需使用一行代码即可使用它。

```python
model.enable_cpu_offload()
```

请注意，在使用此功能之前，必须安装🤗 Accelerate。[点击这里了解如何安装。](https://huggingface.co/docs/accelerate/basic_tutorials/install)

#### 结合优化技术

你可以结合使用优化技术，同时使用CPU卸载、半精度和🤗 Better Transformer。

```python
from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载半精度
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)

# 转换为bettertransformer
model = BetterTransformer.transform(model, keep_original_model=False)

# 启用CPU卸载
model.enable_cpu_offload()
```

在此处了解更多推理优化技术[here](https://huggingface.co/docs/transformers/perf_infer_gpu_one)。

### 提示

Suno提供了一系列多种语言的预设语音库[here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)。
这些预设语音库也上传到了[hub](https://huggingface.co/suno/bark-small/tree/main/speaker_embeddings) 或者 [这里](https://huggingface.co/suno/bark/tree/main/speaker_embeddings)。

```python
>>> from transformers import AutoProcessor, BarkModel

>>> processor = AutoProcessor.from_pretrained("suno/bark")
>>> model = BarkModel.from_pretrained("suno/bark")

>>> voice_preset = "v2/en_speaker_6"

>>> inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

Bark可以生成高度逼真的、**多语言**的语音以及其他音频 - 包括音乐、背景噪音和简单的音效。

```python
>>> # 多语言语音 - 简体中文
>>> inputs = processor("惊人的！我会说中文")

>>> # 多语言语音 - 法语 - 让我们也使用一个声音预设
>>> inputs = processor("Incroyable! Je peux générer du son.", voice_preset="fr_speaker_5")

>>> # Bark还可以生成音乐。你可以在歌词周围添加音乐音符来帮助它。
>>> inputs = processor("♪ Hello, my dog is cute ♪")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

该模型还可以生成**非语言交流**，如笑声、叹息和哭声。

```python
>>> # 在输入文本中添加非语言线索
>>> inputs = processor("Hello uh ... [clears throat], my dog is cute [laughter]")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

要保存音频，只需使用模型配置中的采样率和一些scipy工具即可：

```python
>>> from scipy.io.wavfile import write as write_wav

>>> # 将音频保存到磁盘，但是首先从模型配置中取采样率
>>> sample_rate = model.generation_config.sample_rate
>>> write_wav("bark_generation.wav", sample_rate, audio_array)
```

此模型由[Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe) 和 [Sanchit Gandhi (sanchit-gandhi)](https://github.com/sanchit-gandhi) 贡献。
原始代码可以在[这里](https://github.com/suno-ai/bark)找到。

## BarkConfig

[[autodoc]] BarkConfig
 - all

## BarkProcessor

[[autodoc]] BarkProcessor
 - all
 - __call__

## BarkModel

[[autodoc]] BarkModel
 - generate
 - enable_cpu_offload

## BarkSemanticModel

[[autodoc]] BarkSemanticModel
 - forward

## BarkCoarseModel

[[autodoc]] BarkCoarseModel
 - forward

## BarkFineModel

[[autodoc]] BarkFineModel
 - forward

## BarkCausalModel

[[autodoc]] BarkCausalModel
 - forward

## BarkCoarseConfig

[[autodoc]] BarkCoarseConfig
 - all

## BarkFineConfig

[[autodoc]] BarkFineConfig
 - all

## BarkSemanticConfig

[[autodoc]] BarkSemanticConfig
 - all
