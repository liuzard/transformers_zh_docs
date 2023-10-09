<!--
版权所有2023年 HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”），除非符合许可证的规定，
否则你不能使用此文件。你可以在下面的网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证的规定在“AS IS”的基础上分发，
不提供任何形式的保证或条件，无论是明示的还是默示的。有关详细信息，请参见许可证中的规定。

⚠️请注意，这个文件是用Markdown编写的，但包含我们的doc-builder的特定语法（类似于MDX），
可能无法在你的Markdown查看器中正确渲染。

-->

# MusicGen

## 概述

MusicGen模型在文章[Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)中提出，
作者是Jade Copet、Felix Kreuk、Itai Gat、Tal Remez、David Kant、Gabriel Synnaeve、Yossi Adi和Alexandre Défossez。

MusicGen是一个单阶段的自回归Transformer模型，能够根据文本描述或音频提示生成高质量的音乐样本。文本描述经过一个冻结的文本编码器模型，
得到一个隐藏状态表示的序列。然后，MusicGen通过这些隐藏状态来预测离散的音频令牌或音频码，并使用音频压缩模型（如EnCodec）解码得到音频波形。

通过一种高效的标记交错模式，MusicGen不需要自监督的语义表示，因此不需要级联多个模型来预测一组码本（如层次化或上采样）。
相反，它能够在一次前向传递中生成所有码本。

论文摘要如下：

*我们解决了条件音乐生成任务。我们介绍了MusicGen，一个单阶段的语言模型(LM)，用于多个音乐离散压缩表示流，即令牌。与之前的工作不同，
MusicGen由一个单阶段的transformer LM和高效的标记交错模式组成，这消除了级联多个模型（如层次化或上采样）的需要。
遵循这种方法，我们展示了MusicGen如何生成高质量的样本，并且可以根据文本描述或旋律特征进行条件控制，从而更好地控制生成的输出。
我们进行了广泛的实证评估，包括自动化和人工研究，表明所提出的方法在标准的文本到音乐基准上优于评估的基线。通过消融研究，我们阐明了MusicGen的每个组成部分的重要性。*

该模型由[sanchit-gandhi](https://huggingface.co/sanchit-gandhi)贡献。原始代码可以在[此处](https://github.com/facebookresearch/audiocraft)找到。
预训练的检查点可以在[Hugging Face Hub](https://huggingface.co/models?sort=downloads&search=facebook%2Fmusicgen-)上找到。

## 生成

MusicGen与两种生成模式兼容：greedy（贪婪）和sampling（采样）。实践表明，与greedy相比，sampling可以得到更好的结果，
因此我们鼓励在可能的情况下使用sampling模式。sampling默认为启用状态，并且可以通过在调用[`MusicgenForConditionalGeneration.generate`]时设置`do_sample=True`
或覆盖模型的生成配置来显式指定sampling模式（见下文）。

由于音频波形的正弦位置嵌入，生成受到时间限制为30秒输入的限制。也就是说，MusicGen不能生成超过30秒的音频（1503个标记），
并且由音频提示传递的输入音频会对此限制产生影响，因此对于20秒的音频输入，MusicGen不能生成超过10秒的额外音频。

### 无条件生成

无条件（或“null”）生成的输入可以通过[`MusicgenForConditionalGeneration.get_unconditional_inputs`]方法获得：

```python
>>> from transformers import MusicgenForConditionalGeneration

>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
>>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)

>>> audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
```

音频输出是一个三维的Torch张量，形状为`(batch_size, num_channels, sequence_length)`。你可以在ipynb笔记本中播放生成的音频样本：

```python
from IPython.display import Audio

sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```

或者使用第三方库（如`scipy`）保存为`.wav`文件：

```python
>>> import scipy

>>> sampling_rate = model.config.audio_encoder.sampling_rate
>>> scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
```

### 文本条件生成

该模型可以根据文本提示来生成音频样本，通过使用[`MusicgenProcessor`]来预处理输入：

```python
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> inputs = processor(
...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

`guidance_scale`用于分类器自由引导（CFG），设置条件对数和无条件对数之间权重的权重，
其中条件对数是从文本提示预测的，无条件对数是从无条件或'null'提示预测的。更高的guidance_scale鼓励模型生成更与输入提示相关的样本，
通常以音频质量较差为代价。通过设置`guidance_scale > 1`来启用CFG。为了获得最佳结果，使用`guidance_scale=3`（默认值）。

### 音频条件生成

相同的[`MusicgenProcessor`]可以用于预处理用于音频延续的音频提示。在以下示例中，我们使用🤗 Datasets库加载音频文件，
可以通过以下命令进行pip安装：

```
pip install --upgrade pip
pip install datasets[audio]
```

```python
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
>>> sample = next(iter(dataset))["audio"]

>>> # 取音频样本的前一半
>>> sample["array"] = sample["array"][: len(sample["array"]) // 2]

>>> inputs = processor(
...     audio=sample["array"],
...     sampling_rate=sample["sampling_rate"],
...     text=["80s blues track with groovy saxophone"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

对于批量音频条件生成，可以通过使用[`MusicgenProcessor`]类将生成的`audio_values`进行后处理以去除填充：

```python
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
>>> sample = next(iter(dataset))["audio"]

>>> # 取音频样本的前四分之一
>>> sample_1 = sample["array"][: len(sample["array"]) // 4]

>>> # 取音频样本的前一半
>>> sample_2 = sample["array"][: len(sample["array"]) // 2]

>>> inputs = processor(
...     audio=[sample_1, sample_2],
...     sampling_rate=sample["sampling_rate"],
...     text=["80s blues track with groovy saxophone", "90s rock song with loud guitars and heavy drums"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

>>> # 后处理以将批量音频的填充去除
>>> audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)
```

### 生成配置

控制生成过程的默认参数（如sampling、guidance scale和生成的标记数量）可以在模型的生成配置中找到，并根据需要进行更新：

```python
>>> from transformers import MusicgenForConditionalGeneration

>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> # 检查默认的生成配置
>>> model.generation_config

>>> # 将guidance scale增加到4.0
>>> model.generation_config.guidance_scale = 4.0

>>> # 将最大长度减少到256个标记
>>> model.generation_config.max_length = 256
```

请注意，传递给generate方法的任何参数将**覆盖**生成配置中的相应参数设置，
因此在调用generate时将`do_sample=False`设置为参数会覆盖生成配置中的`model.generation_config.do_sample`设置。

## 模型结构

MusicGen模型可以分解为三个不同的阶段：
1. 文本编码器：将文本输入映射为隐藏状态表示的序列。预训练的MusicGen模型使用来自T5或Flan-T5的冻结文本编码器。
2. MusicGen解码器：一种语言模型（LM），根据编码器隐藏状态表示自回归地生成音频令牌（或码）。
3. 音频编码器/解码器：用于对音频提示进行编码以生成提示标记，并从解码器预测的音频令牌中恢复音频波形。

因此，MusicGen模型可以作为一个独立的解码器模型使用，对应于[`MusicgenForCausalLM`]类，
也可以作为一个包含文本编码器和音频编码器/解码器的复合模型使用，对应于[`MusicgenForConditionalGeneration`]类。
如果只需从预训练的检查点加载解码器，则可以通过首先指定正确的配置来加载解码器，或者通过复合模型的`.decoder`属性进行访问：

```python
>>> from transformers import AutoConfig, MusicgenForCausalLM, MusicgenForConditionalGeneration

>>> # 选项1：获取解码器配置并传递给`.from_pretrained`
>>> decoder_config = AutoConfig.from_pretrained("facebook/musicgen-small").decoder
>>> decoder = MusicgenForCausalLM.from_pretrained("facebook/musicgen-small", **decoder_config)

>>> # 选项2：加载整个复合模型，但只返回解码器
>>> decoder = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").decoder
```

由于文本编码器和音频编码器/解码器模型在训练过程中是冻结的，MusicGen解码器[`MusicgenForCausalLM`]可以
在编码器隐藏状态和音频码的数据集上独立训练。在推断过程中，训练过的解码器可以与冻结的文本编码器和音频编码器/解码器结合，
以恢复复合[`MusicgenForConditionalGeneration`]模型。

提示：
* MusicGen是在32kHz的Encodec检查点上训练的。请确保使用兼容的Encodec模型版本。
* sampling模式往往比greedy模式产生更好的结果-你可以通过在调用[`MusicgenForConditionalGeneration.generate`]时将变量`do_sample`设置为`True`来切换sampling模式。

## MusicgenDecoderConfig

[[autodoc]] MusicgenDecoderConfig

## MusicgenConfig

[[autodoc]] MusicgenConfig

## MusicgenProcessor

[[autodoc]] MusicgenProcessor

## MusicgenModel

[[autodoc]] MusicgenModel
    - forward

## MusicgenForCausalLM

[[autodoc]] MusicgenForCausalLM
    - forward

## MusicgenForConditionalGeneration

[[autodoc]] MusicgenForConditionalGeneration
    - forward
-->