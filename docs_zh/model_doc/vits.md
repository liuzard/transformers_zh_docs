<!--版权 2023年The HuggingFace团队，保留所有权利。

根据Apache许可证第2.0版 (the "License")授权；您除遵守License之外不得使用此文件。
您可以在以下位置获取License的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面约定，按“目前状态”（AS IS）分发的软件不包含任何担保或条件，无论明示还是暗示。有关许可下的特定语言的权限和限制，请参阅许可。

-->

# VITS

## 概述

VITS模型是由Jaehyeon Kim, Jungil Kong, Juhee Son在[Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) 提出的。

VITS（**V**ariational **I**nference with adversarial learning for end-to-end **T**ext-to-**S**peech）是一种端到端的语音合成模型，它根据输入的文本序列来预测语音波形。它是一个由后验编码器、解码器和条件先验构成的条件变分自动编码器（VAE）。

基于变换器的模块会预测一组基于声谱图的声学特征，其中包括文本编码器和多个耦合层。声谱图会使用堆叠的转置卷积层进行解码，与HiFi-GAN声码器的风格类似。考虑到TTS问题的一对多特性，即同一文本输入可能会以多种方式发音，该模型还包括一个随机时长预测器，使模型能够根据相同的输入文本合成具有不同节奏的语音。

模型通过变分下界和对抗训练导出的损失组合进行端到端训练。为了提高模型的表现能力，应用了归一化流到条件先验分布。在推断过程中，基于时长预测模块对文本编码进行上采样，然后使用流模块和HiFi-GAN解码器的级联将其映射到波形。由于时长预测器具有随机性质，模型是非确定性的，因此需要固定的种子来生成相同的语音波形。

论文中的摘要如下所示：

*最近已经提出了几种实现单阶段训练和并行抽样的端到端TTS模型，但它们的采样质量与两阶段TTS系统不匹配。在这项工作中，我们提出了一种并行端到端TTS方法，它生成比当前两阶段模型更自然的声音。我们的方法采用了带有归一化流和对抗性训练过程的变分推理，这提高了生成建模的表达能力。我们还提出了一种随机时长预测器，以从输入文本中合成具有不同音高和节奏的语音。对于潜在变量的不确定性建模和随机时长预测器，我们的方法表达了自然的一对多关系，即文本输入可以以多种方式以不同的音高和节奏发音。对LJ Speech进行的主观人类评估（平均意见分数，或MOS）表明，我们的方法优于最佳公开可用的TTS系统，并获得与真实语音相当的MOS评分。*

该模型还可以与[MMS（Massively Multilingual Speech）](https://arxiv.org/abs/2305.13516)的TTS检查点一起使用，因为这些检查点使用相同的架构和略微修改的分词器。

该模型由[Matthijs](https://huggingface.co/Matthijs)和[sanchit-gandhi](https://huggingface.co/sanchit-gandhi)贡献。原始代码可以在[这里](https://github.com/jaywalnut310/vits)找到。

## 模型使用

VITS和MMS-TTS检查点可以使用相同的API。由于基于流的模型是非确定性的，建议设置一个种子以确保输出的可重复性。对于罗马字母表的语言，如英语或法语，可以直接使用分词器对文本进行预处理。以下代码示例使用MMS-TTS英文检查点进行前向传递：

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

set_seed(555)  # 设置种子以保证确定性

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]
```

生成的波形可以保存为`.wav`文件:

```python
import scipy

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=waveform)
```

或在Jupyter Notebook / Google Colab中显示：

```python
from IPython.display import Audio

Audio(waveform, rate=model.config.sampling_rate)
```

对于某些具有非罗马字母表的语言，如阿拉伯语、普通话或印地语，需要使用[`uroman`](https://github.com/isi-nlp/uroman) Perl包对文本进行预处理。

您可以通过检查预训练`tokenizer`的`is_uroman`属性，来查看您的语言是否需要`uroman`包:

```python
from transformers import VitsTokenizer

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
print(tokenizer.is_uroman)
```

如果需要，您应该在将文本输入传递给`VitsTokenizer`之前，将uroman包应用于文本输入。因为目前的分词器不支持执行预处理本身。

首先将uroman存储库克隆到本地计算机，并将bash变量`UROMAN`设置为本地路径:

```bash
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```

然后，您可以使用以下代码片段使用uroman包进行文本输入的预处理。您可以依赖于使用bash变量`UROMAN`指向uroman存储库，或者将uroman目录作为参数传递给`uromaize`函数：

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import os
import subprocess

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
model = VitsModel.from_pretrained("facebook/mms-tts-kor")

def uromanize(input_string, uroman_path):
    """使用`uroman` perl软件包将非罗马字符串转换为罗马字符串。"""
    script_path = os.path.join(uroman_path, "bin", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 执行perl命令
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"错误 {process.returncode}: {stderr.decode()}")

    # 将输出作为字符串返回，并跳过末尾的换行符
    return stdout.decode()[:-1]

text = "이봐 무슨 일이야"
uromaized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

inputs = tokenizer(text=uromaized_text, return_tensors="pt")

set_seed(555)  # 设置种子以保证确定性
with torch.no_grad():
   outputs = model(inputs["input_ids"])

waveform = outputs.waveform[0]
```

## VitsConfig

[[autodoc]] VitsConfig

## VitsTokenizer

[[autodoc]] VitsTokenizer
    - __call__
    - save_vocabulary

## VitsModel

[[autodoc]] VitsModel
    - forward