版权所有2023年The HuggingFace团队。Apache许可证第2.0版（“许可证”）许可，除非符合许可证的要求,否则你不能使用此文件。你可以获取许可证的副本。http://www.apache.org/licenses/LICENSE-2.0 unless You may obtain a copy of the License at。许可证要求适用法律或以书面形式同意，经授权分发软件仅以“只是”为基础，不附带任何保证或条件。请参阅许可以了解特定语言下许可证的限制和规定。⚠️请注意，此文件是Markdown格式，但包含我们doc-builder(类似于MDX)的特定语法，可能在你的Markdown视图器中无法正确渲染的内容。

# MMS

## 概述

[MMS模型](https://arxiv.org/abs/2305.13516)(将语音技术扩展到1000多种语言)由Vineel Pratap、Andros Tjandra、Bowen Shi、Paden Tomasello、Arun Babu、Sayani Kundu、Ali Elkahky、Zhaoheng Ni、Apoorv Vyas、Maryam Fazel-Zarandi、Alexei Baevski、Yossiad、Xiaohui Zhang、Wei-Ning Hsu、Alexis Conneau、Michael Auli提出。

来自该论文的摘要如下：

扩大语音技术的语言覆盖面，可以提高更多的人获取信息。然而，当前的语音技术仅限于约100种语言，这只是世界上约7000种语言的一小部分。大规模多语言语音(MMS)项目将支持的语言数量增加了10-40倍，具体取决于任务。主要成分是基于公开可用宗教文本的阅读的新数据集，以及有效利用自监督学习。我们构建了预训练的wav2vec 2.0模型，覆盖了1406种语言，一种支持1107种语言的单一多语言自动语音识别模型，为相同数量的语言建立了语音合成模型，以及一种支持4017种语言的语言识别模型。实验结果显示，我们的多语音识别模型在"FLEURS基准测试"的54种语言的"Whisper"的单词错误率下降50%，而练习时，只是使用了一小部分标记数据。

下面是MMS项目中开源的不同模型。这些模型和代码最初在[这里](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)发布。我们将它们添加到了`transformers`框架中，使它们更容易使用。

### 自动语音识别(ASR)

ASR模型检查点可以在这里找到：[mms-1b-fl102](https://huggingface.co/facebook/mms-1b-fl102), [mms-1b-l1107](https://huggingface.co/facebook/mms-1b-l1107), [mms-1b-all](https://huggingface.co/facebook/mms-1b-all)。为了获得最佳准确性，使用`mms-1b-all`模型。

提示：

- 所有ASR模型都接受与语音信号的原始波形对应的浮点数数组。原始波形应使用[`Wav2Vec2FeatureExtractor`]进行预处理。
- 这些模型使用连接时间分类(CTC)进行训练，因此模型输出必须使用[`Wav2Vec2CTCTokenizer`]进行解码。
- 你可以通过[`~Wav2Vec2PreTrainedModel.load_adapter`]为不同的语言加载不同的语言适配器的权重。语言适配器只包含大约200万个参数，因此在需要时可以高效地动态加载。

#### 加载

默认情况下，MMS仅加载英语的适配器权重。如果你想加载其他语言的适配器权重，请确保同时指定`target_lang=<你选择的目标语言>`和`ignore_mismatched_sizes=True`。要允许根据指定语言的词汇表调整语言模型头的大小，必须传递`ignore_mismatched_sizes=True`关键字。同样，处理器应该使用相同的目标语言加载。

```py
from transformers import Wav2Vec2ForCTC, AutoProcessor

model_id = "facebook/mms-1b-all"
target_lang = "fra"

processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)
model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=target_lang, ignore_mismatched_sizes=True)
```

<Tip>

你可以安全地忽略如下警告：

```text
Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at facebook/mms-1b-all and are newly initialized because the shapes did not match:
- lm_head.bias: found shape torch.Size([154]) in the checkpoint and torch.Size([314]) in the model instantiated
- lm_head.weight: found shape torch.Size([154, 1280]) in the checkpoint and torch.Size([314, 1280]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

</Tip>

如果你想使用ASR流水线，可以像这样加载所选的目标语言：

```py
from transformers import pipeline

model_id = "facebook/mms-1b-all"
target_lang = "fra"

pipe = pipeline(model=model_id, model_kwargs={"target_lang": "fra", "ignore_mismatched_sizes": True})
```

#### 推理

接下来，让我们看看如何在推理中运行MMS并在调用[`~PretrainedModel.from_pretrained`]之后更改适配器层。首先，我们使用[Datasets](https://github.com/huggingface/datasets)加载不同语言的音频数据。

```py
from datasets import load_dataset, Audio

# 英语
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

# 法语
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "fr", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
fr_sample = next(iter(stream_data))["audio"]["array"]
```

接下来，我们加载模型和处理器。

```py
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
```

现在我们处理音频数据，将处理后的音频数据传递给模型并转录模型输出，就像我们通常处理[`Wav2Vec2ForCTC`]一样。

```py
inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# 'joe keton disapproved of films and buster also had reservations about the media'
```

现在，我们可以将相同的模型保存在内存中，只需调用模型的方便的[`~Wav2Vec2ForCTC.load_adapter`]功能，以及令牌器的[`~Wav2Vec2CTCTokenizer.set_target_lang`]功能 来更改语言适配器。我们将目标语言作为输入传递给它，对于法语是`"fra"`。

```py
processor.tokenizer.set_target_lang("fra")
model.load_adapter("fra")

inputs = processor(fr_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# "ce dernier est volé tout au long de l'histoire romaine"
```

同样，可以为所有其他支持的语言切换语言。请查看：

```py
processor.tokenizer.vocab.keys()
```

以查看所有支持的语言。

要进一步改善ASR模型的性能，可以使用语言模型解码。有关详细信息，请查看[此处](https://huggingface.co/facebook/mms-1b-all)的文档。

### 语音合成（TTS）

MMS-TTS使用与VITS相同的模型架构，该模型架构在v4.33中添加到了🤗 Transformers 。MMS为项目中的1100多种语言训练了单独的模型检查点。所有可用检查点都可以在Hugging Face Hub上找到：[facebook/mms-tts](https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts)，推理文档在[VITS](https://huggingface.co/docs/transformers/main/en/model_doc/vits)下。

#### 推理

要使用MMS模型，首先请确保将Transformers库升级到最新版本：

```bash
pip install --upgrade transformers accelerate
```

由于VITS中的基于流的模型是非确定性的，为了确保输出的可重复性，最好设置一个种子。

- 对于具有罗马字母的语言（如英语或法语），可以直接使用令牌器对文本进行预处理。以下代码示例运行了使用MMS-TTS英语检查点的正向传递：

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

set_seed(555)  # 让其具有确定性

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]
```

生成的波形可以保存为`.wav`文件：

```python
import scipy

scipy.io.wavfile.write("synthesized_speech.wav", rate=model.config.sampling_rate, data=waveform)
```

或在Jupyter Notebook / Google Colab中显示：

```python
from IPython.display import Audio

Audio(waveform, rate=model.config.sampling_rate)
```

对于某些具有非罗马字母方案（如阿拉伯语、汉语或印地语）的语言，需要[`uroman`](https://github.com/isi-nlp/uroman)Perl软件包来对文本输入进行预处理，将文字转换为罗马字母。

你可以通过检查预训练令牌器的`is_uroman`属性来确定你的语言是否需要`uroman`软件包：

```python
from transformers import VitsTokenizer

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
print(tokenizer.is_uroman)
```

如果需要，你应该在将文本输入传递给`VitsTokenizer`之前，先将uroman软件包应用于你的文本输入。因为目前令牌器不支持执行预处理本身。

要做到这一点，首先将uroman存储库克隆到本地计算机，并将bash变量`UROMAN`设置为本地路径：

```bash
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```

你可以使用以下代码段使用uroman软件包对文本输入进行预处理。你可以依赖使用bash变量`UROMAN`指向uroman存储库，也可以将uroman目录作为参数传递给`uroman`函数：

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import os
import subprocess

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
model = VitsModel.from_pretrained("facebook/mms-tts-kor")

def uromanize(input_string, uroman_path):
    """使用`uroman` Perl软件包将非罗马字母字符串转换为罗马字母。"""
    script_path = os.path.join(uroman_path, "bin", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 执行perl命令
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    # 作为字符串返回输出，并跳过末尾的换行符
    return stdout.decode()[:-1]

text = "이봐 무슨 일이야"
uromaized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

inputs = tokenizer(text=uromaized_text, return_tensors="pt")

set_seed(555)  # make deterministic
with torch.no_grad():
   outputs = model(inputs["input_ids"])

waveform = outputs.waveform[0]
```

**提示：**

* MMS-TTS检查点是在小写、无标点的文本上进行训练的。默认情况下，`VitsTokenizer` *归一化*输入，通过删除所有大小写和标点符号，以避免将未登录字符传递给模型。因此，模型不受大小写和标点符号的影响，因此在文本提示中应避免使用它们。你可以通过在调用令牌器时设置`noramlize=False`来禁用归一化，但这将导致非预期的行为，不建议这样做。
* 通过将属性`model.speaking_rate`设置为所选择的值，可以变化说话速度。同理，噪声的随机性由`model.noise_scale`控制。

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

# make deterministic
set_seed(555)  

# 调节语速和噪声的幅度
model.speaking_rate = 1.5
model.noise_scale = 0.8

with torch.no_grad():
   outputs = model(**inputs)
```


### 语言识别（LID）

根据它们能够识别的语言数量，提供了不同的LID模型-[126](https://huggingface.co/facebook/mms-lid-126), [256](https://huggingface.co/facebook/mms-lid-256), [512](https://huggingface.co/facebook/mms-lid-512), [1024](https://huggingface.co/facebook/mms-lid-1024), [2048](https://huggingface.co/facebook/mms-lid-2048), [4017](https://huggingface.co/facebook/mms-lid-4017)。

#### 推理
首先，我们安装transformers和其他一些库

```bash
pip install torch accelerate datasets[audio]
pip install --upgrade transformers
````

接下来，通过`datasets`加载一些音频样本。确保音频数据采样为16000 kHz。

```py
from datasets import load_dataset, Audio

# 英语
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

# 阿拉伯语
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "ar", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
ar_sample = next(iter(stream_data))["audio"]["array"]
```

接下来，我们加载模型和处理器

```py
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

model_id = "facebook/mms-lid-126"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
```

现在我们处理音频数据，将处理后的音频数据传递给模型进行语言识别，就像我们通常处理诸如[ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition)的Wav2Vec2音频分类模型一样。

```py
# 英语
inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

lang_id = torch.argmax(outputs, dim=-1)[0].item()
detected_lang = model.config.id2label[lang_id]
# 'eng'

# 阿拉伯语
inputs = processor(ar_sample, sampling_rate=16_000, return_tensors="pt")

使用`torch.no_grad()`来禁用梯度计算：
```python
outputs = model(**inputs).logits

lang_id = torch.argmax(outputs, dim=-1)[0].item()
detected_lang = model.config.id2label[lang_id]
# 'ara'
```

要查看检查点支持的所有语言，可以按如下方式打印出语言ID：
```python
processor.id2label.values()
```

### 预训练音频模型

预训练模型有两种不同的大小可用 - [300M](https://huggingface.co/facebook/mms-300m) ，[1Bil](https://huggingface.co/facebook/mms-1b)。模型的架构基于Wav2Vec2模型，因此可以参考[Wav2Vec2的文档页面](wav2vec2)了解如何使用这些模型进行各种下游任务的微调的更多细节。