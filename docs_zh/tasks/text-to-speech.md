<!--版权所有 © 2023 The HuggingFace Team

根据Apache License，Version 2.0 (“许可证”)提供；除非符合许可证规定，否则不得使用此文件。您可以从以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律法规要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件。详见许可证以获取特定语言管理的权限和限制。

⚠️请注意，该文件采用Markdown格式，但包含我们文档构建器（类似于MDX）的特定语法，可能无法在Markdown查看器中正确呈现。

-->

# 文字转语音

[[在colab中打开]]

文字转语音 (TTS) 是将文字转换为自然语音的任务，语音可以生成多种语言并适用于多个说话者。目前🤗Transformers中有多个文本至语音模型，例如[Bark](../model_doc/bark)，[MMS](../model_doc/mms)，[VITS](../model_doc/vits)和[SpeechT5](../model_doc/speecht5)。

您可以使用`"text-to-audio"`流水线（或其别名`"text-to-speech"`）轻松生成音频。像Bark这样的一些模型还可以通过条件生成非语言交流，如笑话，叹息和哭泣，甚至可以添加音乐。
以下是使用Bark的`"text-to-speech"`流水线的示例：

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="suno/bark-small")
>>> text = "[clears throat] This is a test ... and I just took a long pause."
>>> output = pipe(text)
```

以下是在笔记本中使用以下代码段侦听生成的音频的方法：

```python
>>> from IPython.display import Audio
>>> Audio(output["audio"], rate=output["sampling_rate"])
```

有关Bark和其他预训练TTS模型的更多示例，请参阅我们的[音频课程](https://huggingface.co/learn/audio-course/chapter6/pre-trained_models)。

如果您正在寻找微调TTS模型，目前只能对SpeechT5进行微调。SpeechT5经过预训练，融合了文本到语音和语音到文本数据，使其能够学习文本和语音共享的隐藏表示空间。这意味着可以使用相同的预训练模型来微调不同的任务。此外，SpeechT5通过x-vector说话者嵌入支持多个说话者。

本指南的其余部分演示了如何：

1. 将最初在英语语音上训练的SpeechT5微调为荷兰语(`nl`) [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) 数据集的子集。
2. 以其中一种方式使用精化后的模型进行推断：使用流水线或直接。

开始之前，请确保已安装所有必要的库：

```bash
pip install datasets soundfile speechbrain accelerate
```

由于SpeechT5的某些功能尚未合并到官方发布中，因此请从源代码安装🤗Transformers：

```bash
pip install git+https://github.com/huggingface/transformers.git
```

<Tip>

要按照本指南，需要一个GPU。如果在笔记本中工作，请运行以下命令以检查是否有GPU可用：

```bash
!nvidia-smi
```

</Tip>

我们鼓励您登录Hugging Face帐户，以上传和共享模型。在提示时，请输入您的令牌以登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载数据集

[VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) 是一个大型的多语言语音语料库，包含从2009年至2020年欧洲议会事件记录的数据。它包含15种欧洲语言的已标记的音频转录数据。在本指南中，我们使用荷兰语子集，可以自由选择其他子集。

注意，VoxPopuli或任何其他自动语音识别（ASR）数据集可能不是训练TTS模型的最佳选择。一些对ASR有益的特征，如大量背景噪音，通常在TTS中是不可取的。但是，寻找优质的多语言和多说话者的TTS数据库可能是非常具有挑战性的。

让我们来载入数据：

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
>>> len(dataset)
20968
```

20968个示例对于微调应该足够。SpeechT5期望音频数据的采样率为16 kHz，请确保数据集中的示例满足此要求：

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

## 预处理数据

首先，定义要使用的模型检查点并加载相应的处理器：

```py
>>> from transformers import SpeechT5Processor

>>> checkpoint = "microsoft/speecht5_tts"
>>> processor = SpeechT5Processor.from_pretrained(checkpoint)
```

### SpeechT5标记化的文本清理

首先要做的是清理文本数据。您需要处理器的分词器部分来处理文本：

```py
>>> tokenizer = processor.tokenizer
```

数据集示例包含“raw_text”和“normalized_text”特征。在决定使用哪个特征作为输入文本时，请考虑SpeechT5分词器中没有任何数字标记。在“normalized_text”中，数字用文本写出。因此，更适合的是“normalized_text”，我们建议将其用作输入文本。

由于SpeechT5是在英语上训练的，它可能无法识别荷兰语数据集中的某些字符。如果保持不变，这些字符将被转换为`<unk>`标记。然而，在荷兰语中，某些字符（如`à`）用于强调音节。为了保留文本的含义，我们可以将该字符替换为常规的`a`。

要识别在前一步中鉴定的不受支持的标记，请使用`SpeechT5Tokenizer`提取数据集中的所有唯一字符，该分词器以字符为标记工作。编写“提取所有字符”的函数，该函数将来自所有示例的转录连接为一个字符串，然后将其转换为字符集合。确保在`dataset.map()`中设置`batched=True`和`batch_size=-1`，以便所有转录一次性可用于映射函数。

```py
>>> def extract_all_chars(batch):
...     all_text = " ".join(batch["normalized_text"])
...     vocab = list(set(all_text))
...     return {"vocab": [vocab], "all_text": [all_text]}


>>> vocabs = dataset.map(
...     extract_all_chars,
...     batched=True,
...     batch_size=-1,
...     keep_in_memory=True,
...     remove_columns=dataset.column_names,
... )

>>> dataset_vocab = set(vocabs["vocab"][0])
>>> tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
```

现在，您有两个字符集：一个是数据集的词汇表，一个是分词器的词汇表。要识别数据集中不在分词器中的不支持的字符，可以取两个字符集之间的差集。结果集将包含数据集中存在但不在分词器中的字符。

```py
>>> dataset_vocab - tokenizer_vocab
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```

定义一个函数来处理前一步中发现的不支持的字符，将这些字符映射到有效的标记。请注意，分词器中的空格已更换为`▁`，不需要单独处理。

```py
>>> replacements = [
...     ("à", "a"),
...     ("ç", "c"),
...     ("è", "e"),
...     ("ë", "e"),
...     ("í", "i"),
...     ("ï", "i"),
...     ("ö", "o"),
...     ("ü", "u"),
... ]


>>> def cleanup_text(inputs):
...     for src, dst in replacements:
...         inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
...     return inputs


>>> dataset = dataset.map(cleanup_text)
```

现在，您已经处理了文本中的特殊字符，是时候将重点转移到音频数据上了。

### 说话者

VoxPopuli数据集包括多个说话者的语音，但数据集中有多少个说话者呢？为了确定这一点，我们可以统计唯一说话者的数量以及每个说话者对数据集的贡献的示例数量。数据集中共有20968个示例，这些信息将更好地了解数据中说话者和示例的分布。

```py
>>> from collections import defaultdict

>>> speaker_counts = defaultdict(int)

>>> for speaker_id in dataset["speaker_id"]:
...     speaker_counts[speaker_id] += 1
```

通过绘制直方图，可以了解每个说话者有多少数据。

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.hist(speaker_counts.values(), bins=20)
>>> plt.ylabel("Speakers")
>>> plt.xlabel("Examples")
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_speakers_histogram.png" alt="Speakers histogram"/>
</div>

直方图显示，大约有三分之一的说话者在数据集中仅有少于100个示例，而其中大约十个说话者有超过500个示例。为了提高训练效率并平衡数据集，我们可以将数据限制为有100到400个示例的说话者。

```py
>>> def select_speaker(speaker_id):
...     return 100 <= speaker_counts[speaker_id] <= 400


>>> dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```

看看剩下多少个说话者：

```py
>>> len(set(dataset["speaker_id"]))
42
```

看看剩下多少个示例：

```py
>>> len(dataset)
9973
```

您还剩下了约10000个示例，约40个独特说话者，应该足够了。

请注意：一些示例较少的说话者实际上可能有更多的音频，如果示例很长。然而，确定每个说话者的总音频量需要扫描整个数据集，这是一个耗时的过程，需要加载和解码每个音频文件。因此，我们选择在此跳过此步骤。

### 说话者嵌入

为了使TTS模型能够区分多个说话者，您需要为每个示例创建一个说话者嵌入。说话者嵌入是模型的额外输入，可以捕获特定说话者的声音特征。为了生成这些说话者嵌入，使用SpeechBrain 中预训练的 [spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) 模型。

创建一个函数`create_speaker_embedding()`，接受输入音频波形并输出包含相应说话者嵌入的512个元素矢量。

```py
>>> import os
>>> import torch
>>> from speechbrain.pretrained import EncoderClassifier

>>> spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> speaker_model = EncoderClassifier.from_hparams(
...     source=spk_model_name,
...     run_opts={"device": device},
...     savedir=os.path.join("/tmp", spk_model_name),
... )


>>> def create_speaker_embedding(waveform):
...     with torch.no_grad():
...         speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
...         speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
...         speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
...     return speaker_embeddings
```

重要的是要注意，`speechbrain/spkrec-xvect-voxceleb` 模型是根据VoxCeleb数据集的英语语音训练的，而本指南中的训练示例是荷兰语。虽然我们相信该模型仍可为荷兰语数据集生成合理的说话者嵌入，但这一假设可能在所有情况下都不成立。

为了获得最佳结果，建议首先在目标语音上训练X-Vector模型。这将确保模型能够更好地捕捉荷兰语中的独特语音特征。

### 处理数据集

最后，让我们将数据处理为模型期望的格式。创建一个`prepare_dataset`函数，它接受单个示例，并使用`SpeechT5Processor`对象标记化输入文本，并将目标音频加载到对数梅尔频谱图中。 它还应将说话者嵌入作为附加输入加入。

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example = processor(
...         text=example["normalized_text"],
...         audio_target=audio["array"],
...         sampling_rate=audio["sampling_rate"],
...         return_attention_mask=False,
...     )

...     # strip off the batch dimension
...     example["labels"] = example["labels"][0]

...     # use SpeechBrain to obtain x-vector
...     example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

...     return example
```

通过查看单个示例，验证处理是否正确：

```py
>>> processed_example = prepare_dataset(dataset[0])
>>> list(processed_example.keys())
['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']
```

说话者嵌入应该是一个512个元素的矢量：

```py
>>> processed_example["speaker_embeddings"].shape
(512,)
```

标签应该是一个具有80个mel 乐器的对数梅尔频谱图。

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.imshow(processed_example["labels"].T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_1.png" alt="Log-mel spectrogram with 80 mel bins"/>
</div>

顺便说一下：如果您对此频谱图感到困惑，可能是由于您熟悉将低频放在底部，高频放在顶部的绘图惯例。然而，当使用matplotlib库将频谱图作为图像绘制时，y轴是上下翻转的，频谱图会倒置。

现在将处理函数应用于整个数据集，这将花费5到10分钟。

```py
>>> dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

您将看到一个警告，说数据集中的一些示例长度超过了模型的最大输入长度（600个标记）。从数据集中删除这些示例。为了进一步允许更大的批处理大小，我们将删除超过200个标记的数据。

```py
>>> def is_not_too_long(input_ids):
...     input_length = len(input_ids)
...     return input_length < 200


>>> dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
>>> len(dataset)
8259
```

接下来，创建一个基本的训练/测试拆分：

```py
>>> dataset = dataset.train_test_split(test_size=0.1)
```

### 数据整理器(Data Collator)

为了将多个示例合并为一个批次，需要定义一个自定义的数据收集器。这个收集器将使用填充标记填充较短的序列，确保所有示例具有相同的长度。对于频谱标签，填充部分将被特殊值`-100`替换。这个特殊值指示模型在计算频谱损失时忽略该部分的频谱。

```py
>>> from dataclasses import dataclass
>>> from typing import Any, Dict, List, Union


>>> @dataclass
... class TTSDataCollatorWithPadding:
...     processor: Any

...     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
...         input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
...         label_features = [{"input_values": feature["labels"]} for feature in features]
...         speaker_features = [feature["speaker_embeddings"] for feature in features]

...         # collate the inputs and targets into a batch
...         batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)

...         # not used during fine-tuning
...         del batch["decoder_attention_mask"]

...         # round down target lengths to multiple of reduction factor
...         if model.config.reduction_factor > 1:
...             target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
...             target_lengths = target_lengths.new(
...                 [length - length % model.config.reduction_factor for length in target_lengths]
...             )
...             max_length = max(target_lengths)
...             batch["labels"] = batch["labels"][:, :max_length]

...         # also add in the speaker embeddings
...         batch["speaker_embeddings"] = torch.tensor(speaker_features)

...         return batch
```

在SpeechT5中，模型的解码器部分的输入减少了2倍。换句话说，它从目标序列中丢弃了每个时间步长的另一部分。然后，解码器预测一个长度为原始目标序列两倍的序列。由于原始目标序列长度可能是奇数，数据收集器确保将批次的最大长度向下舍入为2的倍数。

```py 
>>> data_collator = TTSDataCollatorWithPadding(processor=processor)
```

## 训练模型

从与加载处理器相同的检查点加载预训练模型：

```py
>>> from transformers import SpeechT5ForTextToSpeech

>>> model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
```

`use_cache=True`选项与梯度检查点不兼容。在训练时将其禁用。

```py 
>>> model.config.use_cache = False
```

定义训练参数。在训练过程中，我们不计算任何评估指标，只关注损失值：

```python
>>> from transformers import Seq2SeqTrainingArguments

>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="speecht5_finetuned_voxpopuli_nl",  # 更改为您选择的仓库名称
...     per_device_train_batch_size=4,
...     gradient_accumulation_steps=8,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=4000,
...     gradient_checkpointing=True,
...     fp16=True,
...     evaluation_strategy="steps",
...     per_device_eval_batch_size=2,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     report_to=["tensorboard"],
...     load_best_model_at_end=True,
...     greater_is_better=False,
...     label_names=["labels"],
...     push_to_hub=True,
... )
```

实例化`Trainer`对象，并将模型、数据集和数据收集器传递给它。

```py
>>> from transformers import Seq2SeqTrainer

>>> trainer = Seq2SeqTrainer(
...     args=training_args,
...     model=model,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     data_collator=data_collator,
...     tokenizer=processor,
... )
```

有了上述准备，您就可以开始训练了！训练需要几个小时的时间。根据您的GPU，当您开始训练时，可能会遇到CUDA "out-of-memory"错误。在这种情况下，您可以逐渐将`per_device_train_batch_size`减小2倍，并将`gradient_accumulation_steps`增加2倍来进行补偿。

```py
>>> trainer.train()
```

为了能够在流水线中使用您的检查点，请确保将处理器与检查点一起保存：

```py
>>> processor.save_pretrained("YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

将最终模型推送到🤗 Hub：

```py
>>> trainer.push_to_hub()
```

## 推理

### 使用流水线进行推理

太好了，现在您已经微调了模型，可以用它进行推理了！首先，让我们看看如何使用对应的流水线。让我们创建一个具有您的检查点的`"text-to-speech"`流水线：

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

选择一段您想要朗读的荷兰语文本，例如：

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
```

要使用流水线的SpeechT5，您需要一个说话人嵌入。让我们从测试数据集中获取一个示例的说话人嵌入：

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

现在，您可以将文本和说话人嵌入传递给流水线，它将为您处理剩下的部分：

```py
>>> forward_params = {"speaker_embeddings": speaker_embeddings}
>>> output = pipe(text, forward_params=forward_params)
>>> output
{'audio': array([-6.82714235e-05, -4.26525949e-04,  1.06134125e-04, ...,
        -1.22392643e-03, -7.76011671e-04,  3.29112721e-04], dtype=float32),
 'sampling_rate': 16000}
```

然后，您可以听到结果：

```py
>>> from IPython.display import Audio
>>> Audio(output['audio'], rate=output['sampling_rate'])
```

### 手动运行推理

您可以在不使用流水线的情况下实现相同的推理结果，但是，需要更多的步骤。

从🤗 Hub加载模型：

```py
>>> model = SpeechT5ForTextToSpeech.from_pretrained("YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
```

从测试数据集中选择一个示例并获取说话人嵌入：

```py 
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

定义输入文本并对其进行标记化：

```py 
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
>>> inputs = processor(text=text, return_tensors="pt")
```

使用模型创建一个频谱图： 

```py
>>> spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```

如果您想要可视化频谱图，可以执行以下操作： 

```py
>>> plt.figure()
>>> plt.imshow(spectrogram.T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_2.png" alt="Generated log-mel spectrogram"/>
</div>

最后，使用语音合成器将频谱图转化为声音。

```py
>>> with torch.no_grad():
...     speech = vocoder(spectrogram)

>>> from IPython.display import Audio

>>> Audio(speech.numpy(), rate=16000)
```

根据我们的经验，从该模型获得令人满意的结果可能具有一定的挑战性。说话人嵌入的质量似乎是一个重要因素。由于SpeechT5是使用英语x-vectors进行预训练的，因此在使用英语说话人嵌入时效果最好。如果合成的语音听起来不好，请尝试使用不同的说话人嵌入。

增加训练持续时间很可能会提高结果的质量。即便如此，语音明显是荷兰语而不是英语，并且它确实捕捉到了说话者的声音特点（与示例中的原始音频相比较）。
还可以尝试使用模型的不同配置。例如，尝试使用`config.reduction_factor = 1`来查看是否改善了结果。

最后，重要的是要考虑伦理问题。虽然TTS技术有许多有用的应用，但也可能被用于恶意目的，例如在没有知情或同意的情况下模仿某人的声音。请谨慎和负责任地使用TTS。