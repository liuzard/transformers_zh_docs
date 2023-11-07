<!--
版权所有2021年The HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（"许可证"），除非符合许可证的规定，否则你不得使用此文件。你可以在以下链接处获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据该许可证分发的软件以"原样"分发，不附带任何明示或暗示的担保或条件。请参阅该许可证以获取特定语言下的许可证限制。

⚠️请注意，此文件采用Markdown格式，但包含我们doc-builder（类似于MDX）的特定语法，可能无法在Markdown查看器中正确显示。

-->

# Wav2Vec2

## 总览

Wav2Vec2模型是由Alexei Baevski、Henry Zhou、Abdelrahman Mohamed和Michael Auli在[“wav2vec 2.0:自我监督学习语音表示的框架”](https://arxiv.org/abs/2006.11477)这篇论文中提出的。

论文中的摘要如下：

*我们首次展示了仅从语音音频中学习强大的表示，然后在转录的语音上进行微调，可以超越最佳的半监督方法，同时概念上更简单。wav2vec 2.0在潜在空间中掩码语音输入，并在联合学习的潜在表示的量化上解决了对比任务。在Librispeech的所有标记数据上进行的实验在干净/其他测试集上实现了1.8/3.3的词错误率（WER）。当将标记数据量减少到一小时时，wav2vec 2.0在使用100倍少的标记数据的情况下超越了之前最先进的在100小时子集上的模型。仅使用10分钟的标记数据，并在53000小时的无标记数据上进行预训练仍然实现了4.8/8.2的WER。这证明了少量标记数据下语音识别的可行性。*

提示：

- Wav2Vec2是一个接受浮点数组作为原始语音信号波形的语音模型。
- Wav2Vec2模型使用了连续时间分类（CTC）进行训练，因此必须使用[`Wav2Vec2CTCTokenizer`]对模型输出进行解码。

该模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。

## 资源

以下是官方和社区（由🌎表示）资源的列表，可帮助你快速入门Wav2Vec2。如果你有兴趣提交资源以包含在此处，请随时提出Pull Request，我们将对其进行审查！该资源理想情况下应展示出一些新内容，而不是重复现有资源。

<PipelineTag pipeline="audio-classification"/>

- 有关如何[利用预训练的Wav2Vec2模型进行情感分类](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb)的笔记本。🌎
- [`Wav2Vec2ForCTC`]在此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb)中得到支持。
- [音频分类任务指南](../tasks/audio_classification)

<PipelineTag pipeline="automatic-speech-recognition"/>

- 一篇关于[增强Wav2Vec2与🤗Transformers中的n-gram](https://huggingface.co/blog/wav2vec2-with-ngram)的博客文章。
- 一篇关于如何[使用🤗Transformers来微调英语ASR中的Wav2Vec2](https://huggingface.co/blog/fine-tune-wav2vec2-english)的博客文章。
- 关于[用🤗Transformers微调XLS-R进行多语言ASR](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)的博客文章。
- 一份有关如何[使用Wav2Vec2从任何视频创建YouTube字幕](https://colab.research.google.com/github/Muennighoff/ytclipcc/blob/main/wav2vec_youtube_captions.ipynb)的笔记本。🌎
- [`Wav2Vec2ForCTC`]由一份关于[如何在英语中微调语音识别模型](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb)和一份关于[如何在任何语言中微调语音识别模型](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb)的笔记本得到支持。
- [自动语音识别任务指南](../tasks/asr)

🚀部署

- 关于如何使用Hugging Face的Transformers和Amazon SageMaker部署Wav2Vec2进行[自动语音识别](https://www.philschmid.de/automatic-speech-recognition-sagemaker)的博客文章。

## Wav2Vec2Config

[[autodoc]] Wav2Vec2Config

## Wav2Vec2CTCTokenizer

[[autodoc]] Wav2Vec2CTCTokenizer
    - __call__
    - save_vocabulary
    - decode
    - batch_decode
    - set_target_lang

## Wav2Vec2FeatureExtractor

[[autodoc]] Wav2Vec2FeatureExtractor
    - __call__

## Wav2Vec2Processor

[[autodoc]] Wav2Vec2Processor
    - __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Wav2Vec2ProcessorWithLM

[[autodoc]] Wav2Vec2ProcessorWithLM
    - __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

### 解码多个音频

如果你计划对多个音频进行解码，应考虑使用[`~Wav2Vec2ProcessorWithLM.batch_decode`]，并传递一个已实例化的`multiprocessing.Pool`。
否则，[`~Wav2Vec2ProcessorWithLM.batch_decode`]的性能将比逐个调用[`~Wav2Vec2ProcessorWithLM.decode`]要慢，因为它在每次调用时内部实例化一个新的`Pool`。 请参考以下示例：

```python
>>> # 让我们看看如何使用用户管理的池来批量解码多个音频
>>> from multiprocessing import get_context
>>> from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
>>> from datasets import load_dataset
>>> import datasets
>>> import torch

>>> # 导入模型、特征提取器和分词器
>>> model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm").to("cuda")
>>> processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

>>> # 加载示例数据集
>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))


>>> def map_to_array(batch):
...     batch["speech"] = batch["audio"]["array"]
...     return batch


>>> # 为批量推理准备语音数据
>>> dataset = dataset.map(map_to_array, remove_columns=["audio"])


>>> def map_to_pred(batch, pool):
...     inputs = processor(batch["speech"], sampling_rate=16_000, padding=True, return_tensors="pt")
...     inputs = {k: v.to("cuda") for k, v in inputs.items()}

...     with torch.no_grad():
...         logits = model(**inputs).logits

...     transcription = processor.batch_decode(logits.cpu().numpy(), pool).text
...     batch["transcription"] = transcription
...     return batch


>>> # 注意：pool应在`Wav2Vec2ProcessorWithLM`之后实例化。否则，LM将对池的子进程不可用
>>> 获取上下文("fork")工具栏被池作为进程表":
...     输入模型，特征提取器，tokenizer
...     batch["speech"] = batch["audio"]["array"]
...     返回批

...     批处理=批处理中的预测,批处理=True,批处理大小=2,fn_kwargs={"pool": pool},remove_columns=["speech"]
... )

>>> batch["transcription"][:2]
['MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL', "NOR IS MISTER COULTER'S MANNER LESS INTERESTING THAN HIS MATTER"]
```

## Wav2Vec2特定输出

[[autodoc]] models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput

[[autodoc]] models.wav2vec2.modeling_wav2vec2.Wav2Vec2BaseModelOutput

[[autodoc]] models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput

[[autodoc]] models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput

[[autodoc]] models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput

## Wav2Vec2Model

[[autodoc]] Wav2Vec2Model
    - forward

## Wav2Vec2ForCTC

[[autodoc]] Wav2Vec2ForCTC
    - forward
    - load_adapter

## Wav2Vec2ForSequenceClassification

[[autodoc]] Wav2Vec2ForSequenceClassification
    - forward

## Wav2Vec2ForAudioFrameClassification

[[autodoc]] Wav2Vec2ForAudioFrameClassification
    - forward

## Wav2Vec2ForXVector

[[autodoc]] Wav2Vec2ForXVector
    - forward

## Wav2Vec2ForPreTraining

[[autodoc]] Wav2Vec2ForPreTraining
    - forward

## TFWav2Vec2Model

[[autodoc]] TFWav2Vec2Model
    - call

## TFWav2Vec2ForSequenceClassification

[[autodoc]] TFWav2Vec2ForSequenceClassification
    - call

## TFWav2Vec2ForCTC

[[autodoc]] TFWav2Vec2ForCTC
    - call

## FlaxWav2Vec2Model

[[autodoc]] FlaxWav2Vec2Model
    - __call__

## FlaxWav2Vec2ForCTC

[[autodoc]] FlaxWav2Vec2ForCTC
    - __call__

## FlaxWav2Vec2ForPreTraining

[[autodoc]] FlaxWav2Vec2ForPreTraining
    - __call__
