<!--版权所有©2021年Huging Face团队。

根据Apache许可证第2版（“许可证”），除非符合许可证，否则你不得使用此文件。
你可以在以下位置获得许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”的基础分发的，不附带任何形式的明示或暗示的担保。请参阅许可证以获取许可证下的特定语言和限制的规定。
⚠️请注意，此文件在Markdown中，但包含我们Doc构建器的特定语法（类似于MDX），你的Markdown查看器可能无法正确呈现。-->

# Speech2Text2

## 概述

Speech2Text2模型是与[Wav2Vec2](wav2vec2)搭配使用的，用于语音翻译模型的一个模型。该模型在[大规模自监督和半监督学习的语音翻译中](https://arxiv.org/abs/2104.06678)一文中由Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau提出。

Speech2Text2是一个*解码器-only*的transformer模型，可以与任何语音*编码器-only*（如[Wav2Vec2](wav2vec2)或[Hubert](hubert)）一起用于语音到文本任务。有关如何将Speech2Text2与任何语音*编码器-only*模型结合的详细信息，请参阅[SpeechEncoderDecoder](speech-encoder-decoder)类。

此模型由[Patrick von Platen](https://huggingface.co/patrickvonplaten)贡献。

原始代码可在[此处](https://github.com/pytorch/fairseq/blob/1f7ef9ed1e1061f8c7f88f8b94c7186834398690/fairseq/models/wav2vec/wav2vec2_asr.py#L266)找到。


提示：
- Speech2Text2在CoVoST Speech Translation数据集上取得了最新的结果。更多信息，请参见[官方模型](https://huggingface.co/models?other=speech2text2)。
- Speech2Text2始终在[SpeechEncoderDecoder](speech-encoder-decoder)框架内使用。
- Speech2Text2的分词器是基于[fastBPE](https://github.com/glample/fastBPE)的。

## 推断

Speech2Text2的[`SpeechEncoderDecoderModel`]模型接受语音的原始波形输入值，并利用[`~generation.GenerationMixin.generate`]将输入语音自回归地转换为目标语言。

[`Wav2Vec2FeatureExtractor`]类负责预处理输入语音，[`Speech2Text2Tokenizer`]将生成的目标标记解码为目标字符串。[`Speech2Text2Processor`]将[`Wav2Vec2FeatureExtractor`]和[`Speech2Text2Tokenizer`]封装到一个单一实例中，既提取输入特征，又解码预测的标记ID。

- 语音翻译的逐步实现

```python
>>> import torch
>>> from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
>>> processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
>>> generated_ids = model.generate(inputs=inputs["input_values"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids)
```

- 通过Pipeline进行语音翻译

自动语音识别Pipeline还可以用于只需几行代码即可翻译语音。

```python
>>> from datasets import load_dataset
>>> from transformers import pipeline

>>> librispeech_en = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> asr = pipeline(
...     "automatic-speech-recognition",
...     model="facebook/s2t-wav2vec2-large-en-de",
...     feature_extractor="facebook/s2t-wav2vec2-large-en-de",
... )

>>> translation_de = asr(librispeech_en[0]["file"])
```

请参见[模型库](https://huggingface.co/models?filter=speech2text2)查找Speech2Text2的检查点。

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)

## Speech2Text2Config

[[autodoc]] Speech2Text2Config

## Speech2TextTokenizer

[[autodoc]] Speech2Text2Tokenizer
    - batch_decode
    - decode
    - save_vocabulary

## Speech2Text2Processor

[[autodoc]] Speech2Text2Processor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Speech2Text2ForCausalLM

[[autodoc]] Speech2Text2ForCausalLM
    - forward