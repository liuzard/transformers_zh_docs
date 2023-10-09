<!--版权所有2021年The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）的规定，除非符合许可证的规定，否则你不得使用此文件。
你可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可以及其附带的条款分发的软件是基于“按原样”基础分发的，无论是明示还是暗示，不附带任何明示或暗示的保证。有关许可证的详细信息，请参阅许可证。

⚠️请注意，此文件采用Markdown格式，但包含特定于我们文档生成器（类似于MDX）的语法，可能无法在你的Markdown查看器中正确显示。-->

# 语音转文本

## 概述

[S2T: 基于fairseq的快速语音转文本模型](https://arxiv.org/abs/2010.05171)由Changhan Wang、Yun Tang、Xutai Ma、Anne Wu、Dmytro Okhonko、Juan Pino在论文中提出。这是一个基于transformer的序列到序列（编码器-解码器）模型，用于端到端自动语音识别（ASR）和语音翻译（ST）。它使用卷积下采样器在输入进入编码器之前将语音输入的长度减少3/4。模型使用标准的自回归交叉熵损失进行训练，并自回归地生成转录文本/翻译结果。Speech2Text模型已在多个ASR和ST数据集上进行了微调：[LibriSpeech](http://www.openslr.org/12)、[CoVoST 2](https://github.com/facebookresearch/covost)、[MuST-C](https://ict.fbk.eu/must-c/)。

该模型由[valhalla](https://huggingface.co/valhalla)贡献。原始代码可以在[此处](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text)找到。


## 推理

Speech2Text是一个接受从语音信号中提取出的对数梅尔滤波器组特征的浮点张量的语音模型。它是一个基于transformer的序列到序列模型，因此转录文本/翻译结果是自回归生成的。可以使用`generate()`方法进行推理。

[`Speech2TextFeatureExtractor`]类负责提取对数梅尔滤波器组特征。[`Speech2TextProcessor`]将[`Speech2TextFeatureExtractor`]和[`Speech2TextTokenizer`]封装到一个实例中，用于同时提取输入特征并解码预测的标记ID。

特征提取器依赖于`torchaudio`，分词器依赖于`sentencepiece`，所以在运行示例之前，请务必安装这些软件包。你可以使用`pip install transformers"[speech, sentencepiece]"`将它们作为额外的语音依赖项安装，或者使用`pip install torchaudio sentencepiece`单独安装软件包。此外，`torchaudio`需要[libsndfile](http://www.mega-nerd.com/libsndfile/)软件包的开发版本，该软件包可以通过系统软件包管理器安装。在Ubuntu上，可以按照以下步骤进行安装：`apt install libsndfile1-dev`


- ASR和语音翻译

```python
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> transcription
['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel']
```

- 多语言语音翻译

  对于多语言语音翻译模型，`eos_token_id`用作`decoder_start_token_id`，并且目标语言ID被强制作为第一个生成的标记。要将目标语言ID强制为第一个生成的标记，请将`forced_bos_token_id`参数传递给`generate()`方法。以下示例演示如何使用*facebook/s2t-medium-mustc-multilingual-st*检查点将英语语音翻译为法语文本。

```python
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(
...     inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"],
... )

>>> translation = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> translation
["(Vidéo) Si M. Kilder est l'apossible des classes moyennes, et nous sommes heureux d'être accueillis dans son évangile."]
```

请参阅[model hub](https://huggingface.co/models?filter=speech_to_text)以查找Speech2Text检查点。


## Speech2TextConfig

[[autodoc]] Speech2TextConfig

## Speech2TextTokenizer

[[autodoc]] Speech2TextTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## Speech2TextFeatureExtractor

[[autodoc]] Speech2TextFeatureExtractor
    - __call__

## Speech2TextProcessor

[[autodoc]] Speech2TextProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Speech2TextModel

[[autodoc]] Speech2TextModel
    - forward

## Speech2TextForConditionalGeneration

[[autodoc]] Speech2TextForConditionalGeneration
    - forward

## TFSpeech2TextModel

[[autodoc]] TFSpeech2TextModel
    - call

## TFSpeech2TextForConditionalGeneration

[[autodoc]] TFSpeech2TextForConditionalGeneration
    - call