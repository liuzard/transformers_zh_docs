<!--版权所有2021年The HuggingFace团队

根据Apache许可证2.0版("许可证")，除非符合许可证的规定，否则你不得使用此文件。你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以获取特定语言的权限和限制。

⚠️请注意，该文件是使用Markdown格式编写的，但包含有关我们doc-builder的特定语法(MDX类似)，在Markdown查看器中可能无法正确显示。-->

# UniSpeech

## 概述

UniSpeech模型是由Chengyi Wang、Yu Wu、Yao Qian、Kenichi Kumatani、Shujie Liu、Furu Wei、Michael Zeng、Xuedong Huang在《UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data》中提出的。论文的摘要如下：

*在本文中，我们提出了一个统一的预训练方法，称为UniSpeech，用于使用有标签和无标签的数据学习语音表示，其中以多任务学习的方式进行监督音标CTC学习和音标感知对比自监督学习。结果表示能够捕获更与音标结构相关的信息，并提高跨语言和领域的泛化能力。我们在公共的CommonVoice语料库上评估了UniSpeech在跨语言表示学习方面的有效性。结果显示，UniSpeech相对于自监督预训练和监督迁移学习在语音识别方面的相对电话错误率降低最大可达13.4%和17.8%（在所有测试语言上取平均）。UniSpeech的可迁移性也在领域转移语音识别任务中得到了证明，即相对于先前方法有6%的相对词错误率降低。*

提示：

- UniSpeech是一个接受与语音信号的原始波形相对应的浮点数数组的语音模型。请使用[`Wav2Vec2Processor`]进行特征提取。
- UniSpeech模型可以使用连接主义时间分类（CTC）进行微调，因此必须使用[`Wav2Vec2CTCTokenizer`]对模型的输出进行解码。

此模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。作者的代码可以在[这里](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech)找到。

## 文档资源

- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)

## UniSpeechConfig

[[autodoc]] UniSpeechConfig

## UniSpeech特定的输出

[[autodoc]] models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput

## UniSpeechModel

[[autodoc]] UniSpeechModel
    - forward

## UniSpeechForCTC

[[autodoc]] UniSpeechForCTC
    - forward

## UniSpeechForSequenceClassification

[[autodoc]] UniSpeechForSequenceClassification
    - forward

## UniSpeechForPreTraining

[[autodoc]] UniSpeechForPreTraining
    - forward