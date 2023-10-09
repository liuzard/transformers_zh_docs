<!--版权所有2021年HuggingFace团队。保留所有权利。

根据Apache License，版本2.0（“许可证”）进行许可；除非符合许可证，否则你不得使用此文件。你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，此软件根据“按现状提供”基础进行分发，不附带任何形式的担保或条件。请参阅许可证，了解许可下的特定语言和限制。

⚠️ 请注意，此文件采用Markdown格式，但包含特定于我们的文档生成器的语法（类似于MDX），可能在Markdown查看器中无法正常渲染。

-->

# SEW

## 概述

SEW（Squeezed and Efficient Wav2Vec）是由Felix Wu、Kwangyoun Kim、Jing Pan、Kyu Han、Kilian Q. Weinberger和Yoav Artzi在[性能效率与无监督语音识别预训练的权衡](https://arxiv.org/abs/2109.06870)中提出的。

来自该论文的摘要如下：

*本文是关于自动语音识别（ASR）预训练模型的性能效率权衡研究。我们专注于wav2vec 2.0，并规范了几种影响模型性能和效率的架构设计。综合我们的所有观察结果，我们引入了SEW（Squeezed and Efficient Wav2vec），这是一个在性能和效率两个维度上都有显著改进的预训练模型架构，适用于各种训练设置。例如，在LibriSpeech的100h-960h半监督设置下，相比于wav2vec 2.0，SEW在推理速度上提高了1.9倍，而词误率相对减少了13.5%。在类似的推理时间下，SEW可以在不同的模型尺寸上将词误率减少25-50%。*

提示：

- SEW是一个接受与语音信号的原始波形相对应的浮点数组的语音模型。
- SEWForCTC使用连续时间分类（CTC）进行微调，因此模型输出必须使用[`Wav2Vec2CTCTokenizer`]进行解码。

该模型由[anton-l](https://huggingface.co/anton-l)贡献。

## 文档资源

- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)

## SEWConfig

[[autodoc]] SEWConfig

## SEWModel

[[autodoc]] SEWModel
    - forward

## SEWForCTC

[[autodoc]] SEWForCTC
    - forward

## SEWForSequenceClassification

[[autodoc]] SEWForSequenceClassification
    - forward