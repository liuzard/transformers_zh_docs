<!--版权所有 2021 年 HuggingFace 团队保留。

根据 Apache 许可证，版本 2.0（"许可证"），除非符合许可证，否则你不得使用此文件。
你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据许可证分发的软件是基于"按原样"的基础分发的，没有任何明示或暗示的保证和条件。
请参阅许可证以获取适用的语言和许可证下的限制。

⚠️ 请注意，此文件以 Markdown 格式编写，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能无法在你的 Markdown 查看器中正确呈现。

-->

# SEW-D

## 概述

SEW-D（Squeezed and Efficient Wav2Vec with Disentangled attention）由Felix Wu、Kwangyoun Kim、Jing Pan、Kyu Han、Kilian Q. Weinberger、Yoav Artzi在文章《性能和效率之间的权衡：用于语音识别的无监督预训练研究》（https://arxiv.org/abs/2109.06870）中提出。

来自论文的摘要如下：

*本文研究了自动语音识别（ASR）预训练模型在性能和效率之间的权衡。我们着重研究了wav2vec 2.0，并形式化了一些影响模型性能和效率的架构设计。综合我们的所有观察，我们引入了SEW（Squeezed and Efficient Wav2vec），这是一个预训练模型架构，在各种训练设置下在性能和效率两个方面都有显著改进。例如，在LibriSpeech的100h-960h半监督设置下，与wav2vec 2.0相比，SEW的推理速度提高了1.9倍，相对于单词错误率降低了13.5%。在相似的推理时间内，SEW在不同模型大小上将单词错误率降低了25-50%。*

提示：

- SEW-D是一个接受与语音信号的原始波形对应的浮点数组的语音模型。
- SEWDForCTC使用链接时序分类（CTC）进行微调，因此模型输出必须使用[`Wav2Vec2CTCTokenizer`]进行解码。

此模型由[anton-l](https://huggingface.co/anton-l)贡献。

## 文档资源

- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)

## SEWDConfig

[[autodoc]] SEWDConfig

## SEWDModel

[[autodoc]] SEWDModel
    - forward

## SEWDForCTC

[[autodoc]] SEWDForCTC
    - forward

## SEWDForSequenceClassification

[[autodoc]] SEWDForSequenceClassification
    - forward