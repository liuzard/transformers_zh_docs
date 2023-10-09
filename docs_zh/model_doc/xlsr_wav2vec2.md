<!--
版权所有2021 The HuggingFace Team。

根据Apache许可证第2版（“许可证”），你不得使用本文件，除非符合许可证的规定。
你可以在以下网址获取许可证副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“AS IS”的基础上分发的，不附带任何明示或暗示的保证或条件。
请查看许可证以了解许可证下的特定语言和限制。

⚠️请注意，此文件是Markdown格式，但包含我们文档构建器（类似于MDX）的特定语法，可能无法在你的Markdown查看器中正确渲染。

-->

# XLSR-Wav2Vec2

## 概述

XLSR-Wav2Vec2模型是由Alexis Conneau，Alexei Baevski，Ronan Collobert，Abdelrahman Mohamed，Michael Auli在论文[Unsupervised Cross-Lingual Representation Learning For Speech Recognition](https://arxiv.org/abs/2006.13979)中提出的。

论文中的摘要如下：

*本文提出了XLSR，通过对多种语言中的语音的原始波形进行预训练，从而学习跨语言语音表示。我们建立在wav2vec 2.0的基础上，该模型通过解决对遮蔽的潜在语音表示进行对比任务来进行训练，并同时学习跨语言共享的潜在语音的量化。结果模型在标记数据上进行微调，实验表明，跨语言预训练显著优于单语言预训练。在CommonVoice基准测试中，相对于已知最佳结果，XLSR显示出72%的音素错误率降低。在BABEL上，我们的方法相对于类似系统将词错误率提高了16%。我们的方法实现了一个单一的多语言语音识别模型，与强大的个体模型相比具有竞争力。分析表明，潜在的离散语音表示在语言之间是共享的，与相关语言的共享增加。我们希望通过发布XLSR-53，一个在53种语言中进行预训练的大型模型，以推动低资源语音理解的研究。*

提示：

- XLSR-Wav2Vec2是一个语音模型，接受与语音信号的原始波形相对应的浮点数组。
- XLSR-Wav2Vec2模型使用连接主义时间分类（CTC）进行训练，因此模型输出必须使用[`Wav2Vec2CTCTokenizer`]进行解码。

XLSR-Wav2Vec2的架构基于Wav2Vec2模型，因此可以参考[Wav2Vec2的文档页面](wav2vec2)。

可以在[此处](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec)找到原始代码。