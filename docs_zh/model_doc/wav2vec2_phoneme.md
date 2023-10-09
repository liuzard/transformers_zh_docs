<!--版权所有2021年HuggingFace团队。保留所有权利。

根据Apache License，Version 2.0许可（“许可证”）进行许可；除非符合许可证，否则您不得使用此文件。
您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以“按原样”分发的软件在
"AS IS"基础上提供，没有任何形式的担保或条件，无论是明示或暗示。有关详细信息，请参见许可证。
-->

# Wav2Vec2Phoneme（Wav2Vec2音素）

## 概述

Wav2Vec2Phoneme模型是由Qiantong Xu，Alexei Baevski和Michael Auli在[Simple and Effective Zero-shot Cross-lingual Phoneme Recognition (Xu et al., 2021)](https://arxiv.org/abs/2109.11680)中提出的。

论文摘要如下：

*自我训练、自监督预训练和无监督学习的最新进展使得无需任何标注数据即可实现良好的语音识别系统成为可能。然而，在许多情况下，这些方法并未利用到相关语言的标注数据。本文通过微调多语言预训练的wav2vec 2.0模型，使用语音特征将训练语言的音素映射到目标语言，从而在零-shot跨语言迁移学习方面扩展了先前的工作。实验证明，这种简单的方法显著优于之前引入了任务特定架构且仅使用部分单语预训练模型的工作。*

提示：

- Wav2Vec2Phoneme使用与Wav2Vec2完全相同的架构。
- Wav2Vec2Phoneme是一个接受与语音信号的原始波形对应的浮点数组的语音模型。
- Wav2Vec2Phoneme模型使用连接主义时序分类（CTC）进行训练，因此必须使用[`Wav2Vec2PhonemeCTCTokenizer`]对模型输出进行解码。
- Wav2Vec2Phoneme可以同时在多种语言上进行微调，并在单次前向传递中对未知语言进行解码，得到一个音素序列。
- 默认情况下，该模型输出一个音素序列。要将音素转换为单词序列，应使用字典和语言模型。

相关的检查点可以在https://huggingface.co/models?other=phoneme-recognition下找到。

此模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。

原始代码可以在[这里](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec)找到。

Wav2Vec2Phoneme的架构基于Wav2Vec2模型，因此可以参考[`Wav2Vec2`]的文档页，除了令牌化器。

## Wav2Vec2PhonemeCTCTokenizer

[[autodoc]] Wav2Vec2PhonemeCTCTokenizer
	- __call__
	- batch_decode
	- decode
	- phonemize