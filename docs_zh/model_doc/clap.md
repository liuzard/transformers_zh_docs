<!--版权2023年The HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）授权您根据
许可证进行除非合规操作，否则不得使用本文件。您可以获取
许可证的副本在这里

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面约定，按照许可证分发的软件按照
“按现状”提供，无论是明示的还是隐含的保证或条件。请参阅许可证中的
特定语言来限制许可和限制的保证。

⚠️请注意，此文件是Markdown格式的，但包含特定的语法我们的doc-builder（类似于MDX），可能不适合
在您的Markdown查看器中正确显示。

--> 

# CLAP

## 概述

CLAP模型是由Yusong Wu，Ke Chen，Tianyu Zhang，Yuchen Hui，Taylor Berg-Kirkpatrick和Shlomo Dubnov在[Large Scale Contrastive Language-Audio pretraining with
feature fusion and keyword-to-caption augmentation](https://arxiv.org/pdf/2211.06687.pdf)一文中提出的。

CLAP（对比语音-文本预训练）是一个在各种（音频，文本）对上进行训练的神经网络。它可以通过音频预测出最相关的文本片段，而无需直接优化该任务。CLAP模型使用SWINTransformer从对数梅尔频谱输入中获取音频特征，并使用RoBERTa模型获取文本特征。然后，将文本和音频特征投射到具有相同维度的潜在空间中。然后使用投影音频和文本特征之间的点积作为相似度得分。

来自论文的摘要如下：

*对比学习在多模式表示学习领域取得了显着的成功。在本文中，我们提出了一种对比语音-文本预训练的流程，通过将音频数据与自然语言描述相结合来开发音频表示。为了实现这一目标，我们首先发布了LAION-Audio-630K，一个包含633,526个来自不同数据源的音频和文本对的大型数据集。其次，我们根据不同的音频编码器和文本编码器构建了一个对比语音-文本预训练模型。我们将特征融合机制和关键词到字幕增强技术融入到模型设计中，以进一步使模型能够处理长度可变的音频输入并增强性能。第三，我们进行了全面的实验来评估我们的模型在三个任务下的性能：文本到音频检索，零样本音频分类和有监督音频分类。结果表明，在文本到音频检索任务中，我们的模型表现出优越的性能。在音频分类任务中，该模型在零样本设置下达到了最新水平的性能，并且能够获得与非零样本设置中模型结果相当的性能。LAION-Audio-6*

该模型的贡献者为[Younes Belkada](https://huggingface.co/ybelkada)和[Arthur Zucker](https://huggingface.co/ArtZucker)。
原始代码可以在[此处](https://github.com/LAION-AI/Clap)找到。

## ClapConfig

[[autodoc]] ClapConfig
    - from_text_audio_configs

## ClapTextConfig

[[autodoc]] ClapTextConfig

## ClapAudioConfig

[[autodoc]] ClapAudioConfig

## ClapFeatureExtractor

[[autodoc]] ClapFeatureExtractor

## ClapProcessor

[[autodoc]] ClapProcessor

## ClapModel

[[autodoc]] ClapModel
    - forward
    - get_text_features
    - get_audio_features

## ClapTextModel

[[autodoc]] ClapTextModel
    - forward

## ClapTextModelWithProjection

[[autodoc]] ClapTextModelWithProjection
    - forward

## ClapAudioModel

[[autodoc]] ClapAudioModel
    - forward

## ClapAudioModelWithProjection

[[autodoc]] ClapAudioModelWithProjection
    - forward