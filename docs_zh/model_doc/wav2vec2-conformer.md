<!--
版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版进行许可（“许可证”）;除非你遵守许可证，否则你不得使用此文件。你可以在此处获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于"按原样"的基础上，不提供任何明示或暗示的担保或条件。请参阅许可证限制下的特定语言，了解具体的管理权限和限制。

⚠️注意，此文件是Markdown格式的，但包含特定的语法以进行doc-builder（类似于MDX）的渲染。它可能无法在你的Markdown查看器中正确显示。

-->

# Wav2Vec2-Conformer

## 概述

Wav2Vec2-Conformer是由Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Sravya Popuri, Dmytro Okhonko, Juan Pino在[《fairseq S2T: Fast Speech-to-Text Modeling with fairseq》](https://arxiv.org/abs/2010.05171)的更新版本中添加的。

模型的官方结果可在该论文的Table 3和Table 4中找到。

Wav2Vec2-Conformer的权重由Meta AI团队在[Fairseq库](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md#pre-trained-models)中发布。

提示：

- Wav2Vec2-Conformer遵循与Wav2Vec2相同的架构，但将*Attention*模块替换为[Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)中介绍的*Conformer*模块。
- 对于相同数量的层，Wav2Vec2-Conformer需要比Wav2Vec2更多的参数，但同时也具有改进的词错误率。
- Wav2Vec2-Conformer使用与Wav2Vec2相同的分词器和特征提取器。
- 通过设置正确的`config.position_embeddings_type`，Wav2Vec2-Conformer可以使用无相对位置嵌入、类似Transformer-XL的位置嵌入或旋转位置嵌入。

该模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。
原始代码可在[此处](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec)找到。

## 文档资源

- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)

## Wav2Vec2ConformerConfig

[[autodoc]] Wav2Vec2ConformerConfig

## Wav2Vec2Conformer特定的输出

[[autodoc]] models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForPreTrainingOutput

## Wav2Vec2ConformerModel

[[autodoc]] Wav2Vec2ConformerModel
    - forward

## Wav2Vec2ConformerForCTC

[[autodoc]] Wav2Vec2ConformerForCTC
    - forward

## Wav2Vec2ConformerForSequenceClassification

[[autodoc]] Wav2Vec2ConformerForSequenceClassification
    - forward

## Wav2Vec2ConformerForAudioFrameClassification

[[autodoc]] Wav2Vec2ConformerForAudioFrameClassification
    - forward

## Wav2Vec2ConformerForXVector

[[autodoc]] Wav2Vec2ConformerForXVector
    - forward

## Wav2Vec2ConformerForPreTraining

[[autodoc]] Wav2Vec2ConformerForPreTraining
    - forward
-->