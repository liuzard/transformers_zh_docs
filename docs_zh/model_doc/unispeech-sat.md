<!--版权所有2021年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（"许可证"）授权；除非符合许可证的规定，
否则不得使用此文件。你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以"按原样"的方式分发的软件是根据许可证分发的，
不带有任何形式的明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言和限制的规定。

⚠️请注意，此文件是Markdown格式，但包含针对我们的文档构建程序（类似MDX）的特定语法，
这可能在你的Markdown查看器中无法正确渲染。-->

# UniSpeech-SAT

## 概述

UniSpeech-SAT模型是由Sanyuan Chen、Yu Wu、Chengyi Wang、Zhengyang Chen、Zhuo Chen、Shujie Liu、Jian Wu、Yao Qian、 Furu Wei、Jinyu Li、Xiangzhan Yu在[UniSpeech-SAT: Universal Speech Representation Learning with Speaker Aware Pre-Training](https://arxiv.org/abs/2110.05752)中提出的。

论文摘要如下：

*自我监督学习（SSL）是语音处理的长期目标，因为它利用大规模的无标注数据，并避免了大量的人工标记。近年来，在语音识别中应用自我监督学习取得了巨大的成功，但在对建模说话人特征应用SSL方面，探索还很有限。本文旨在改进现有的用于说话人表示学习的SSL框架。为了增强无监督说话人信息提取，引入了两种方法。首先，将多任务学习应用于当前的SSL框架，将话语级对比损失与SSL目标函数相结合。其次，为了更好的说话人判别，我们提出了一种用于数据增强的话语混合策略，在训练过程中无监督地创建额外的重叠话语并进行合并。我们将这些方法集成到HuBERT框架中。在SUPERB基准测试上的实验结果表明，所提出的系统在通用表示学习方面取得了最先进的性能，特别适用于以说话人标识为导向的任务。通过一项消融研究验证了每种提出方法的有效性。最后，我们扩大了训练数据集，达到了94000小时的公共音频数据，进一步提高了SUPERB任务的性能。*

提示:

- UniSpeechSat是一个接受与语音信号的原始波形相对应的浮点数组的语音模型。请使用[`Wav2Vec2Processor`]进行特征提取。
- UniSpeechSat模型可以使用联结时序分类（CTC）进行微调，因此必须使用[`Wav2Vec2CTCTokenizer`]对模型输出进行解码。
- UniSpeechSat在说话人验证、说话人识别和说话人分割任务上表现特别好。

该模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。作者的代码可在[此处](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT)找到。

## 文档资源

- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)

## UniSpeechSatConfig

[[autodoc]] UniSpeechSatConfig

## UniSpeechSat特定的输出

[[autodoc]] models.unispeech_sat.modeling_unispeech_sat.UniSpeechSatForPreTrainingOutput

## UniSpeechSatModel

[[autodoc]] UniSpeechSatModel
    - forward

## UniSpeechSatForCTC

[[autodoc]] UniSpeechSatForCTC
    - forward

## UniSpeechSatForSequenceClassification

[[autodoc]] UniSpeechSatForSequenceClassification
    - forward

## UniSpeechSatForAudioFrameClassification

[[autodoc]] UniSpeechSatForAudioFrameClassification
    - forward

## UniSpeechSatForXVector

[[autodoc]] UniSpeechSatForXVector
    - forward

## UniSpeechSatForPreTraining

[[autodoc]] UniSpeechSatForPreTraining
    - forward