<!--版权所有2021 HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）的规定，你不得使用此文件，除非符合许可证的规定。
你可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证的规定分发，不附带任何明示或暗示的担保或条件。请参阅许可证以获取
适用于具体语言的权限和限制的条款。

⚠️请注意，此文件是Markdown格式，但包含了特定语法，用于我们的文档构建工具（类似于MDX），可能在你的Markdown阅读器
中无法正确呈现。

-->

# WavLM

## 概述

[WavLM：大规模自监督预训练用于全叠输语音处理](https://arxiv.org/abs/2110.13900)是由Sanyuan Chen、Chengyi Wang、
Zhengyang Chen、Yu Wu、Shujie Liu、Zhuo Chen、Jinyu Li、Naoyuki Kanda、Takuya Yoshioka、Xiong Xiao、Jian Wu、Long Zhou、
Shuo Ren、Yanmin Qian、Yao Qian、Jian Wu、Michael Zeng、Furu Wei提出的。

论文摘要如下：

*自监督学习（SSL）在语音识别方面取得了巨大的成功，但是在其他语音处理任务上仍然有限的探索。由于语音信号包含了包括
说话者身份、语音语音、不言自明的内容等多方面的信息，学习适用于所有语音任务的通用表示形式具有挑战性。在本文中，
我们提出了一种新的预训练模型WavLM，用于解决全叠跟进语音任务。WavLM基于HuBERT框架构建，重点在于语音内容建模和
说话者身份保护。我们首先采用门限相对位置偏差将变压器结构装备起来，以提高其在识别任务上的能力。为了更好的说话者区分，
我们提出了一种话语混合训练策略，在模型训练过程中无监督地创建额外的重叠话语并将其合并。最后，我们将训练数据集从60k小时
扩展到94k小时。WavLM Large在SUPERB基准测试中取得了最先进的性能，并为他们的代表性基准测试中的各种语音处理任务
带来了显著的改进。*

提示：

- WavLM是一个接受与语音信号的原始波形对应的浮点数组的语音模型。请使用[`Wav2Vec2Processor`]进行特征提取。
- WavLM模型可以使用连接时序分类（CTC）进行微调，因此必须使用[`Wav2Vec2CTCTokenizer`]对模型输出进行解码。
- WavLM在发音验证、发音识别和说话人分割任务上表现特别出色。

相关检查点可以在https://huggingface.co/models?other=wavlm找到。

此模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。作者的代码可以在
[这里](https://github.com/microsoft/unilm/tree/master/wavlm)找到。

## 文档资源

- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)

## WavLMConfig

[[autodoc]] WavLMConfig

## WavLMModel

[[autodoc]] WavLMModel
    - forward

## WavLMForCTC

[[autodoc]] WavLMForCTC
    - forward

## WavLMForSequenceClassification

[[autodoc]] WavLMForSequenceClassification
    - forward

## WavLMForAudioFrameClassification

[[autodoc]] WavLMForAudioFrameClassification
    - forward

## WavLMForXVector

[[autodoc]] WavLMForXVector
    - forward