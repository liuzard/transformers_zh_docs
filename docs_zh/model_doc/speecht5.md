版权所有2023年HuggingFace团队。版权所有。

根据Apache许可证，版本2.0（“许可证”），您不得使用此文件，除非符合许可证的规定。您可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于“按原样”提供，不附带任何明示或暗示的保证或条件。请参阅许可证中的特定语言，了解许可证下的特定权限和限制。

⚠️请注意，此文件以Markdown格式编写，但包含我们doc-builder的特定语法（类似于MDX），可能无法在Markdown查看器中正确呈现。

# SpeechT5

## 概述

SpeechT5模型是由Junyi Ao，Rui Wang，Long Zhou，Chengyi Wang，Shuo Ren，Yu Wu，Shujie Liu，Tom Ko，Qing Li，Yu Zhang，Zhihua Wei，Yao Qian，Jinyu Li和Furu Wei在论文《SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing》中提出的。论文链接：[SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205)。

论文摘要如下：

受T5（Text-To-Text Transfer Transformer）在预训练的自然语言处理模型中的成功启发，我们提出了一个统一模态SpeechT5框架，用于自监督语音/文本表示学习的编码器-解码器预训练。SpeechT5框架由一个共享的编码器-解码器网络和六个模态特定（语音/文本）的预处理网络和后处理网络组成。通过预处理输入的语音/文本，共享的编码器-解码器网络对序列-序列转换进行建模，然后后处理网络根据解码器的输出在语音/文本模态中生成输出。通过利用大规模的无标签语音和文本数据，我们对SpeechT5进行预训练，以学习统一模态表示，希望提高语音和文本的建模能力。为了将文本和语音信息对齐到统一的语义空间中，我们提出了一种交叉模态向量量化方法，该方法将语音/文本状态与潜在单元随机混合在编码器和解码器之间作为接口。广泛的评估表明，该SpeechT5框架在包括自动语音识别、语音合成、语音翻译、语音转换、语音增强和说话人识别在内的各种口语处理任务上具有优越性能。

该模型由[Matthijs](https://huggingface.co/Matthijs)贡献。原始代码可在[此处](https://github.com/microsoft/SpeechT5)找到。

## SpeechT5Config

[[autodoc]] SpeechT5Config

## SpeechT5HifiGanConfig

[[autodoc]] SpeechT5HifiGanConfig

## SpeechT5Tokenizer

[[autodoc]] SpeechT5Tokenizer
    - __call__
    - save_vocabulary
    - decode
    - batch_decode

## SpeechT5FeatureExtractor

[[autodoc]] SpeechT5FeatureExtractor
    - __call__

## SpeechT5Processor

[[autodoc]] SpeechT5Processor
    - __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## SpeechT5Model

[[autodoc]] SpeechT5Model
    - forward

## SpeechT5ForSpeechToText

[[autodoc]] SpeechT5ForSpeechToText
    - forward

## SpeechT5ForTextToSpeech

[[autodoc]] SpeechT5ForTextToSpeech
    - forward
    - generate

## SpeechT5ForSpeechToSpeech

[[autodoc]] SpeechT5ForSpeechToSpeech
    - forward
    - generate_speech

## SpeechT5HifiGan

[[autodoc]] SpeechT5HifiGan
    - forward