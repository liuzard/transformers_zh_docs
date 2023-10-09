<!--
版权 2022 年 HuggingFace 团队。版权所有。

根据 Apache 许可证第 2.0 版（"许可证"），你不得使用此文件，除非符合许可证。
你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

在适用法律规定或书面协议同意的情况下，根据许可证分发的软件按"原样"分发，不提供任何担保或条件，无论是明示的还是默示的。有关许可证下的特定语言的权限和限制的详细信息，请参阅许可证。

⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特殊语法，你的 Markdown 查看器可能不会正确呈现。

-->

# M-CTC-T

<Tip warning={true}>

此模型仅处于维护模式，因此我们不会接受任何更改其代码的新 PR。

如果在运行此模型时遇到任何问题，请重新安装最后支持此模型的版本：v4.30.0。
你可以通过运行以下命令来实现：`pip install -U transformers==4.30.0`。

</Tip>

## 概述

M-CTC-T 模型是由 Loren Lugosch、Tatiana Likhomanenko、Gabriel Synnaeve 和 Ronan Collobert 在论文[Pseudo-Labeling For Massively Multilingual Speech Recognition](https://arxiv.org/abs/2111.00161)中提出的。该模型是一个 10 亿参数的 Transformer 编码器，带有一个 CTC 头，其中包含 8065 个字符标签，以及一个语言识别头，其中包含 60 个语言 ID 标签。它使用 Common Voice（版本 6.1，2020 年 12 月发布）和 VoxPopuli 进行训练。在对 Common Voice 和 VoxPopuli 进行训练后，该模型只使用 Common Voice 进行训练。标签是未归一化的字符级转录文本（标点符号和大写字母未去除）。模型将 16kHz 音频信号的 Mel 滤波器组特征作为输入。

论文中的摘要如下：

*通过伪标记的半监督学习已成为最先进的单语言语音识别系统的基础。在这项工作中，我们将伪标记扩展到了具有 60 种语言的大规模多语言语音识别。我们提出了一种简单的伪标记配方，即使用监督的多语言模型进行初步调优，然后使用半监督学习在目标语言上进行微调，为该语言生成伪标签，并使用所有语言的伪标签训练最终模型，可以从零开始或通过微调。对标记的 Common Voice 数据集和未标记的 VoxPopuli 数据集的实验表明，我们的配方可以产生性能更好的模型，对许多语言也具有良好的转移能力，适用于 LibriSpeech。*



此模型由[cwkeam](https://huggingface.co/cwkeam)提供。原始代码可以在[此处](https://github.com/flashlight/wav2letter/tree/main/recipes/mling_pl)找到。

## 文档资源

- [自动语音识别任务指南](../tasks/asr)

提示：

- 此模型的 PyTorch 版本仅在torch 1.9及更高版本中可用。

## MCTCTConfig

[[autodoc]] MCTCTConfig

## MCTCTFeatureExtractor

[[autodoc]] MCTCTFeatureExtractor
    - __call__

## MCTCTProcessor

[[autodoc]] MCTCTProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode


## MCTCTModel

[[autodoc]] MCTCTModel
    - forward

## MCTCTForCTC

[[autodoc]] MCTCTForCTC
    - forward
