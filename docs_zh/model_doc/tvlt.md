<!--版权 2023 年HuggingFace团队。版权所有。

根据 Apache 许可证，版本 2.0（“许可证”），除非符合许可证，否则不得使用此文件。
你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，根据许可证分发的软件是基于“按原样”提供的，不附带任何担保或条件，无论是明示的还是暗示的。有关许可证的特定语言、权利和限制，请参阅许可证。

⚠️ 注意：此文件是 Markdown 格式的，但包含我们的文档生成器（类似于 MDX）的特定语法，你的 Markdown 查看器可能无法正确渲染。

-->

# TVLT

## 概述

TVLT 模型是由 Tang, Jaemin Cho, Yixin Nie, Mohit Bansal（前三位作者贡献相等）在《TVLT: Textless Vision-Language Transformer》一文中提出的。Textless Vision-Language Transformer（TVLT）是一种使用原始视觉和音频输入进行视觉与语言表示学习的模型，而不使用文本特定的模块，如标记化或自动语音识别（ASR）。它可以执行各种音频视觉和视觉语言任务，如检索、问题回答等。

以下是论文摘要：

*在本研究中，我们提出了 Textless Vision-Language Transformer（TVLT），其中均匀的Transformer块采用原始的视觉和音频输入进行视觉与语言表示学习，具有最小的模态特定设计，并且不使用文本特定的模块，如标记化或自动说话识别（ASR）。TVLT通过重建连续视频帧和音频频谱的蒙版补丁（蒙版自编码）和对比建模来进行训练，以对齐视频和音频。TVLT在各种多模态任务上都达到了与基于文本的对应模型相当的性能，比如视觉问答、图像检索、视频检索和多模态情感分析，而推理速度快了28倍，参数仅为原先的1/3。我们的研究结果表明，在不假设文本存在的情况下，有可能从低级别的视觉和音频信号中学习到紧凑高效的视觉-语言表示。*

提示：

- TVLT 模型以 `pixel_values` 和 `audio_values` 作为输入。可以使用 [`TvltProcessor`] 来为模型准备数据。
  该处理器将图像处理器（用于图像/视频模态）和音频特征提取器（用于音频模态）封装为一个处理器。
- TVLT 使用各种大小的图像/视频和音频进行训练：作者对输入图像/视频进行调整和裁剪至 224，并限制音频频谱的长度为 2048。为了使视频和音频的分批处理成为可能，作者使用 `pixel_mask` 指示真实/填充的像素，以及 `audio_mask` 指示真实/填充的音频值。
- TVLT 的设计与标准 Vision Transformer（ViT）和蒙版自编码器（MAE）类似，如 [ViTMAE](vitmae) 中所示。不同之处在于该模型包括音频模态的嵌入层。
- 该模型的 PyTorch 版本仅适用于 torch 1.10 及更高版本。

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/tvlt_architecture.png"
alt="drawing" width="600"/>
</p>

<small> TVLT 架构。来自<a href="[https://arxiv.org/abs/2102.03334](https://arxiv.org/abs/2209.14156)">原始论文</a>。 </small>

可以在[这里](https://github.com/zinengtang/TVLT)找到原始代码。该模型由[Zineng Tang](https://huggingface.co/ZinengTang)贡献。

## TvltConfig

[[autodoc]] TvltConfig

## TvltProcessor

[[autodoc]] TvltProcessor
    - __call__

## TvltImageProcessor

[[autodoc]] TvltImageProcessor
    - preprocess

## TvltFeatureExtractor

[[autodoc]] TvltFeatureExtractor
    - __call__
    
## TvltModel

[[autodoc]] TvltModel
    - forward

## TvltForPreTraining

[[autodoc]] TvltForPreTraining
    - forward

## TvltForAudioVisualClassification

[[autodoc]] TvltForAudioVisualClassification
    - forward