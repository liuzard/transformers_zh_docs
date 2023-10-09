<!--专有权利 2022 The HuggingFace Team.版权所有。

根据Apache许可证，版本2.0（“许可证”）进行许可；除非符合许可证的规定，否则你不得使用此文件。
你可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“AS IS”基础分发的，不附带任何担保或条件，无论是明示还是暗示。请查看许可证，了解特定语言的权限和限制。

⚠️请注意，此文件采用Markdown格式，但包含我们的文档生成器（类似于MDX）的特定句法，可能无法在Markdown查看器中正确呈现。-->

# 音频光谱图Transformer

## 概述

音频光谱图Transformer模型是由Yuan Gong，Yu-An Chung和James Glass在[AST: 音频光谱图Transformer](https://arxiv.org/abs/2104.01778)中提出的。音频光谱图Transformer将Vision Transformer应用于音频，通过将音频转换成图像（光谱图）进行处理。该模型在音频分类方面取得了最先进的结果。

摘要如下：

*在过去的十年中，卷积神经网络（CNN）广泛被用作端到端音频分类模型的主要构建块，其目标是学习将音频光谱图直接映射到相应标签。为了更好地捕捉长距离的全局上下文，最近的一个趋势是在CNN之上添加自注意机制，形成CNN-attention混合模型。然而，目前尚不清楚是否有必要依赖CNN，并且基于注意力的纯粹神经网络是否足以在音频分类中获得良好的性能。在本文中，我们通过引入音频光谱图Transformer（AST）来回答这个问题，这是一种基于纯注意力的无卷积音频分类模型。我们在各种音频分类基准测试上评估AST，在AudioSet上实现了0.485 mAP的新的最先进结果，在ESC-50上实现了95.6%的准确率，在Speech Commands V2上实现了98.1%的准确率。*

提示：

- 当在自己的数据集上对音频光谱图Transformer（AST）进行微调时，建议注意输入归一化（确保输入的均值为0，标准差为0.5）。`ASTFeatureExtractor`负责此任务。请注意，它默认使用AudioSet的均值和标准差。你可以查看[`ast/src/get_norm_stats.py`](https://github.com/YuanGongND/ast/blob/master/src/get_norm_stats.py)以了解作者如何计算下游数据集的统计信息。
- 请注意，AST需要较低的学习率（作者在[PSLA论文](https://arxiv.org/abs/2102.01243)中使用了比他们的CNN模型小10倍的学习率），并且收敛速度较快，因此请为你的任务搜索合适的学习率和学习率调度器。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/audio_spectogram_transformer_architecture.png"
alt="drawing" width="600"/>

<small>音频光谱图Transformer架构。来自<a href="https://arxiv.org/abs/2104.01778">原始论文</a>。</small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[此处](https://github.com/YuanGongND/ast)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）资源列表，可帮助你开始使用音频光谱图Transformer。

<PipelineTag pipeline="audio-classification"/>

- 可在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/AST)找到使用AST进行音频分类的推理笔记本。
- [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb)支持对`ASTForAudioClassification`进行操作。
- 另请参阅：[音频分类](../tasks/audio_classification)。

如果你有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将对其进行审查！该资源理想情况下应该展示出一些新的东西，而不是重复现有的资源。

## ASTConfig

[[autodoc]] ASTConfig

## ASTFeatureExtractor

[[autodoc]] ASTFeatureExtractor
    - __call__

## ASTModel

[[autodoc]] ASTModel
    - forward

## ASTForAudioClassification

[[autodoc]] ASTForAudioClassification
    - forward