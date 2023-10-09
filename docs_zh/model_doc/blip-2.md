<!--版权所有2023年HuggingFace Team。保留所有权利。

根据Apache许可证第2版（“许可证”）获得许可; 在符合许可证的条件下，您不得使用本文件。您可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则在许可证下分发的软件是基于“按原样” BASIS进行分发的，不附带任何形式的担保或条件，无论是明示的还是暗示的。有关许可证的特定语言，请参阅许可证。

⚠️请注意，该文件是Markdown格式，但包含我们的文档构建器（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正确显示。-->

# BLIP-2

## 概述

BLIP-2模型是由Junnan Li、Dongxu Li、Silvio Savarese和Steven Hoi在[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)中提出的。BLIP-2利用冻结的预训练图像编码器和大型语言模型（LLM），通过在它们之间训练一个轻量级的12层Transformer编码器，实现了各种视觉语言任务的最新性能。特别值得注意的是，BLIP-2在零样本VQAv2上改善了[Flamingo](https://arxiv.org/abs/2204.14198)（一个800亿参数模型）8.7%，且可训练参数减少了54倍。

论文中的摘要如下：

*由于大型模型的端到端训练，视觉与语言预训练的成本变得越来越高。本文提出了BLIP-2，一种通用且高效的预训练策略，从现成的冻结预训练图像编码器和冻结大型语言模型中引导视觉语言预训练。BLIP-2通过轻量级的查询变换器桥接了模态差距，该变换器分为两个阶段进行预训练。第一阶段从冻结图像编码器中引导视觉语言表示学习。第二阶段从冻结语言模型中引导视觉到语言的生成学习。尽管BLIP-2的可训练参数明显少于现有方法，但在各种视觉语言任务中，其性能达到了最新水平。例如，我们的模型在零样本VQAv2上的性能超过了Flamingo80B，准确率提高了8.7%，而可训练参数减少了54倍。我们还展示了该模型零样本图像到文本生成的新能力，该能力可以遵循自然语言指令。*

提示：

- BLIP-2可以用于给定图像和可选择的文本提示的条件文本生成。在推理时，建议使用[`generate`]方法。
- 可以使用[`Blip2Processor`]来准备模型的图像，并将预测的标记ID解码成文本。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/blip2_architecture.jpg"
alt="drawing" width="600"/> 

<small> BLIP-2模型架构。源自<a href="https://arxiv.org/abs/2301.12597">原始论文。</a> </small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[此处](https://github.com/salesforce/LAVIS/tree/5ee63d688ba4cebff63acee04adaef2dee9af207)找到。

## 资源

以下是官方Hugging Face和社区（用🌎表示）资源列表，可帮助您开始使用BLIP-2模型。

- 可在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/BLIP-2)找到有关BLIP-2用于图像字幕生成、视觉问答（VQA）和类似对话的演示笔记本。

如果您有兴趣提交资源以包含在此处，请随时提出拉取请求，我们将进行审核！资源 ideally 应该展示一些新内容，而不是重复现有资源。

## Blip2Config

[[autodoc]] Blip2Config
    - from_vision_qformer_text_configs

## Blip2VisionConfig

[[autodoc]] Blip2VisionConfig

## Blip2QFormerConfig

[[autodoc]] Blip2QFormerConfig

## Blip2Processor

[[autodoc]] Blip2Processor

## Blip2VisionModel

[[autodoc]] Blip2VisionModel
    - forward

## Blip2QFormerModel

[[autodoc]] Blip2QFormerModel
    - forward

## Blip2Model

[[autodoc]] Blip2Model
    - forward
    - get_text_features
    - get_image_features
    - get_qformer_features

## Blip2ForConditionalGeneration

[[autodoc]] Blip2ForConditionalGeneration
    - forward
    - generate