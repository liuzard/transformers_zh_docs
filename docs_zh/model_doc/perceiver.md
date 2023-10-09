<!--版权 2021 HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证的规定，
否则，您不能使用此文件。您可以在以下位置获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则在许可证下分发的软件是基
于“按现状提供”的基础上的，没有任何明示或暗示的保证或条件。请参阅
许可证以了解许可证下的具体语言和限制。

⚠ 注意，此文件为Markdown格式，但包含特定语法以便于doc-builder（类似于MDX），
可能在Markdown查看器中无法正确渲染。

-->

# Perceiver

## 概述

Perceiver IO模型由Andrew Jaegle、Sebastian Borgeaud、Jean-Baptiste Alayrac、Carl Doersch、Catalin Ionescu、
David Ding、Skanda Koppula、Daniel Zoran、Andrew Brock、Evan Shelhamer、Olivier Hénaff、Matthew M.
Botvinick、Andrew Zisserman、Oriol Vinyals和João Carreira在《Perceiver IO: A General Architecture for Structured Inputs &
Outputs》一文中提出。

Perceiver IO是Perceiver的一般化扩展，用于处理任意类型的输出，除了任意类型的输入。原始的Perceiver只能生成单个分类标签。
除了分类标签，Perceiver IO还可以生成（例如）语言、光流和具有音频的多模态视频。这是使用与原始Perceiver相同的构建块完成的。
Perceiver IO的计算复杂性与输入和输出的大小呈线性关系，大部分处理发生在潜在空间中，使我们能够处理比标准Transformer处理能力更大的输入和输出。
这意味着，例如，Perceiver IO可以直接使用字节而不是分词输入来进行BERT风格的遮蔽语言建模。

该论文的摘要如下：

*最近提出的Perceiver模型在多个领域（图像、音频、多模态、点云）上取得了良好的结果，而且计算和内存要求随输入的大小呈线性增长。
虽然Perceiver支持多种输入类型，但它只能生成非常简单的输出，如类别分数。Perceiver IO通过学习灵活查询模型的潜在空间以生成任意大小和语义的输出，
而不牺牲原始模型的吸引特性来克服了这种限制。Perceiver IO仍然将模型深度与数据大小分离，并且仍然与数据大小呈线性缩放，但现在相对于输入和输出大小。
完整的Perceiver IO模型在高度结构化的输出空间任务上取得了强大的结果，例如自然语言和视觉理解、StarCraft II以及多任务和多模态领域。
特别值得一提的是，不需要输入分词化，Perceiver IO与基于Transformer的BERT基准在GLUE语言基准上匹配，并在Sintel光流估计
方面实现了最先进的性能。*

下面是Perceiver工作原理的TLDR解释：

Transformer的自注意机制的主要问题是其时间和内存需求与序列长度呈二次比例关系。因此，像BERT和RoBERTa这样的模型在最大序列长度为512的限制下。
Perceiver试图通过在输入上执行自注意力操作的代替方式，即在一组潜在变量上执行自注意力操作，并且仅将输入用于交叉注意力。这样，时间和内存需求不再取决于
输入的长度，因为使用了固定数量的潜在变量，如256或512。这些变量在随机初始化后，通过反向传播进行端到端训练。

在内部，[PerceiverModel]将创建潜在空间，其形状为`(batch_size，num_latents，d_latents)`的张量。必须向模型提供输入（可以是文本、图像、音频等），
用于与潜在空间执行交叉关注。Perceiver编码器的输出是相同形状的张量。然后，类似于BERT，可以通过沿着序列维度进行平均并在其上放置线性层，
将潜在变量的最后隐藏状态转换为分类logit，从而将`d_latents`投影为`num_labels`。

这就是原始Perceiver论文的观点。然而，它只能输出分类logits。在后续的PerceiverIO工作中，他们将其推广为模型也能够产生任意大小的输出。
您可能会问，如何做到的？实际上，这个想法相对简单：首先定义一个任意大小的输出，然后使用潜在变量的最后隐藏状态作为键和值，
使用输出作为查询执行交叉注意力。

因此，假设想要使用Perceiver执行遮蔽语言建模（BERT风格）。由于Perceiver的输入长度不会对自注意力层的计算时间产生影响，
因此可以提供原始字节作为输入，为模型提供长度为2048的`inputs`。如果现在屏蔽了这2048个令牌中的某些令牌，
可以将`outputs`定义为`shape：(batch_size，2048，768)`。接下来，使用潜在变量的最后隐藏状态执行交叉注意力以更新`outputs`张
量。在交叉注意力之后，仍然有一个`shape：(batch_size，2048，768)`的张量。然后可以在其上放置一个常规语言建模头，将最后一个维度投影到模型的
词汇大小，即创建`shape：(batch_size，2048，262)`的logits（因为Perceiver使用262个字节ID的词汇大小）。

<small>Perceiver IO架构。来自<a href="https://arxiv.org/abs/2105.15203">原始论文</a></small>

此模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可在[此处](https://github.com/deepmind/deepmind-research/tree/master/perceiver)找到。

提示：

- 上手Perceiver的最快方法是查看[教程笔记本](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Perceiver)。
- 如果想要全面了解模型的工作原理和在库中的实现，请参考[博客文章](https://huggingface.co/blog/perceiver)。
请注意，库中提供的模型仅展示了Perceiver的一些例子。还有许多其他用途，包括问题回答、命名实体识别、对象检测、音频分类、视频分类等。

**注意**：

- 由于PyTorch中的一个错误，Perceiver与`torch.nn.DataParallel`无法正常工作，参见[问题＃36035](https://github.com/pytorch/pytorch/issues/36035)。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [遮蔽语言建模任务指南](../tasks/masked_language_modeling)
- [图像分类任务指南](../tasks/image_classification)

## Perceiver特定输出

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverModelOutput

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverDecoderOutput

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMaskedLMOutput

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverClassifierOutput

## PerceiverConfig

[[autodoc]] PerceiverConfig

## PerceiverTokenizer

[[autodoc]] PerceiverTokenizer
    - __call__

## PerceiverFeatureExtractor

[[autodoc]] PerceiverFeatureExtractor
    - __call__

## PerceiverImageProcessor

[[autodoc]] PerceiverImageProcessor
    - preprocess

## PerceiverTextPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverTextPreprocessor

## PerceiverImagePreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverImagePreprocessor

## PerceiverOneHotPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverOneHotPreprocessor

## PerceiverAudioPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverAudioPreprocessor

## PerceiverMultimodalPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor

## PerceiverProjectionDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverProjectionDecoder

## PerceiverBasicDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverBasicDecoder

## PerceiverClassificationDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverClassificationDecoder

## PerceiverOpticalFlowDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverOpticalFlowDecoder

## PerceiverBasicVideoAutoencodingDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverBasicVideoAutoencodingDecoder

## PerceiverMultimodalDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder

## PerceiverProjectionPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverProjectionPostprocessor

## PerceiverAudioPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverAudioPostprocessor

## PerceiverClassificationPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverClassificationPostprocessor

## PerceiverMultimodalPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMultimodalPostprocessor

## PerceiverModel

[[autodoc]] PerceiverModel
    - forward

## PerceiverForMaskedLM

[[autodoc]] PerceiverForMaskedLM
    - forward

## PerceiverForSequenceClassification

[[autodoc]] PerceiverForSequenceClassification
    - forward

## PerceiverForImageClassificationLearned

[[autodoc]] PerceiverForImageClassificationLearned
    - forward

## PerceiverForImageClassificationFourier

[[autodoc]] PerceiverForImageClassificationFourier
    - forward

## PerceiverForImageClassificationConvProcessing

[[autodoc]] PerceiverForImageClassificationConvProcessing
    - forward

## PerceiverForOpticalFlow

[[autodoc]] PerceiverForOpticalFlow
    - forward

## PerceiverForMultimodalAutoencoding

[[autodoc]] PerceiverForMultimodalAutoencoding
    - forward
