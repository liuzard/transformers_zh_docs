<!--2021年版权归HuggingFace团队所有。

根据Apache许可证第2.0版（“许可证”）授权; 除非符合许可证要求，
否则不得使用此文件。你可以在以下位置获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件基础上，
无论是明示还是暗示的都是无任何担保或条件。请参阅许可证
以及许可证下的特定语言的权限和限制条款。

⚠️请注意，此文件以Markdown格式编写，但包含我们doc-builder专用的特定语法（类似于MDX），
可能在Markdown查看器中不能正确渲染。-->

# ViLT

## 概述

ViLT模型在[ViLT:Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)中
被Wonjae Kim，Bokyung Son，Ildoo Kim提出。ViLT将文本嵌入集成到Vision Transformer（ViT）中，使其具有最小设计 
用于视觉和语言预训练（VLP）。

论文中的摘要如下所示：

“视觉与语言预训练（VLP）在各种联合视觉与语言的下游任务中提高了性能。
当前的VLP方法在很大程度上依赖于图像特征提取过程，其中大部分涉及区域监督（如，对象检测）和卷积架构（如，ResNet）。
尽管在文献中被忽略，但我们发现这在效率/速度（即，仅提取输入特征所需的计算量远高于多模态交互步骤）和表达能力方面存在问题，
因为它的上界受到可视嵌入器及其预定义的视觉词汇表的表达能力的限制。在本文中，我们提出了一个最小的VLP模型，
Vision-and-Language Transformer（ViLT），在视觉输入的处理方面采用了与处理文本输入相同的无卷积方式。
我们展示了ViLT比以前的VLP模型快上数十倍，但下游任务的性能相比竞争对手或更好。”

提示：

- 开始使用ViLT的最快方法是查看 [示例笔记本](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ViLT)
（展示了自定义数据的推理和微调）。
- ViLT是一个同时接受`pixel_values`和`input_ids`作为输入的模型。可以使用[`ViltProcessor`] 来为模型准备数据。
  这个处理器将图像处理器（用于图像模态）和分词工具（用于语言模态）包装在一个处理器中。
- ViLT以各种尺寸的图像进行训练：作者将输入图像的较短边缩放到384，并将较长边限制在640以下，同时保持纵横比。
  为了实现图像的批处理，作者使用一个`pixel_mask`来指示哪些像素值是真实的，哪些是填充的。[`ViltProcessor`]会为你自动生成这个值。
- ViLT的设计与标准的Vision Transformer（ViT）非常相似。唯一的区别是模型为语言模态包括了额外的嵌入层。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vilt_architecture.jpg"
alt="drawing" width="600"/>

<small> ViLT架构。来源于[原始论文](https://arxiv.org/abs/2102.03334)。 </small>

这个模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可在[此处](https://github.com/dandelin/ViLT)找到。

提示：

- 此模型的PyTorch版本仅适用于torch 1.10及更高版本。

## ViltConfig

[[autodoc]]ViltConfig

## ViltFeatureExtractor

[[autodoc]]ViltFeatureExtractor
    - __call__

## ViltImageProcessor

[[autodoc]]ViltImageProcessor
    - preprocess

## ViltProcessor

[[autodoc]]ViltProcessor
    - __call__

## ViltModel

[[autodoc]]ViltModel
    - forward

## ViltForMaskedLM

[[autodoc]]ViltForMaskedLM
    - forward

## ViltForQuestionAnswering

[[autodoc]]ViltForQuestionAnswering
    - forward

## ViltForImagesAndTextClassification

[[autodoc]]ViltForImagesAndTextClassification
    - forward

## ViltForImageAndTextRetrieval

[[autodoc]]ViltForImageAndTextRetrieval
    - forward

## ViltForTokenClassification

[[autodoc]]ViltForTokenClassification
    - forward