<!--版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），除非符合许可证的规定，
否则您不能使用此文件。您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于
“按原样”基础分发的，不附带任何明示或暗示的保证或条件。有关许可证中的
特定语言的权限和限制，请参阅许可证。

⚠️请注意，此文件是Markdown格式的，但包含特定于我们的文档构建器（类似于MDX）的语法，可能无法在Markdown查看器中正确渲染。

-->

# GIT

## 概览

GIT模型是在《GIT：用于视觉和语言的生成图像到文本的变换器》（[GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100)）一文中提出的。
该论文的作者是Jianfeng Wang、Zhengyuan Yang、Xiaowei Hu、Linjie Li、Kevin Lin、Zhe Gan、Zicheng Liu和Ce Liu。GIT是一个仅有解码器的Transformer模型，
它利用了[CLIP](clip)的视觉编码器，除了文本外，还将模型进行了视觉输入的条件设置。该模型在图像字幕和视觉问答基准测试上取得了最先进的结果。

该论文的摘要如下：

*本文中，我们设计并训练了一个生成式图像到文本的变换器模型，即GIT模型，用于统一图像/视频字幕和问题回答等视觉-语言任务。虽然生成模型提供了一种在预训练和微调之间保持一致的网络架构，但现有工作通常包含了复杂的结构（单/多模态编码器/解码器），
并依赖于外部模块，例如目标检测器/标签器和光学字符识别（OCR）。在GIT模型中，我们将架构简化为一个图像编码器和一个文本解码器，在一个单一的语言建模任务下完成。我们还扩大了预训练数据和模型大小，以提高模型的性能。在没有花哨的技巧的情况下，
我们的GIT模型在12个具有挑战性的基准上创造了新的最高水平，优势非常大。例如，我们的模型在TextCaps上的性能首次超越了人类（在CIDEr上为138.2 vs. 125.5）。此外，我们还提出了一种新的基于生成的图像分类和场景文本识别方法，在标准基准测试上取得了不错的性能。*

提示：

- GIT的实现方式与GPT-2非常相似，唯一的区别是该模型还受到'pixel_values'的约束。
- 可以使用[`GitProcessor`]来为模型准备图像，并使用`generate`方法进行自回归生成。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/git_architecture.jpg"
alt="drawing" width="600"/>

<small> GIT架构。引自《[original paper](https://arxiv.org/abs/2205.14100)》。</small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[此处](https://github.com/microsoft/GenerativeImage2Text)找到。

## 资源

以下是用于入门GIT的官方Hugging Face和社区资源（用🌎表示）的列表。

- 关于在自定义数据上进行推断和微调GIT的演示笔记本可以在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/GIT)找到。
- 参见：[因果语言建模任务指南](../tasks/language_modeling)

如果您有兴趣提交资源以包含在此处，请随时提交拉取请求，我们会对其进行审查。
该资源应该是新颖的，而不是重复现有的资源。

## GitVisionConfig

[[autodoc]] GitVisionConfig

## GitVisionModel

[[autodoc]] GitVisionModel
    - forward

## GitConfig

[[autodoc]] GitConfig
    - all

## GitProcessor

[[autodoc]] GitProcessor
    - __call__

## GitModel

[[autodoc]] GitModel
    - forward

## GitForCausalLM

[[autodoc]] GitForCausalLM
    - forward