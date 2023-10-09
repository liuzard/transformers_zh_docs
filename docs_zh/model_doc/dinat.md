<!--版权所有 2022 年 The HuggingFace 团队保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”），除非符合许可证，否则你不得使用此文件。
你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，
没有任何明示或暗示的担保或条件。有关许可证的特定语言和限制，请参阅许可证。

⚠️ 请注意，这个文件是用 Markdown 编写的，但包含了我们文档构建器（类似于 MDX）的特殊语法，
这可能在你的 Markdown 查看器中无法正确显示。

-->

# Dilated Neighborhood Attention Transformer

## 概述

DiNAT 是由 Ali Hassani 和 Humphrey Shi 在文章《Dilated Neighborhood Attention Transformer》（[Dilated Neighborhood Attention Transformer](https://arxiv.org/abs/2209.15001)）中提出的。

它通过向 [NAT](nat) 添加了一种 Dilated Neighborhood Attention 模式来捕捉全局上下文，并在性能上显示出显著的改进。

论文中的摘要如下：

*Transformer 快速成为跨模态、领域和任务中应用最广泛的深度学习架构之一。在视觉领域，除了持续努力改进的 plain transformer 之外，
层级 transformer 也因其性能和容易集成到现有框架中而受到了重视。这些模型通常采用局部注意力机制，例如滑动窗口 Neighborhood Attention（NA）或
Swin Transformer 的 Shifted Window Self Attention。尽管这些方式有效地降低了自注意力的二次复杂度，但是局部注意力削弱了自注意力的两个最有吸引力的特性：
长距离相互依赖建模和全局感受野。在本文中，我们引入拓展的 Dilated Neighborhood Attention（DiNA），
这是一种自然、灵活且高效的 NA 扩展，可以在不增加额外成本的情况下捕捉更多的全局上下文并指数级地扩展感受野。
NA 的局部注意力和 DiNA 的稀疏全局注意力相辅相成，因此我们提出了一种新的层级视觉 Transformer，即 Dilated Neighborhood Attention Transformer（DiNAT）。
与 NAT、Swin 和 ConvNeXt 等强基线相比，DiNAT 的变种在各项指标上都有显著的改进。
我们的大模型在 COCO 目标检测中的 box AP 上比 Swin 提高了 1.5%，在 COCO 实例分割中的 mask AP 上比 Swin 提高了 1.3%，
在 ADE20K 语义分割中的 mIoU 上比 Swin 提高了 1.1%。
搭配新的框架，我们的大模型是 COCO（58.2 PQ）和 ADE20K（48.5 PQ）全景分割模型的最新技术水平，
也是 Cityscapes（44.5 AP）和 ADE20K（35.4 AP）实例分割模型的最新技术水平（不需要额外的数据）。
它还与最新技术水平专门的 ADE20K 语义分割模型相匹配（58.2 mIoU），
并在 Cityscapes 上排名第二（84.5 mIoU）（不需要额外的数据）。*

提示：
- 你可以使用 [`AutoImageProcessor`] API 来为模型准备图像。
- DiNAT 可以用作*主干模型*。当设置 `output_hidden_states = True` 时，
它将输出 `hidden_states` 和 `reshaped_hidden_states`。`reshaped_hidden_states` 的形状为 `(batch, num_channels, height, width)`，
而不是 `(batch_size, height, width, num_channels)`。

注意：
- DiNAT 依赖于 [NATTEN](https://github.com/SHI-Labs/NATTEN/) 的 Neighborhood Attention 和 Dilated Neighborhood Attention 实现。
你可以通过参考 [shi-labs.com/natten](https://shi-labs.com/natten) 安装适用于 Linux 的预编译库，或通过运行 `pip install natten` 在你的系统上构建。
请注意，后者可能需要一些时间进行编译。NATTEN 尚不支持 Windows 设备。
- 目前仅支持 4 的补丁大小。

<img
src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dilated-neighborhood-attention-pattern.jpg"
alt="drawing" width="600"/>

<small> 具有不同扩张值的 Neighborhood Attention。
来自<a href="https://arxiv.org/abs/2209.15001">原始论文</a>。</small>

这个模型由 [Ali Hassani](https://huggingface.co/alihassanijr) 贡献。
可以在[这里](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)找到原始代码。

## 资源

Hugging Face 官方和社区资源列表（🌎 表示社区资源），以帮助你开始使用 DiNAT。

<PipelineTag pipeline="image-classification"/>

- 使用此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持 [`DinatForImageClassification`]。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果你有兴趣提交资源供收录，欢迎提出拉取请求，我们将会审核！资源最好展示的是一些新的东西，而不是重复的现有资源。

## DinatConfig

[[autodoc]] DinatConfig

## DinatModel

[[autodoc]] DinatModel
    - forward

## DinatForImageClassification

[[autodoc]] DinatForImageClassification
    - forward