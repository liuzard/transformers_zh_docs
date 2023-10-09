<!--版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证，2.0版（“许可证”）的规定，你不得使用此文件，除非符合许可证的规定。
你可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样”的分发，
没有任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️ 请注意，这个文件是Markdown格式的，但包含了我们文档生成器（类似MDX）的特定语法，可能无法在Markdown查看器中正确显示。-->

# Neighborhood Attention Transformer

## 概述

NAT是由Ali Hassani、Steven Walton、Jiachen Li、Shen Li和Humphrey Shi在[Neighborhood Attention Transformer](https://arxiv.org/abs/2204.07143)中提出的。

它是一种基于Neighborhood Attention的分层视觉Transformer，采用滑动窗口的自注意力模式。

来自论文的摘要如下：

*我们提出了Neighborhood Attention（NA）这是一种用于视觉的高效且可扩展的滑动窗口自注意力机制。NA是一种逐像素的操作，将自注意力（SA）局限于最近的相邻像素，并因此具有与SA二次复杂度相比线性时间和空间复杂度的优势。滑动窗口模式允许NA的感受野增加，而无需额外的像素移位，并且与Swin Transformer的窗口自注意力（WSA）不同，保持了平移等变性。我们开发了NATTEN（Neighborhood Attention Extension），这是一个带有高效C++和CUDA内核的Python包，可以使NA的运行速度比Swin的WSA快40％，同时使用的内存少25％。我们进一步提出了基于NA的新型分层Transformer设计Neighborhood Attention Transformer（NAT），它提升了图像分类和下游视觉性能。NAT的实验结果是有竞争力的；NAT-Tiny在ImageNet上得到了83.2％的top-1准确率，在MS-COCO上得到了51.4％的mAP，在ADE20K上得到了48.4％的mIoU，比Swin模型的准确率分别提高了1.9％，COCO mAP提高了1.0％，ADE20K mIoU提高了2.6％，并且二者具有相似的大小。*

提示：
- 你可以使用[`AutoImageProcessor`] API来准备模型的图像。
- NAT可以用作*骨干*。当`output_hidden_states = True`时，它将输出`hidden_states`和`reshaped_hidden_states`两者。`reshaped_hidden_states`的形状为`(batch, num_channels, height, width)`，而不是`(batch_size, height, width, num_channels)`。

注意：
- NAT依赖于[NATTEN](https://github.com/SHI-Labs/NATTEN/)对Neighborhood Attention的实现。
你可以通过参考[shi-labs.com/natten](https://shi-labs.com/natten)上的预构建的Linux软件包进行安装，
或通过运行`pip install natten`在你的系统上进行构建。后者编译时间可能较长。
NATTEN尚不支持Windows设备。
- 目前仅支持4作为块的大小。

<img
src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/neighborhood-attention-pattern.jpg"
alt="drawing" width="600"/>

<small> Neighborhood Attention与其他注意力模式进行比较。
来源：<a href="https://arxiv.org/abs/2204.07143">原文链接</a>。</small>

此模型由[Ali Hassani](https://huggingface.co/alihassanijr)提供。
原始代码可以在[这里](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)找到。

## 资源

以下是官方Hugging Face和社区（🌎）资源列表，可帮助你开始使用NAT。

<PipelineTag pipeline="image-classification"/>

- 本[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持[`NatForImageClassification`]。
- 另请参阅：[图像分类任务指南（英文）](../tasks/image_classification)

如果你有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将进行审核！资源应该展示出新的内容，而不是重复现有的资源。

## NatConfig

[[autodoc]] NatConfig


## NatModel

[[autodoc]] NatModel
    - forward

## NatForImageClassification

[[autodoc]] NatForImageClassification
    - forward