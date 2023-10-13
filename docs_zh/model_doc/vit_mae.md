<!--版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”），你不得使用此文件，除非符合许可证的规定
License。你可以从下面获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于
“按原样”基础，没有任何明示或暗示的保证或条件。请参阅许可证以了解
特定语言下的权限和限制。

⚠️ 请注意，此文件是Markdown格式，但包含特定的语法，用于我们的doc-builder（类似于MDX的语法），在你的Markdown查看器中可能无法
正确呈现。

-->

# ViTMAE

## 概述

ViTMAE模型由Kaiming He、Xinlei Chen、Saining Xie、Yanghao Li、Piotr Dollár和Ross Girshick在[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377v2)一文中提出。该论文表明，通过预训练视觉Transformer（ViT）以重建被掩蔽补丁的像素值，经过微调后，可以获得优于监督预训练的结果。

论文摘要如下：

*本文表明，掩蔽自编码器（MAE）是可扩展的计算机视觉自监督学习器。我们的MAE方法很简单：我们mask输入图像的随机补丁，并重建缺失的像素。它基于两个核心设计。首先，我们开发了一种非对称的编码器-解码器架构，编码器仅在可见补丁的子集（没有掩蔽标记）上运行，而轻量级解码器则从潜在表示和掩蔽标记中重建原始图像。其次，我们发现对输入图像进行高比例的掩蔽，例如75％，可以产生一个复杂且有意义的自监督任务。将这两个设计结合起来可以有效地训练大型模型：我们加速训练（3倍或更多）并提高准确性。我们的可扩展方法允许学习具有良好泛化能力的高容量模型：例如，普通的ViT-Huge模型在仅使用ImageNet-1K数据的方法中获得最佳准确率（87.8％）。下游任务的转移性能优于监督预训练，并呈现出有希望的缩放行为。*

提示：

- MAE（掩蔽自编码）是用于视觉Transformer（ViT）的自监督预训练方法。预训练目标相对简单：通过mask大部分（75％）图像补丁，模型必须重建原始像素值。可以使用[`ViTMAEForPreTraining`]来实现此目的。
- 在预训练之后，将丢弃用于重建像素的解码器，并使用编码器进行微调/线性探测。这意味着在微调之后，可以将权重直接插入[`ViTForImageClassification`]模型中。
- 可以使用[`ViTImageProcessor`]来为模型准备图像。有关更多信息，请参阅代码示例。
- 请注意，MAE的编码器仅用于对视觉补丁的编码。然后，编码的补丁与掩蔽标记连接在一起，解码器（也由Transformer块组成）将其作为输入。每个掩蔽标记都是一个共享的学习向量，表示要预测的缺失补丁的存在。固定的sin/cos位置嵌入被添加到编码器和解码器的输入中。
- 如需了解MAE工作原理的可视化，请参阅此[文章](https://keras.io/examples/vision/masked_image_modeling/)。

<img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png"
alt="drawing" width="600"/> 

<small> MAE架构。来源于《[原始论文](https://arxiv.org/abs/2111.06377)》 </small>

该模型由[nielsr](https://huggingface.co/nielsr)贡献。[sayakpaul](https://github.com/sayakpaul)和[ariG23498](https://github.com/ariG23498)（贡献相同）贡献了模型的TensorFlow版本。原始代码可以在[这里](https://github.com/facebookresearch/mae)找到。

## 资源

以下是Hugging Face官方和社区资源（由🌎表示），可帮助你开始使用ViTMAE。

- 通过[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)，支持[`ViTMAEForPreTraining`]，可以让你从头开始预训练模型/在自定义数据上进一步预训练模型。
- 可在[这里](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb)找到一个演示如何使用[`ViTMAEForPreTraining`]可视化重建的像素值的笔记本。

如果你有兴趣提交资源以包含在此处，请随时打开Pull Request，我们将进行审核！该资源应该理想地展示新内容，而不是重复现有资源。

## ViTMAEConfig

[[autodoc]] ViTMAEConfig


## ViTMAEModel

[[autodoc]] ViTMAEModel
    - forward


## ViTMAEForPreTraining

[[autodoc]] transformers.ViTMAEForPreTraining
    - forward


## TFViTMAEModel

[[autodoc]] TFViTMAEModel
    - call


## TFViTMAEForPreTraining

[[autodoc]] transformers.TFViTMAEForPreTraining
    - call
-->