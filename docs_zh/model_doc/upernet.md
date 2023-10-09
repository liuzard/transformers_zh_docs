<!--
版权所有 © 2022 HuggingFace 团队。保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”），您只有在符合许可证的情况下才能使用此文件。您可以在下方链接中获取许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本许可证下的软件是基于“按现状”分发的，不附带任何明示或暗示的担保或条件。详细了解许可证中规定的特定权限和限制。

⚠️ 请注意，此文件是 Markdown 格式，但包含针对我们文档构建器（类似于 MDX）的特殊语法，可能无法在您的 Markdown 查看器中正确显示。

-->

# UPerNet

## 概述

UPerNet 模型是由 Tete Xiao、Yingcheng Liu、Bolei Zhou、Yuning Jiang、Jian Sun 在[《统一感知解析用于场景理解》](https://arxiv.org/abs/1807.10221)一文中提出的。UPerNet 是一个通用框架，可以有效地从图像中分割广泛的概念，利用像[ConvNeXt](convnext)或[Swin](swin)这样的视觉骨干。

论文中的摘要如下：

*人类在多个层次上识别视觉世界：我们轻松地对场景进行分类并检测其中的对象，同时还能识别对象的纹理和表面以及它们的不同组成部分。在本文中，我们研究了一个称为统一感知解析的新任务，它要求机器视觉系统能够从给定的图像中识别尽可能多的视觉概念。开发了一种名为 UPerNet 的多任务框架和训练策略，以从异构图像注释中进行学习。我们在统一感知解析上对我们的框架进行了基准测试，并展示了它能够从图像中有效地分割广泛的概念。训练好的网络进一步应用于在自然场景中发现视觉知识。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/upernet_architecture.jpg"
alt="drawing" width="600"/>

<small> UPerNet 框架。摘自<a href="https://arxiv.org/abs/1807.10221">原论文</a>。 </small>

这个模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码基于 OpenMMLab 的 mmsegmentation，在[此处](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/uper_head.py)可找到。

## 资源

以下是官方 Hugging Face 和社区（由 🌎 表示）资源列表，可帮助您开始使用 UPerNet。

- 可在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/UPerNet)找到 UPerNet 的演示笔记本。
- [`UperNetForSemanticSegmentation`] 可在这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb)中使用。
- 另请参阅：[语义分割任务指南](../tasks/semantic_segmentation)

如果您有兴趣提交要包含在此处的资源，请随时打开拉取请求，我们将进行审查！该资源理想情况下应该展示一些新东西，而不是重复现有资源。

## 使用

UPerNet 是一个用于语义分割的通用框架。它可以与任何视觉骨干一起使用，如下所示：

```py
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)
```

要使用其他视觉骨干，比如 [ConvNeXt](convnext)，只需使用相应的骨干实例化模型即可：

```py
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation

backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)
```

请注意，这将随机初始化模型的所有权重。

## UperNetConfig

[[autodoc]] UperNetConfig

## UperNetForSemanticSegmentation

[[autodoc]] UperNetForSemanticSegmentation
    - forward