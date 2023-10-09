<!--版权所有 2022 The HuggingFace Team. 保留所有权利。

根据 Apache 2.0 许可证 (the "License")，除非符合许可证，否则你不能使用此文件。
你可以获取许可证的副本，网址为

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，按“现状”分发的软件，没有任何形式的明示或暗示担保或条件，参见许可证下的特定语言

⚠️ 注意，此文件是 Markdown 文件，但包含我们的文档构建器 (类似于MDX) 的特定语法，可能无法在你的 Markdown 查看器中正确显示。

-->

# DETA

## 概述

DETA模型是由Jeffrey Ouyang-Zhang、Jang Hyun Cho、Xingyi Zhou和Philipp Krähenbühl在[NMS Strikes Back](https://arxiv.org/abs/2212.06137)中提出的。
DETA（Detection Transformers with Assignment的缩写）通过将传统检测器中的一对一二分匹配损失替换为一对多标签分配（使用非最大抑制，NMS），改进了[Deformable DETR](deformable_detr)。这导致AP提高了最多2.5个点。

论文摘要如下：

*Detection Transformer（DETR）在训练过程中使用一对一二分匹配直接将查询转换为唯一对象，实现了端到端的目标检测。最近，这些模型在COCO上的性能已超越传统检测器而具有不可否认的优势。然而，它们与传统检测器在多个设计方面有所不同，包括模型架构和训练调度，因此一对一匹配的有效性尚未完全理解。在这项工作中，我们对DETR中的一对一匈牙利匹配和传统检测器中使用非最大抑制（NMS）的一对多标签分配进行了严格比较。令人惊讶的是，我们观察到在相同设置下，使用NMS的一对多分配一贯优于标准的一对一匹配，AP提高了多达2.5个点。我们的检测器在传统IoU-based标签分配的基础上，使用Deformable-DETR进行训练，在ResNet50骨干网络上在12个epoch（1x调度）内实现了50.2的COCO mAP，超越了所有现有传统或基于Transformer的检测器。在多个数据集、调度和架构上，我们始终表明，对于性能良好的检测变压器，二分匹配是没有必要的。此外，我们将检测Transformer的成功归因于其表达能力强大的Transformer架构。*

提示：

- 你可以使用[`DetaImageProcessor`]来为模型准备图像和可选的目标。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/deta_architecture.jpg"
alt="drawing" width="600"/>

<small> DETA概述。图片来源于<a href="https://arxiv.org/abs/2212.06137">原论文</a>。</small>

此模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[此处](https://github.com/jozhang97/DETA)找到。

## 资源

以下是官方Hugging Face和社区（使用🌎表示）提供的资源列表，可帮助你开始使用DETA。

- DETA的演示笔记本可在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETA)找到。
- 请参阅：[目标检测任务指南](../tasks/object_detection)

如果你有兴趣提交资源以包含在此处，请随时提交拉取请求，我们将进行审核！该资源应该展示出新的东西，而不是重复现有的资源。

## DetaConfig

[[autodoc]] DetaConfig

## DetaImageProcessor

[[autodoc]] DetaImageProcessor
    - preprocess
    - post_process_object_detection

## DetaModel

[[autodoc]] DetaModel
    - forward

## DetaForObjectDetection

[[autodoc]] DetaForObjectDetection
    - forward