<!--
版权所有2023年 HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”），除非符合许可证的要求，
否则你不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”的基础
，不附带任何明示或暗示的保证或条件。有关许可证的具体规定
语种，请参见许可证。-->
<!--版权声明-->

# ViTDet

## 概述

ViTDet模型由李阳好、毛涵梓、罗斯·吉尔希克和何恺明在[Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527)提出。
VitDet利用普通的[视觉Transformer](vit)来完成目标检测任务。

来自论文的摘要如下：

*我们探索用于目标检测的普通、非层次化的视觉Transformer（ViT）作为骨干网络。这个设计使得可以对原始ViT架构进行微调，而无需重新设计用于预训练的层次化骨干。通过最小量的微调适应，我们的普通骨干检测器可以取得竞争性的结果。令人惊讶的是，我们观察到：（i）从单尺度特征图中构建简单的特征金字塔（而不使用常见的FPN设计）是足够的，（ii）只用窗口注意（而不使用平移）以及极少量的窗口传播块是足够的。通过以Masked Autoencoders (MAE)为预训练方式预训练的普通ViT骨干，我们的检测器ViTDet可以与以层次化骨干为基础的先前领先方法竞争，在COCO数据集上使用仅ImageNet-1K预训练可达到61.3 AP_box。希望我们的研究能引起对普通骨干检测器的关注。*

提示：

- 目前只提供了骨干网络。

该模型由[nielsr](https://huggingface.co/nielsr)贡献。
可以在[这里](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)找到原始代码。


## VitDetConfig

[[autodoc]] VitDetConfig

## VitDetModel

[[autodoc]] VitDetModel
    - forward
-->