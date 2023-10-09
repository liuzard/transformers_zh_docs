<!--版权所有©2022 HuggingFace团队。

根据Apache许可证第2版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。
您可以在以下位置获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

注意：本文件以Markdown格式编写，但包含了我们doc-builder的特定语法（类似于MDX），可能在您的Markdown查看器中无法正确呈现。

-->

# FocalNet

## 概述

FocalNet模型在Jianwei Yang、Chunyuan Li、Xiyang Dai、Lu Yuan和Jianfeng Gao的论文[Focal Modulation Networks](https://arxiv.org/abs/2203.11926)中提出。
FocalNet通过一种聚焦调制机制完全替换了自注意力（在模型如[ViT](vit)和[Swin](swin)中使用）来对视觉中的标记交互建模。
作者声称，相对于基于自注意力的模型，FocalNet在图像分类、目标检测和分割任务上具有相似的计算成本，但性能更好。

论文摘要如下：

*我们提出了一种称为聚焦调制网络（FocalNets）的方法，其中自注意力（SA）完全被用于视觉中建立标记交互的聚焦调制机制取代。聚焦调制由三个组成部分组成：（i）层次化背景建模，使用一系列深度可分离卷积层来编码从短范围到长范围的视觉背景；（ii）门控聚合，根据查询标记的内容选择性地收集背景；（iii）逐元素调制或仿射变换，将聚集的背景注入查询。大量实验证明，FocalNets在图像分类、目标检测和分割等任务上超越了最先进的自注意力对应物（例如Swin和FocalTransformers），且其计算成本相似。具体而言，FocalNet在尺寸为tiny和base时分别在ImageNet-1K上实现了82.3%和83.9%的top-1准确率。在以224分辨率对ImageNet-22K进行预训练后，分别在分辨率为224和384的微调上达到了86.5%和87.3%的top-1准确率。在转移到下游任务时，FocalNets表现出明显的优势。对于使用Mask R-CNN的对象检测，使用1\times训练的FocalNet base优于Swin对应物2.1个点，并且已经超过使用3\times计划训练的Swin（49.0比48.5）。对于使用UPerNet的语义分割，单尺度的FocalNet base优于Swin 2.4，多尺度上也超过Swin（50.5比49.7）。使用大型FocalNet和Mask2former，在ADE20K语义分割上实现了58.5的mIoU，并在COCO全景分割上实现了57.9的PQ。使用超大型FocalNet和DINO，在COCO minival和test-dev上分别实现了64.3和64.4的mAP，建立在比Swinv2-G和BEIT-3等更大的基于注意力的模型之上。*

提示：

- 您可以使用 [`AutoImageProcessor`] 类为模型准备图像。

此模型由[nielsr](https://huggingface.co/nielsr)提供。
原始代码可在[此处](https://github.com/microsoft/FocalNet)找到。

## FocalNetConfig

[[autodoc]] FocalNetConfig

## FocalNetModel

[[autodoc]] FocalNetModel
    - forward

## FocalNetForMaskedImageModeling

[[autodoc]] FocalNetForMaskedImageModeling
    - forward

## FocalNetForImageClassification

[[autodoc]] FocalNetForImageClassification
    - forward