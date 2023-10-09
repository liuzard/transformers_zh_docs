<!--版权所有2022年抱抱龙团队。保留所有权利。

根据Apache许可证，第2版（“许可证”），你不得使用此文件，除非符合许可证的规定。
可以在下面的网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的，
不附带任何明示或隐含的担保条款。有关许可证的具体语言和限制，请参阅许可证。

⚠️注意，这个文件是Markdown格式的，但包含特定的语法，
是我们的文档构建器（类似于MDX）可以正确渲染的。-->

# MaskFormer

<Tip>

这是最近推出的模型，因此API尚未进行过广泛测试。未来可能会修复一些错误或稍微更改，以确保其正常运行。如果你看到了奇怪的情况，请[提交Github问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

</Tip>

## 概述

MaskFormer模型是由Bowen Cheng、Alexander G. Schwing和Alexander Kirillov在论文《Per-Pixel Classification is Not All You Need for Semantic Segmentation》中提出的。MaskFormer通过使用蒙版分类范式来解决语义分割问题，而不是进行传统的像素级分类。

论文摘要如下：

*现代方法通常将语义分割定为逐像素分类任务，而实例级分割则使用备用蒙版分类处理。我们的关键观点是：蒙版分类对于以统一的方式解决语义分割和全景分割任务是足够通用的，使用相同的模型、损失和训练过程。在这个观察的基础上，我们提出了MaskFormer，一个简单的蒙版分类模型，它预测一组二进制蒙版，每个蒙版与一个单一的全局类别标签预测相关联。总的来说，我们提出的基于蒙版分类的方法简化了解决语义分割和全景分割任务的有效方法的面貌，并展现了出色的实证结果。特别地，我们观察到在类别数目较大时，MaskFormer优于逐像素分类基线模型。我们的基于蒙版分类的方法优于当前最先进的语义分割（ADE20K上的55.6 mIoU）和全景分割（COCO上的52.7 PQ）模型。*

提示：
- MaskFormer的Transformer解码器与[DETR](detr)的解码器相同。在训练过程中，DETR的作者发现在解码器中使用辅助损失对于帮助模型输出每个类别的正确目标数量很有帮助。如果将[`MaskFormerConfig`]的参数`use_auxilary_loss`设置为`True`，则在每个解码器层之后将添加预测前馈神经网络和匈牙利损失（FFNs共享参数）。
- 如果要在多个节点的分布式环境中训练模型，则应该更新`modeling_maskformer.py`中的`MaskFormerLoss`类的`get_num_masks`函数。在多节点训练时，这应该设置为所有节点的目标蒙版数量的平均值，可以在原始实现中看到[这里](https://github.com/facebookresearch/MaskFormer/blob/da3e60d85fdeedcb31476b5edd7d328826ce56cc/mask_former/modeling/criterion.py#L169)。
- 可以使用[`MaskFormerImageProcessor`]来为模型准备图像和可选的目标。
- 要获得最终的分割结果，可以根据任务调用[`~MaskFormerImageProcessor.post_process_semantic_segmentation`]或[`~MaskFormerImageProcessor.post_process_panoptic_segmentation`]。这两个任务都可以使用[`MaskFormerForInstanceSegmentation`]输出来解决，全景分割可以接受一个可选的`label_ids_to_fuse`参数，以将目标对象（例如天空）的实例合并在一起。

以下图示了MaskFormer的架构。摘自[原始论文](https://arxiv.org/abs/2107.06278)。

<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/maskformer_architecture.png"/>

该模型由[francesco](https://huggingface.co/francesco)贡献。原始代码可以在[这里](https://github.com/facebookresearch/MaskFormer)找到。

## 资源

<PipelineTag pipeline="image-segmentation"/>

- 可在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MaskFormer)找到演示使用MaskFormer进行推理以及在自定义数据上进行微调的所有笔记本。

## MaskFormer特定输出

[[autodoc]] models.maskformer.modeling_maskformer.MaskFormerModelOutput

[[autodoc]] models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput

## MaskFormerConfig

[[autodoc]] MaskFormerConfig

## MaskFormerImageProcessor

[[autodoc]] MaskFormerImageProcessor
    - preprocess
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## MaskFormerFeatureExtractor

[[autodoc]] MaskFormerFeatureExtractor
    - __call__
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## MaskFormerModel

[[autodoc]] MaskFormerModel
    - forward

## MaskFormerForInstanceSegmentation

[[autodoc]] MaskFormerForInstanceSegmentation
    - forward