<!--版权所有2021年HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”）的规定，除非符合许可证的规定，否则不得使用此文件。
您可以在以下位置获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，在许可证下分发的软件是按
“按原样”基础分发的，没有任何形式的明示或暗示的担保或条件。
请参阅许可证以获取权限和限制的特定语言。

⚠️ 请注意，此文件是Markdown格式，但包含特定于我们的doc-builder（类似于MDX）的语法，可能在Markdown查看器中无法正确显示。

-->

# DETR

详细信息

DETR模型是由Nicolas Carion、Francisco Massa、Gabriel Synnaeve、Nicolas Usunier、Alexander Kirillov和Sergey Zagoruyko在《使用Transformer的端到端目标检测》（https://arxiv.org/abs/2005.12872）中提出的。DETR由一个卷积骨干网络和一个编码器-解码器Transformer组成，可用于进行端到端的目标检测训练。它极大地简化了像Faster-R-CNN和Mask-R-CNN这样的模型的复杂性，这些模型使用了区域提议、非最大抑制过程和锚点生成等技术。此外，通过在解码器输出之上简单地添加一个掩码头，DETR还可以自然地扩展到执行全景分割。

论文的摘要如下：

我们提出了一种将目标检测视为直接集合预测问题的新方法。我们的方法简化了检测流程，有效地消除了许多手动设计的组件，例如非最大抑制过程或明确编码我们对任务的先验知识的锚点生成。新框架DEtection TRansformer或DETR的主要组成部分是强制唯一预测的基于集合的全局损失，通过二部图匹配，以及一种transformer编码器-解码器架构。给定一组固定的学习对象查询，DETR通过推理对象之间的关系和全局图像上下文，直接输出最终的预测集。这个新模型在概念上很简单，不需要专门的库，不像许多其他现代检测器那样。DETR在具有挑战性的COCO目标检测数据集上展示了与高度优化的Faster RCNN基线相当的准确性和运行时性能。此外，DETR可以被轻松地推广到以统一的方式产生全景分割。我们证明了它在竞争基线上明显优于竞争基线。

该模型由nielsr贡献（https://huggingface.co/nielsr）。原始代码可以在此处找到（https://github.com/facebookresearch/detr）。

下面是有关[`~transformers.DetrForObjectDetection`]工作原理的TLDR：

首先，将图像发送到预训练的卷积骨干网络（在论文中，作者使用ResNet-50/ResNet-101）。假设我们还添加了一个批处理维度。这意味着骨干网络的输入是一个形状为`(batch_size, 3, height, width)`的张量，假设图像具有3个颜色通道（RGB）。CNN骨干网络输出一个新的较低分辨率的特征图，通常形状为`(batch_size, 2048, height/32, width/32)`。然后，使用`nn.Conv2D`层将其投影到DETR Transformer的隐藏维度，该维度默认为`256`。因此，现在，我们有一个形状为`(batch_size, 256, height/32, width/32)`的张量。接下来，将特征图展平和转置，以获得形状为`(batch_size, seq_len, d_model)` =`(batch_size, width/32*height/32, 256)`的张量。因此，与NLP模型不同，序列长度实际上比通常使用更小的`d_model`（在NLP中通常为768或更高）更长。

接下来，将其发送到编码器，输出形状相同的`encoder_hidden_states`（可以将其视为图像特征）。接下来，所谓的**对象查询**被发送到解码器。这是一个形状为`(batch_size, num_queries, d_model)`的张量，其中`num_queries`通常设置为100，并且用零初始化。这些输入嵌入是位置编码，作者将其称为对象查询，并且与编码器类似，它们被添加到每个关注层的输入中。每个对象查询将在图像中寻找特定对象。解码器通过多个自注意力和编码器-解码器注意力层来更新这些嵌入，并输出相同形状的`decoder_hidden_states`：`(batch_size, num_queries, d_model)`。接下来，为目标检测添加了两个头：一个线性层，用于将每个对象查询分类为对象之一或“没有对象”，以及一个MLP，用于预测每个查询的边界框。

模型使用**二部图匹配损失**进行训练：实际上，我们比较了每个N=100个对象查询的预测类别+边界框与地面实况注释的长度N（因此，如果图像只包含4个对象，则96个注释将只有一个“无对象”作为类别和一个“无边界框”作为边界框）。使用匈牙利匹配算法寻找每个N查询与每个N注释的最佳一对一映射。接下来，使用标准交叉熵（用于类别）和L1损失的线性组合以及[广义IoU损失]（https://giou.stanford.edu/）（用于边界框）来优化模型的参数。

DETR可以自然地扩展以执行全景分割（将语义分割和实例分割统一起来）。[`~transformers.DetrForSegmentation`]在[`~transformers.DetrForObjectDetection`]上方添加了一个分割掩码头。可以同时训练掩码头，或者在一个两步过程中，首先训练[`~transformers.DetrForObjectDetection`]模型来检测“物体”（实例）和“物体”周围的边界框（背景物体，如树木，道路，天空），然后冻结所有权重，仅训练掩码头25个时期。实验证明，这两种方法的结果相似。请注意，预测框是训练可能，因为匈牙利匹配是使用框距离计算的。

提示：

- DETR使用所谓的**对象查询**来检测图像中的对象。查询的数量决定了可以在单个图像中检测到的最大对象数，默认设置为100（请参见[`~transformers.DetrConfig`]的`num_queries`参数）。请注意，最好有一些预留空间（在COCO中，作者使用了100，而COCO图像中的最大对象数为~70）。
- DETR的解码器并行地更新查询嵌入。这与使用自回归解码而不是并行解码的语言模型（例如GPT-2）不同。因此，不使用因果关注蒙版。
- DETR在查询和键进行投影之前，在每个自注意和交叉注意层的隐藏状态中添加位置嵌入。对于图像的位置嵌入，可以选择固定的正弦或学习的绝对位置嵌入。默认情况下，[`~transformers.DetrConfig`]的`position_embedding_type`参数设置为`sine`。
- 在训练期间，DETR的作者发现在解码器中使用辅助损失对于帮助模型输出每个类别的正确数量非常有帮助。如果将[`~transformers.DetrConfig`]的`auxiliary_loss`参数设置为`True`，则在每个解码器层之后添加预测前馈神经网络和匈牙利损失（FFN共享参数）。
- 如果要在多个节点的分布环境中训练模型，应该在modeling_detr.py的DetrLoss类中更新_num_boxes_变量。当在多个节点上进行训练时，应将其设置为所有节点上平均目标框的数量，可以在原始实现[here](https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/detr.py#L227-L232)中找到。
- [`~transformers.DetrForObjectDetection`]和[`~transformers.DetrForSegmentation`]可以使用[timm库](https://github.com/rwightman/pytorch-image-models)中可用的任何卷积骨干网络进行初始化。例如，可以通过将[`~transformers.DetrConfig`]的`backbone`属性设置为`"tf_mobilenetv3_small_075"`，然后使用该配置初始化模型来使用MobileNet骨干网。
- DETR调整输入图像的大小，使最短边至少有一定数量的像素，而最长边至多为1333像素。在训练时，使用缩放增强来随机将最短边设置为至少480像素，最多800像素。在推理时间，最短边被设置为800。可以使用[`~transformers.DetrImageProcessor`]为模型准备图像（和可选的COCO格式的注释）。由于这种调整大小，批处理中的图像大小可能不同。DETR通过将图像填充到批处理中的最大大小，并创建一个像素掩码来指示哪些像素是实际的/哪些是填充来解决这个问题。或者，您还可以定义一个自定义的`collate_fn`，以便使用[`~transformers.DetrImageProcessor.pad_and_create_pixel_mask`]将图像批处理在一起。
- 图像的大小将决定使用的内存量，并因此决定`batch_size`。建议每个GPU使用2个batch大小。有关更多信息，请参见[此Github线程](https://github.com/facebookresearch/detr/issues/150)。

有三种实例化DETR模型的方法（取决于您的首选）：

选项1：使用预训练权重实例化整个模型的DETR
```py
>>> from transformers import DetrForObjectDetection

>>> model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```

选项2：使用Transformer的随机初始化权重，但使用骨干网络的预训练权重实例化DETR
```py
>>> from transformers import DetrConfig, DetrForObjectDetection

>>> config = DetrConfig()
>>> model = DetrForObjectDetection(config)
```
选项3：使用随机初始化权重的骨干网络+ Transformer实例化DETR
```py
>>> config = DetrConfig(use_pretrained_backbone=False)
>>> model = DetrForObjectDetection(config)
```

总之，请考虑以下表格：

| 任务| 目标检测| 实例分割| 全景分割|
|---------|----------|-----------|-----------|
| **说明**| 在图像中预测围绕物体的边界框和类标签| 在图像中预测围绕物体（即实例）的掩码| 在图像中预测围绕物体（即实例）以及“物品”（即背景物体，如树木和道路）的掩码|
| **模型**| [`~transformers.DetrForObjectDetection`]| [`~transformers.DetrForSegmentation`]| [`~transformers.DetrForSegmentation`]|
| **示例数据集**| COCO检测| COCO检测、COCO全景| COCO全景  |                                                                  |
| **提供给**  [`~transformers.DetrImageProcessor`]的批注格式| {'image_id': `int`, 'annotations': `List[Dict]`}，每个字典都是COCO对象注释  | {'image_id': `int`, 'annotations': `List[Dict]`}（在COCO检测的情况下）或{'file_name': `str`，'image_id': `int`，'segments_info': `List[Dict]`}（在COCO全景的情况下）| {'file_name': `str`, 'image_id': `int`, 'segments_info': `List[Dict]`，masks_path（指向包含掩码PNG文件的目录的路径）|
| **后处理**（即将模型的输出转换为COCO API）| [`~transformers.DetrImageProcessor.post_process`] | [`~transformers.DetrImageProcessor.post_process_segmentation`] | [`~transformers.DetrImageProcessor.post_process_segmentation`]，[`~transformers.DetrImageProcessor.post_process_panoptic`] |
| **评估器** | `CocoEvaluator`，`iou_types="bbox"` | `CocoEvaluator`，`iou_types="bbox"`或`"segm"` | `CocoEvaluator`，`iou_tupes="bbox"`或`"segm"`，`PanopticEvaluator` |

简而言之，应以COCO检测或COCO全景格式准备数据，然后使用[`~transformers.DetrImageProcessor`]创建`pixel_values`，`pixel_mask`和可选的`labels`，然后用于训练（或微调）模型。对于评估，应首先使用[`~transformers.DetrImageProcessor`]的其中一种后处理方法将模型的输出转换。这些可以提供给`CocoEvaluator`或`PanopticEvaluator`，允许您计算诸如均值平均精度（mAP）和全景质量（PQ）之类的指标。后者对象在[原始存储库](https://github.com/facebookresearch/detr)中实现。有关评估的更多信息，请参见[示例笔记本](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR)。

## 资源

官方Hugging Face和社区（由🌎表示）资源列表，以帮助您开始使用DETR。

<PipelineTag pipeline="object-detection"/>

- 所有示例笔记本演示了在自定义数据集上微调[`DetrForObjectDetection`]和[`DetrForSegmentation`]，可以在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR)找到。
- 另请参阅：[对象检测任务指南](../tasks/object_detection)

如果您有兴趣提交一个要包含在这里的资源，请随时提交拉取请求，我们将对其进行审核！资源应理想地展示新内容，而不是重复现有资源。

## DETR特定的输出

[[autodoc]] models.detr.modeling_detr.DetrModelOutput

[[autodoc]] models.detr.modeling_detr.DetrObjectDetectionOutput

[[autodoc]] models.detr.modeling_detr.DetrSegmentationOutput

## DetrConfig

[[autodoc]] DetrConfig

## DetrImageProcessor

[[autodoc]] DetrImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## DetrFeatureExtractor

[[autodoc]] DetrFeatureExtractor
    - __call__
    - post_process_object_detection
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## DetrModel

[[autodoc]] DetrModel
    - forward

## DetrForObjectDetection

[[autodoc]] DetrForObjectDetection
    - forward

## DetrForSegmentation

[[autodoc]] DetrForSegmentation
    - forward