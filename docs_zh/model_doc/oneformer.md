## 概览

OneFormer模型由Jitesh Jain、Jiachen Li、MangTik Chiu、Ali Hassani、Nikita Orlov和Humphrey Shi在《OneFormer: One Transformer to Rule Universal Image Segmentation》中提出。OneFormer是一个通用的图像分割框架，可以在单个全景数据集上进行训练，执行语义、实例和全景分割任务。OneFormer使用任务token来将模型针对当前任务进行调整，使得架构在训练时具有任务引导性，并在推断时具有任务动态性。

本文摘要如下：

*通用图像分割并非新概念。过去几十年来，统一图像分割的尝试包括场景解析、全景分割以及最近的新全景架构。然而，这些全景架构并没有真正统一图像分割，因为它们需要分别在语义分割、实例分割或全景分割上进行训练，以达到最佳性能。理想情况下，一个真正的通用框架只需要训练一次，并在所有三种图像分割任务上实现SOTA性能。为此，我们提出了OneFormer，这是一个通用的图像分割框架，通过多任务一次性训练的设计来统一分割。首先，我们提出了一个任务依赖的联合训练策略，可以在单个多任务训练过程中使用每个领域（语义、实例和全景分割）的真实标注进行训练。其次，我们引入了一个任务token来将我们的模型与当前任务关联起来，使得我们的模型可以根据具体任务进行动态调整，以支持多任务训练和推断。第三，我们提出了在训练过程中使用查询文本对比损失来建立更好的任务间和类别间区分。值得注意的是，尽管后者需要使用三倍的资源来分别在ADE20k、CityScapes和COCO上进行训练，但我们的单个OneFormer模型在这三个分割任务上的性能均优于专门的Mask2Former模型。通过使用新的ConvNeXt和DiNAT主干，我们观察到了更多的性能提升。我们相信OneFormer是使图像分割更加通用和可访问的重要一步。*

提示：

- 在推断过程中，OneFormer需要两个输入：*图像*和*任务token*。
- 在训练过程中，OneFormer只使用全景注释。
- 如果要在跨多个节点的分布式环境中训练模型，则应更新`modeling_oneformer.py`中的`OneFormerLoss`类中的`get_num_masks`函数。在多个节点进行训练时，它应设置为所有节点上目标掩码的平均数，如原始实现中所示[here](https://github.com/SHI-Labs/OneFormer/blob/33ebb56ed34f970a30ae103e786c0cb64c653d9a/oneformer/modeling/criterion.py#L287)。
- 可以使用[`OneFormerProcessor`]来为模型准备输入图像和任务输入以及可选的模型目标。[`OneFormerProcessor`]将[`OneFormerImageProcessor`]和[`CLIPTokenizer`]封装到一个实例中，用于同时准备图像和编码任务输入。
- 要获得最终的分割结果，可以根据任务调用[`~OneFormerProcessor.post_process_semantic_segmentation`]、[`~OneFormerImageProcessor.post_process_instance_segmentation`]或[`~OneFormerImageProcessor.post_process_panoptic_segmentation`]。这三种任务都可以使用[`OneFormerForUniversalSegmentation`]输出来解决，全景分割还接受一个可选的`label_ids_to_fuse`参数，用于将目标对象（例如天空）的实例合并在一起。

下图显示了OneFormer的架构，摘自[原始论文](https://arxiv.org/abs/2211.06220)。

![oneformer_architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_architecture.png)

此模型由[Jitesh Jain](https://huggingface.co/praeclarumjj3)贡献。原始代码可以在[这里](https://github.com/SHI-Labs/OneFormer)找到。

## 资源

下面是一些官方Hugging Face资源和社区资源（由🌎表示），可帮助你开始使用OneFormer。

- 有关推断+对自定义数据进行微调的演示笔记本可在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/OneFormer)找到。

如果你有兴趣提交资源以包括在此处，请随时打开拉取请求，我们将对其进行审核。资源应该展示出一些新东西，而不是重复已有的资源。

## OneFormer特定输出

[[autodoc]] models.oneformer.modeling_oneformer.OneFormerModelOutput

[[autodoc]] models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput

## OneFormerConfig

[[autodoc]] OneFormerConfig

## OneFormerImageProcessor

[[autodoc]] OneFormerImageProcessor
    - preprocess
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## OneFormerProcessor

[[autodoc]] OneFormerProcessor

## OneFormerModel

[[autodoc]] OneFormerModel
    - forward

## OneFormerForUniversalSegmentation

[[autodoc]] OneFormerForUniversalSegmentation
    - forward