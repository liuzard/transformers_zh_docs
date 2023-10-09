<!--版权所有2022年NVIDIA和HuggingFace团队。

根据Apache许可证第2版（“许可证”）获得许可；在遵守许可证的情况下，除非有适用法律要求或书面同意，否则你不得使用此文件。你可以获取许可证的副本，在

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则以“按原样”为基础的软件分发是根据许可证分发的，不附带任何形式的担保或条件。请参阅许可证，了解许可证下的特定语言和限制。

⚠️ 注意，此文件为Markdown格式，但包含我们doc-builder（类似于MDX）的特定语法，你的Markdown查看器可能无法正确呈现。

-->

# GroupViT

## 概览

GroupViT模型是由Jiarui Xu、Shalini De Mello、Sifei Liu、Wonmin Byeon、Thomas Breuel、Jan Kautz和Xiaolong Wang在[GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094)中提出的。在[CLIP](clip)的启发下，GroupViT是一种视觉-语言模型，可以在任何给定词汇类别上执行零样本语义分割。

来自论文的摘要如下：

*分组和识别是视觉场景理解的重要组成部分，例如目标检测和语义分割。在端到端深度学习系统中，图像区域的分组通常通过自上而下的监督（来自像素级识别标签）隐式进行。而本文中，我们提出将分组机制重新引入到深度网络中，从而使只有文本监督的情况下自动产生语义分割。我们提出了一种层次分组视觉Transformer（GroupViT），它超越了常规的网格结构表示，并学习将图像区域分组成逐渐更大的任意形状的区段。我们使用对比损失在大规模图像-文本数据集上联合训练GroupViT和文本编码器。仅通过文本监督，而不需要任何像素级注释，GroupViT学会了将语义区域组合在一起，并成功地以零样本方式转移到语义分割任务上，即无需进一步的微调。在PASCAL VOC 2012数据集上，它在零样本精度上达到了52.3％的mIoU，在PASCAL Context数据集上达到了22.4％的mIoU，并且与需要更高级别的监督的最先进的迁移学习方法竞争性地表现。*

提示：

- 你可以在`GroupViTModel`的前向过程中指定`output_segmentation=True`，以获取输入文本的分割logits。

该模型由[xvjiarui](https://huggingface.co/xvjiarui)贡献。TensorFlow版本由[ariG23498](https://huggingface.co/ariG23498)在[Yih-Dar SHIEH](https://huggingface.co/ydshieh)，[Amy Roberts](https://huggingface.co/amyeroberts)和[Joao Gante](https://huggingface.co/joaogante)的帮助下贡献。原始代码可以在[这里](https://github.com/NVlabs/GroupViT)找到。

## 资源

官方Hugging Face和社区（由🌎表示）资源列表，可帮助你快速入门GroupViT。

- 快速入门GroupViT的最简单方法是查看[示例笔记本](https://github.com/xvjiarui/GroupViT/blob/main/demo/GroupViT_hf_inference_notebook.ipynb)（展示了零样本分割推断）。
- 你还可以查看[HuggingFace Spaces演示](https://huggingface.co/spaces/xvjiarui/GroupViT)，体验GroupViT。

## GroupViTConfig

[[autodoc]] GroupViTConfig
    - from_text_vision_configs

## GroupViTTextConfig

[[autodoc]] GroupViTTextConfig

## GroupViTVisionConfig

[[autodoc]] GroupViTVisionConfig

## GroupViTModel

[[autodoc]] GroupViTModel
    - forward
    - get_text_features
    - get_image_features

## GroupViTTextModel

[[autodoc]] GroupViTTextModel
    - forward

## GroupViTVisionModel

[[autodoc]] GroupViTVisionModel
    - forward

## TFGroupViTModel

[[autodoc]] TFGroupViTModel
    - call
    - get_text_features
    - get_image_features

## TFGroupViTTextModel

[[autodoc]] TFGroupViTTextModel
    - call

## TFGroupViTVisionModel

[[autodoc]] TFGroupViTVisionModel
    - call  