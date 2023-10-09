<!--
版权 2023 HuggingFace团队。 版权所有。

根据Apache许可证第2.0版（“许可证”）的规定，您不得使用除符合许可证的使用地外的此文件。 您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，否则根据许可证分发的软件是按“原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证，了解特定语言下权限的具体限制和限制。

⚠️请注意，此文件是Markdown格式的，但包含我们的文档构建器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确呈现。

-->

# ConvNeXt V2

## 概览

ConvNeXt V2模型是由Sanghyun Woo，Shoubhik Debnath，Ronghang Hu，Xinlei Chen，Zhuang Liu，In So Kweon，Saining Xie在论文“ConvNeXt V2：使用掩蔽的自动编码器进行卷积神经网络的共同设计和扩展”中提出的[ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)。
ConvNeXt V2是一个纯卷积模型(ConvNet)，受到Vision Transformers设计的启发，并是[ConvNeXT](convnext)的后继者。

论文的摘要如下：

*在改进的架构和更好的表示学习框架的推动下，图像识别领域在2020年代初取得了快速现代化和性能提升。 例如，ConvNeXt等现代卷积神经网络在各种场景中表现出强大的性能。虽然这些模型最初是为使用ImageNet标签的监督学习而设计的，但它们也有可能从掩蔽式自动编码器（MAE）等自我监督学习技术中受益。 然而，我们发现简单地将这两种方法结合起来会导致次优的性能。在本文中，我们提出了一种完全卷积的掩蔽式自动编码器框架和一种新的全局响应归一化（GRN）层，可以添加到ConvNeXt架构中以增强通道间特征竞争。 这种自我监督学习技术和架构改进的共同设计导致了一个称为ConvNeXt V2的新模型系列，该模型显着提升了纯卷积神经网络在各种了解基准测试中的性能，包括ImageNet分类，COCO检测和ADE20K分割。我们还提供了各种尺寸的预训练ConvNeXt V2模型，从高效的3.7M参数Atto模型，准确率为76.7％的ImageNet，到仅使用公共训练数据即可实现89.9％准确率的650M Huge模型。*

提示：

- 有关用法，请参见每个模型下面的代码示例。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnextv2_architecture.png"
alt="drawing" width="600"/>

<small>ConvNeXt V2架构。 摘自<a href="https://arxiv.org/abs/2301.00808">原始论文</a>。</small>

此模型由[adirik](https://huggingface.co/adirik)贡献。原始代码可以在[这里](https://github.com/facebookresearch/ConvNeXt-V2)找到。

## 资源

以下是官方Hugging Face和社区（由🌎表示）资源的列表，以帮助您开始使用ConvNeXt V2。

<PipelineTag pipeline="image-classification"/>

- 此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持[`ConvNextV2ForImageClassification`]。

如果您有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将对其进行审核！该资源应该展示出新的东西，而不是重复现有的资源。

## ConvNextV2Config

[[autodoc]] ConvNextV2Config

## ConvNextV2Model

[[autodoc]] ConvNextV2Model
    - forward

## ConvNextV2ForImageClassification

[[autodoc]] ConvNextV2ForImageClassification
    - forward
