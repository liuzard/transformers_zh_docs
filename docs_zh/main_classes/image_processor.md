<!--版权所有2022年The HuggingFace团队保留所有权利。

根据Apache许可证2.0版（“许可证”）的规定，除非符合许可证，否则您不得使用此文件。
您可以获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于"原样"分发的，不附带任何明示或暗示的担保。请查看许可证以了解许可证下的特定语言和限制。

⚠️请注意，此文件为Markdown格式，但包含我们doc-builder（类似于MDX）的特定语法，可能无法正确渲染在您的Markdown查看器中。-->

# 图像处理器

图像处理器负责为视觉模型准备输入特征并后处理其输出。其中包括调整大小、归一化和转换为PyTorch、TensorFlow、Flax和Numpy张量等的转换。它还可以包括模型特定的后处理，例如将逻辑转换为分割掩模。

## ImageProcessingMixin

[[autodoc]] image_processing_utils.ImageProcessingMixin
    - from_pretrained
    - save_pretrained

## BatchFeature

[[autodoc]] BatchFeature

## BaseImageProcessor

[[autodoc]] image_processing_utils.BaseImageProcessor