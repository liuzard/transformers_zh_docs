<!--版权所有2022年The HuggingFace团队。 保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；您不得使用此文件，除非符合
许可证。 您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证的规定分发
“按现状”基础，不附带任何明示或暗示的保证或条件。
有关许可证下的特定语言，请参阅许可证的
特定语言约束向导。

⚠️请注意，此文件采用Markdown格式，但包含特定于我们的文档构建器（类似于MDX）的语法，可能无法正确呈现在您的Markdown查看器中。

-->

# 图像处理工具

此页面列出了图像处理器使用的所有实用函数，主要是用于处理图像的功能变换。

如果您正在研究库中的图像处理器的代码，则大多数这些函数只有在您才有用。

## 图像变换

[[autodoc]] image_transforms.center_crop

[[autodoc]] image_transforms.center_to_corners_format

[[autodoc]] image_transforms.corners_to_center_format

[[autodoc]] image_transforms.id_to_rgb

[[autodoc]] image_transforms.normalize

[[autodoc]] image_transforms.pad

[[autodoc]] image_transforms.rgb_to_id

[[autodoc]] image_transforms.rescale

[[autodoc]] image_transforms.resize

[[autodoc]] image_transforms.to_pil_image

## 图像处理Mixin

[[autodoc]] image_processing_utils.ImageProcessingMixin