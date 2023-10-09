<!--
版权所有 2021 HuggingFace 团队。保留所有权利。

根据Apache许可证2.0版（"许可证"）许可；除非符合许可证的规定，
否则你不得使用此文件。你可以获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，按“原样”分发的软件基于许可证，不带任何担保或条件，
也不带任何形式的明示或暗示担保。请参阅许可证获取特定语言下的权限和限制。

️ 请注意，此文件为Markdown格式，但包含我们文档构建器（类似于MDX）的特定语法，可能无法在Markdown查看器中正确显示。

-->

# 特征提取器

特征提取器负责准备音频或视觉模型的输入特征。这包括从序列中进行特征提取，例如将音频文件预处理为Log-Mel Spectrogram特征，
从图像中进行特征提取，例如裁剪图像文件，还包括填充、归一化和转换为Numpy、PyTorch和TensorFlow张量等。

## FeatureExtractionMixin

[[autodoc]] feature_extraction_utils.FeatureExtractionMixin
- from_pretrained
- save_pretrained

## SequenceFeatureExtractor

[[autodoc]] SequenceFeatureExtractor
- pad

## BatchFeature

[[autodoc]] BatchFeature

## ImageFeatureExtractionMixin

[[autodoc]] image_utils.ImageFeatureExtractionMixin
