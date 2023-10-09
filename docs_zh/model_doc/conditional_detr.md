<!--版权2022年HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”）的规定，除非符合
许可证规定的要求，否则你不得使用本文件。
你可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件以“原样”
分布，不附带任何明示或暗示的担保或条件。请参阅许可证以了解
许可证下特定语言的权限和限制。

⚠️请注意，此文件是Markdown格式的，但包含我们文档构建器的特定语法（类似于MDX），可能无法在你的Markdown查看器中正确渲染。
-->

# 条件DETR

## 概述

条件DETR模型在[《为快速训练收敛的条件DETR》（Conditional DETR for Fast Training Convergence）](https://arxiv.org/abs/2108.06152)一文中被提出，作者是Depu Meng、Xiaokang Chen、Zejia Fan、Gang Zeng、Houqiang Li、Yuhui Yuan和Jingdong Wang。条件DETR使用了一种条件交叉注意力机制来实现快速DETR训练。条件DETR的收敛速度比DETR快6.7倍到10倍。

论文的摘要如下：

*最近开发的DETR方法将Transformer编码器和解码器架构应用于目标检测，并取得了令人满意的性能。在本文中，我们解决了训练收敛速度缓慢的关键问题，并提出了一种用于快速DETR训练的条件交叉注意力机制。我们的方法是受到DETR中的交叉注意力高度依赖内容嵌入来定位四个极点和预测框的启发。这增加了对高质量内容嵌入的需求，从而增加了训练难度。我们的方法名为条件DETR，它通过学习一个来自解码器嵌入的条件空间查询用于解码器的多头交叉注意力。好处是通过条件空间查询，每个交叉注意力头都能关注包含不同区域的带状区域，例如一个目标极点或目标框内的区域。这缩小了用于定位目标分类和框回归的不同区域的空间范围，从而减轻了对内容嵌入的依赖和训练困难。实验证明，对于骨干网R50和R101，条件DETR的收敛速度分别比原始DETR快6.7倍和10倍，对于更强的骨干网DC5-R50和DC5-R101，收敛速度分别比原始DETR快10倍。代码可在https://github.com/Atten4Vis/ConditionalDETR找到。*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/conditional_detr_curve.jpg"
alt="图像" width="600"/>

<small> 条件DETR的收敛速度较原始DETR明显更快。来自<a href="https://arxiv.org/abs/2108.06152">原始论文</a>。</small>

此模型由[DepuMeng](https://huggingface.co/DepuMeng)贡献。原始代码可在[此处](https://github.com/Atten4Vis/ConditionalDETR)找到。

## 文档资源

- [目标检测任务指南](../tasks/object_detection)

## ConditionalDetrConfig

[[autodoc]] ConditionalDetrConfig

## ConditionalDetrImageProcessor

[[autodoc]] ConditionalDetrImageProcessor
    - 预处理
    - 后处理目标检测
    - 后处理实例分割
    - 后处理语义分割
    - 后处理全景分割

## ConditionalDetrFeatureExtractor

[[autodoc]] ConditionalDetrFeatureExtractor
    - __call__
    - 后处理目标检测
    - 后处理实例分割
    - 后处理语义分割
    - 后处理全景分割

## ConditionalDetrModel

[[autodoc]] ConditionalDetrModel
    - forward

## ConditionalDetrForObjectDetection

[[autodoc]] ConditionalDetrForObjectDetection
    - forward

## ConditionalDetrForSegmentation

[[autodoc]] ConditionalDetrForSegmentation
    - forward