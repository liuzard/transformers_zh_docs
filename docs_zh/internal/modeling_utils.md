<!--版权 2020 年 HuggingFace 团队。保留所有权利。

根据 Apache 许可证，版本 2.0（“许可证”）进行许可；除非符合许可证的规定，否则您不能使用此文件。
您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，本软件按“原样”分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️ 请注意，该文件采用 Markdown 格式，但包含特定语法以供我们的文档构建器（类似于 MDX）使用，可能在您的 Markdown 查看器中无法正确渲染。

-->

# 自定义层和实用工具

本页面列出了库中使用的所有自定义层，以及用于模型建模的实用函数。

如果您正在研究库中的模型代码，则这些大部分只有在此情况下才有用。


## PyTorch 自定义模块

[[autodoc]] pytorch_utils.Conv1D

[[autodoc]] modeling_utils.PoolerStartLogits
    - forward

[[autodoc]] modeling_utils.PoolerEndLogits
    - forward

[[autodoc]] modeling_utils.PoolerAnswerClass
    - forward

[[autodoc]] modeling_utils.SquadHeadOutput

[[autodoc]] modeling_utils.SQuADHead
    - forward

[[autodoc]] modeling_utils.SequenceSummary
    - forward

## PyTorch 辅助函数

[[autodoc]] pytorch_utils.apply_chunking_to_forward

[[autodoc]] pytorch_utils.find_pruneable_heads_and_indices

[[autodoc]] pytorch_utils.prune_layer

[[autodoc]] pytorch_utils.prune_conv1d_layer

[[autodoc]] pytorch_utils.prune_linear_layer

## TensorFlow 自定义层

[[autodoc]] modeling_tf_utils.TFConv1D

[[autodoc]] modeling_tf_utils.TFSequenceSummary

## TensorFlow 损失函数

[[autodoc]] modeling_tf_utils.TFCausalLanguageModelingLoss

[[autodoc]] modeling_tf_utils.TFMaskedLanguageModelingLoss

[[autodoc]] modeling_tf_utils.TFMultipleChoiceLoss

[[autodoc]] modeling_tf_utils.TFQuestionAnsweringLoss

[[autodoc]] modeling_tf_utils.TFSequenceClassificationLoss

[[autodoc]] modeling_tf_utils.TFTokenClassificationLoss

## TensorFlow 辅助函数

[[autodoc]] modeling_tf_utils.get_initializer

[[autodoc]] modeling_tf_utils.keras_serializable

[[autodoc]] modeling_tf_utils.shape_list