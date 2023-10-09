<!--版权所有2020年The HuggingFace团队。

根据Apache License，Version 2.0（"许可证"），除非获得许可证，
否则不得使用此文件。

您可以在以下位置获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可条款分发的软件是按"原样"分发的，
不附带任何明示或暗示的担保或条件。
请参阅许可证以获取特定的语言管理权限和限制。

⚠️请注意，此文件在Markdown中，但包含我们doc-builder的特定语法（类似于MDX），
可能无法在Markdown查看器中正确显示。

-->

# 导出 🤗 Transformers 模型至 ONNX

🤗 Transformers 提供了一个`transformers.onnx`软件包，可通过利用配置对象，将模型检查点转换为ONNX图形。

如需详细了解导出🤗 Transformers模型的内容，请参见 [guide](../serialization.md)。

## ONNX 配置

我们提供了三个抽象类供您继承，具体取决于您希望导出的模型架构类型：

* 基于Encoder的模型继承自[`~onnx.config.OnnxConfig`]
* 基于Decoder的模型继承自[`~onnx.config.OnnxConfigWithPast`]
* 基于Encoder-Decoder的模型继承自[`~onnx.config.OnnxSeq2SeqConfigWithPast`]

### OnnxConfig

[[autodoc]] onnx.config.OnnxConfig

### OnnxConfigWithPast

[[autodoc]] onnx.config.OnnxConfigWithPast

### OnnxSeq2SeqConfigWithPast

[[autodoc]] onnx.config.OnnxSeq2SeqConfigWithPast

## ONNX 特性

每个ONNX配置都与一组“特性”相关联，使您能够为不同类型的拓扑或任务导出模型。

### FeaturesManager

[[autodoc]] onnx.features.FeaturesManager