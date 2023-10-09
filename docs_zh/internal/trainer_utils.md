<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证第2版（“许可证”），除非符合许可证的规定，否则你不得使用此文件。你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件根据许可证以“现有的”基础分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解具体语言约束和限制。

⚠️ 请注意，此文件是Markdown格式的，但包含我们的文档构建器（类似于MDX）的特定语法，可能在你的Markdown查看器中不能正确渲染。-->

# Trainer的实用工具

此页面列出了[`Trainer`]所使用的所有实用函数。

其中大部分只在你研究库中的Trainer代码时有用。

## 实用工具

[[autodoc]] EvalPrediction

[[autodoc]] IntervalStrategy

[[autodoc]] enable_full_determinism

[[autodoc]] set_seed

[[autodoc]] torch_distributed_zero_first

## 回调函数内部

[[autodoc]] trainer_callback.CallbackHandler

## 分布式评估

[[autodoc]] trainer_pt_utils.DistributedTensorGatherer

## 分布式评估

[[autodoc]] HfArgumentParser

## 调试工具

[[autodoc]] debug_utils.DebugUnderflowOverflow