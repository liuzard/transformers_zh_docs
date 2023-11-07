<!--版权 2020 年 HuggingFace 小组。保留所有权利。

根据 Apache 许可证 2.0 版（“许可证”）的规定，在符合许可证的前提下，你不得使用本文件。
你可以获取许可证拷贝的链接如下：

http://www.apache.org/licenses/LICENSE-2.0

除非根据适用的法律规定或书面同意，否则按“原样”分发的软件在任何情况下都没有任何形式的担保或条件，无论是明示的还是暗示的。请参阅许可证以获取
许可证下的特定语言和限制。

⚠️ 请注意，此文件使用 Markdown 编写，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在你的 Markdown 预览器中正确显示。

-->

# 哲学

🤗Transformers 是一个为以下目标而构建的观点化库：

- 机器学习研究人员和教育工作者，希望使用、研究或扩展大规模 Transformers 模型。
- 实践者，希望对这些模型进行微调或在生产环境中使用它们，或两者兼而有之。
- 工程师，只想下载一个预训练的模型并用它来解决给定的机器学习任务。

该库的设计具有两个强烈的目标：

1. 尽可能简单和快速地使用：

  - 我们确实限制了用户界面的抽象数量，事实上，几乎没有抽象，只需要三个标准类来使用每个模型：[配置文件](main_classes/configuration)、[模型](main_classes/model)和预处理类（用于 NLP 的 [分词器](main_classes/tokenizer)、用于视觉的 [图像处理器](main_classes/image_processor)、用于音频的 [特征提取器](main_classes/feature_extractor) 和多模态输入的 [处理器](main_classes/processors)）。
  - 所有这些类都可以通过使用通用的 `from_pretrained()` 方法从预训练实例中以简单和统一的方式进行初始化。该方法会下载（如有需要）、缓存并加载与预训练检查点相关联的类实例和关联数据（配置的超参数、分词器的词汇表和模型的权重），
    这些预训练检查点位于[Hugging Face Hub](https://huggingface.co/models)提供的，或者是你自己保存的检查点上。
  - 除这三个基础类之外，该库提供了两个 API：[`pipeline`]，用于在给定任务上快速使用模型进行推断；
    [`Trainer`]，用于快速训练或微调 PyTorch 模型（所有 TensorFlow 模型与 `Keras.fit` 兼容）。
  - 因此，该库**不是**神经网络构建模块的模块化工具箱。如果你想扩展或构建库，只需使用常规的 Python、PyTorch、TensorFlow、Keras 模块，并从库的基类中继承以重用模型加载和保存等功能。如果想了解有关我们的模型编码哲学的更多信息，请查看我们的[Repeat Yourself](https://huggingface.co/blog/transformers-design-philosophy)博客文章。

2. 提供尽可能接近原始模型的最先进性能模型：

  - 我们为每个架构提供至少一个示例，以重现该架构的官方作者提供的结果。
  - 代码通常尽量接近原始代码库，这意味着一些 PyTorch 代码可能不是*非常“pytorchic”*，因为它可能是由 TensorFlow 代码转换而来的，反之亦然。

另外还有一些目标：

- 尽可能一致地显示模型的内部内容：

  - 我们使用单个 API 访问完整的隐藏状态和注意权重。
  - 预处理类和基本模型 API 统一标准化，以便轻松切换模型。

- 使用可靠的工具来进行微调和研究这些模型：

  - 一种简单而一致的方法来向词汇表和嵌入中添加新标记以进行微调。
  - 对 Transformer 头进行掩码和修剪的简单方法。

- 轻松切换 PyTorch、TensorFlow 2.0 和 Flax，允许使用一个框架进行训练，使用另一个框架进行推断。

## 主要概念

该库围绕每个模型的三种类型的类构建：

- **模型类**可以是 PyTorch 模型（[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)）、Keras 模型（[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)）或 JAX/Flax 模型（[flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html)），它们与库中提供的预训练权重一起使用。
- **配置类**存储构建模型所需的超参数（例如层数和隐藏大小）。你不一定总是需要实例化这些类。特别是，如果你使用不进行任何修改的预训练模型，则创建模型时会自动处理实例化配置（这是模型的一部分）。
- **预处理类**将原始数据转换为模型接受的格式。[分词器](main_classes/tokenizer)存储每个模型的词汇表，并提供将字符串编码和解码为要馈送给模型的token嵌入索引列表的方法。[图像处理器](main_classes/image_processor)预处理视觉输入，[特征提取器](main_classes/feature_extractor)预处理音频输入，[处理器](main_classes/processors)处理多模态输入。

所有这些类都可以从预训练实例进行实例化、本地保存，并通过三种方法在 Hub 上共享：

- `from_pretrained()` 允许你从库本身提供的预训练版本，或者用户保存在本地（或服务器上）的版本实例化模型、配置和预处理类。
- `save_pretrained()` 允许你在本地保存模型、配置和预处理类，以便可以使用 `from_pretrained()` 进行重新加载。
- `push_to_hub()` 允许你将模型、配置和预处理类共享到 Hub，以便其他人可以轻松访问它们。