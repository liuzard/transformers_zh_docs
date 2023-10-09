<!--
2021年The HuggingFace团队保留所有权利。

根据Apache许可证2.0版（“许可证”）授权；
除非符合许可证的规定，否则不得使用此文件。
可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

请注意，此文件采用Markdown格式，但包含我们文档生成器（类似于MDX）的特定语法，可能在你的Markdown查看器中无法正确渲染。

-->

# 性能和可扩展性

训练大型Transformer模型并将其部署到生产环境中存在各种挑战。 
在训练过程中，模型可能需要比可用内存更多的GPU内存，或者训练速度较慢。在部署阶段，
模型可能无法在生产环境中处理所需的吞吐量。

本文档旨在帮助你克服这些挑战，并找到适合你用例的最佳设置。
指南分为训练和推断两个部分，因为每个部分都有不同的挑战和解决方案。
在每个部分中，你会找到针对不同硬件配置的独立指南，例如单个GPU与多个GPU用于训练，或CPU与GPU用于推断。

使用本文档作为你开始导航到符合你情景的方法的起点。

## 训练

高效地训练大型Transformer模型需要加速器，如GPU或TPU。最常见的情况是你只有一个GPU。
你可以应用于单个GPU上的提高训练效率的方法也适用于其他设置，例如多个GPU。但是，还有一些特定于多GPU或CPU训练的技术。我们在单独的部分中介绍它们。

* [在单个GPU上进行高效训练的方法和工具](perf_train_gpu_one.md)：从这里开始学习常见的方法，可以帮助优化GPU内存利用率，加速训练或同时达到这两个目的。
* [多GPU训练部分](perf_train_gpu_many.md)：浏览此部分，了解适用于多GPU环境的进一步优化方法，如数据、张量和pipeline并行。
* [CPU训练部分](perf_train_cpu.md)：了解在CPU上进行混合精度训练。
* [在多个CPU上高效训练](perf_train_cpu_many.md)：了解分布式CPU训练。
* [使用TensorFlow训练TPU](perf_train_tpu_tf.md)：如果你是第一次使用TPU，请参考此部分了解在TPU上进行训练和使用XLA的指南。
* [自定义用于训练的硬件](perf_hardware.md)：在构建自己的深度学习平台时获取技巧和窍门。
* [使用Trainer API进行超参数搜索](hpo_train.md)

## 推断

在生产环境中高效执行大型模型的推断与训练同样具有挑战性。在下面的部分中，我们介绍如何在CPU和单个/多个GPU环境中运行推断的步骤。

* [在单个CPU上进行推断](perf_infer_cpu.md)
* [在单个GPU上进行推断](perf_infer_gpu_one.md)
* [多GPU推断](perf_infer_gpu_many.md)
* [TensorFlow模型的XLA集成](tf_xla.md)


## 训练和推断

这里你将找到适用于训练模型或使用模型进行推断的技术、技巧和窍门。

* [实例化大型模型](big_models.md)
* [解决性能问题](debugging.md)

## 贡献

此文档远未完善，还需要添加更多内容，所以如果你有添加或更正的意见，请随时提交PR，或者如果你不确定，请启动一个Issue，我们可以在那里讨论细节。

在做出A优于B的贡献时，请尽量包含可复现的基准测试和/或该信息来源的链接（除非该信息直接来自你）。
