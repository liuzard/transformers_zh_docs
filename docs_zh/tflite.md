<!--版权2023年赞美脸团队。保留所有权利。

根据Apache License，版本2.0（“许可证”）获得许可；你不得使用此文件，除非符合许可证的要求。你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件可以根据“按原样的基础”分发，不附带任何形式的保证或条件，无论是明示的还是默示的。有关特定语言的详细信息，请参见许可证下面的限制。

⚠️请注意，此文件为Markdown格式，但包含我们doc-builder（类似于MDX）的特定语法，你的Markdown查看器可能无法正确渲染。

-->

# 导出为TFLite

[TensorFlow Lite](https://www.tensorflow.org/lite/guide) 是一个轻量级框架，用于在资源受限的设备上部署机器学习模型，例如手机、嵌入式系统和物联网设备。TFLite专为在计算能力、内存和电源消耗有限的设备上优化和运行模型而设计。
TensorFlow Lite模型以 `.tflite` 文件扩展名标识的特殊高效便携格式表示。

🤗Optimum通过 `exporters.tflite` 模块提供了将🤗Transformers模型导出为TFLite的功能。有关支持的模型架构列表，请参阅[🤗Optimum文档](https://huggingface.co/docs/optimum/exporters/tflite/overview)。

要将模型导出为TFLite，请安装所需的依赖项：

```bash
pip install optimum[exporters-tf]
```

要查看所有可用的参数，请参阅[🤗Optimum文档](https://huggingface.co/docs/optimum/main/en/exporters/tflite/usage_guides/export_a_model)，或在命令行中查看帮助：

```bash
optimum-cli export tflite --help
```

要从🤗Hub导出模型的检查点，例如`bert-base-uncased`，请运行以下命令：

```bash
optimum-cli export tflite --model bert-base-uncased --sequence_length 128 bert_tflite/
```

你将看到指示进度并显示保存的 `model.tflite` 文件的位置的日志，如下所示：

```bash
验证TFLite模型...
	-[✓] TFLite模型输出名称与参考模型（logits）匹配
	- 验证TFLite模型输出 "logits"：
		-[✓] (1, 128, 30522) 与 (1, 128, 30522) 匹配
		-[x] 值不够接近，最大差异：5.817413330078125e-05（atol：1e-05）
TensorFlow Lite导出成功，并出现警告：参考模型和TFLite导出模型之间的输出的最大绝对差异不在设置的容限1e-05范围内：
- logits：最大差异=5.817413330078125e-05。
导出的模型已保存在：bert_tflite
```

上面的示例说明了从🤗Hub导出检查点。在导出本地模型时，首先确保将模型的权重和分词器文件保存在同一个目录（`local_path`）下。使用CLI时，将 `local_path` 传递给 `model` 参数，而不是用🤗Hub上的检查点名称。