<!--著作权 2020 年 HuggingFace 团队保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”），除非符合许可证的要求，否则不得使用此文件。
可以在以下位置获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，按原样发布的软件分发在“即使在没有任何担保或条件的情况下，在
基础”基础上。请参阅许可证下的特定语言以及许可限制等错误。

⚠️ 请注意这个文件是在 Markdown 中，但包含我们的文档构建器（类似于 MDX）的特定语法，这可能不会在你的 Markdown 查看器中正确显示。

-->

# 导出到 ONNX

将 🤗 Transformers 模型部署到生产环境通常需要将模型导出为序列化格式，以便在专用运行时和硬件上加载和执行。

🤗 Optimum 是 Transformers 的扩展，它通过其 `exporters` 模块使模型能够从 PyTorch 或 TensorFlow 导出为 ONNX 和 TFLite 等序列化格式。🤗 Optimum 还提供了一套性能优化工具，以实现在目标硬件上以最高效率进行模型训练和运行。

本指南演示了如何使用 🤗 Optimum 将 🤗 Transformers 模型导出为 ONNX，关于将模型导出为 TFLite 请参阅[导出到 TFLite 页面](tflite.md)。

## 导出到 ONNX

[ONNX（Open Neural Network eXchange）](http://onnx.ai)是一种开放标准，用于定义一组通用算子和通用文件格式，以在包括 PyTorch 和 TensorFlow 在内的各种框架中表示深度学习模型。当一个模型被导出为 ONNX 格式时，这些算子会被用于构建一个计算图（通常称为“中间表示”），代表数据在神经网络中的流动。

通过公开具有标准化算子和数据类型的图，ONNX 使得在不同框架之间切换变得容易。例如，一个在 PyTorch 中训练的模型可以被导出为 ONNX 格式，然后在 TensorFlow 中导入（反之亦然）。

导出为 ONNX 格式后，可以对模型进行以下操作：
- 通过诸如 [图优化](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) 和 [量化](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization) 等技术对推理进行优化。
- 使用 ONNX Runtime 通过 [`ORTModelForXXX` 类](https://huggingface.co/docs/optimum/onnxruntime/package_reference/modeling_ort)运行，
它们与 🤗 Transformers 中你习惯使用的 `AutoModel` API 相同。
- 使用[优化的推理pipeline](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines)，其与 🤗 Transformers 中的 [`pipeline`] 函数具有相同的 API。

🤗 Optimum 通过利用配置对象提供对 ONNX 导出的支持。这些配置对象针对许多模型体系结构都已准备好，并设计易于扩展到其他体系结构。

有两种方法可以将 🤗 Transformers 模型导出为 ONNX，我们在这里展示两种方法：

- 使用 🤗 Optimum 的 CLI 导出。
- 使用 `optimum.onnxruntime` 导出。

### 使用 CLI 将 🤗 Transformers 模型导出到 ONNX

要将 🤗 Transformers 模型导出到 ONNX，首先安装额外的依赖项：

```bash
pip install optimum[exporters]
```

要查看所有可用的参数，请参阅[🤗 Optimum 文档](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli)，
或在命令行中查看帮助：

```bash
optimum-cli export onnx --help
```

要从 🤗 Hub 导出模型的检查点，例如 `distilbert-base-uncased-distilled-squad`，运行以下命令：

```bash
optimum-cli export onnx --model distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
```

你应该会看到日志显示进度并显示保存了结果 `model.onnx` 的位置，例如：

```bash
Validating ONNX model distilbert_base_uncased_squad_onnx/model.onnx...
	-[✓] ONNX model output names match reference model (start_logits, end_logits)
	- Validating ONNX Model output "start_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
	- Validating ONNX Model output "end_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
The ONNX export succeeded and the exported model was saved at: distilbert_base_uncased_squad_onnx
```

上面的示例演示了如何导出来自 🤗 Hub 的检查点。当导出本地模型时，首先确保将模型的权重和分词器文件保存在同一个目录（`local_path`）。当使用 CLI 时，将 `local_path` 传递给 `model` 参数，而不是检查点名称在 🤗 Hub 中，并提供 `--task` 参数。你可以在[🤗 Optimum 文档](https://huggingface.co/docs/optimum/exporters/task_manager)中查看支持的任务列表。如果未提供 `task` 参数，它将默认为不具有任何任务特定头的模型体系结构。

```bash
optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/
```

然后可以在支持 ONNX 标准的[许多加速器](https://onnx.ai/supported-tools.html#deployModel)之一上运行结果 `model.onnx`。例如，我们可以使用 [ONNX Runtime](https://onnxruntime.ai/) 加载和运行模型，如下所示：

```python
>>> from transformers import AutoTokenizer
>>> from optimum.onnxruntime import ORTModelForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> model = ORTModelForQuestionAnswering.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> inputs = tokenizer("What am I using?", "Using DistilBERT with ONNX Runtime!", return_tensors="pt")
>>> outputs = model(**inputs)
```

在 🤗 Hub 上的 TensorFlow 检查点的过程相同。例如，这是如何导出来自 [Keras 组织](https://huggingface.co/keras-io)的纯 TensorFlow 检查点：

```bash
optimum-cli export onnx --model keras-io/transformers-qa distilbert_base_cased_squad_onnx/
```

### 使用 `optimum.onnxruntime` 将 🤗 Transformers 模型导出到 ONNX

与 CLI 相比，你也可以按以下方式以编程方式将 🤗 Transformers 模型导出为 ONNX：

```python
>>> from optimum.onnxruntime import ORTModelForSequenceClassification
>>> from transformers import AutoTokenizer

>>> model_checkpoint = "distilbert_base_uncased_squad"
>>> save_directory = "onnx/"

>>> # 从 transformers 加载模型并导出为 ONNX
>>> ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
>>> tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

>>> # 保存 ONNX 模型和分词器
>>> ort_model.save_pretrained(save_directory)
>>> tokenizer.save_pretrained(save_directory)
```

### 导出不支持的体系结构的模型

如果你希望通过为当前无法导出的模型添加支持来进行贡献，你应首先检查 [`optimum.exporters.onnx`](https://huggingface.co/docs/optimum/exporters/onnx/overview) 是否支持该模型，如果不支持，你可以直接[对 🤗 Optimum 进行贡献](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute)。

### 使用 `transformers.onnx` 导出模型

<Tip warning={true}>

`tranformers.onnx` 不再维护，请参考上述使用 🤗 Optimum 导出模型的方法。该部分将在将来的版本中被删除。

</Tip>

要使用 `tranformers.onnx` 导出 🤗 Transformers 模型到 ONNX，请先安装额外的依赖项：

```bash
pip install transformers[onnx]
```

使用 `transformers.onnx` 包作为 Python 模块，通过使用现成的配置导出检查点：

```bash
python -m transformers.onnx --model=distilbert-base-uncased onnx/
```

这会导出由 `--model` 参数定义的检查点的 ONNX 图。传递任何在 🤗 Hub 上或本地存储的检查点。
然后，可以在许多支持 ONNX 标准的加速器上加载和运行生成的 `model.onnx`，例如使用 ONNX Runtime 运行模型的示例如下：

```python
>>> from transformers import AutoTokenizer
>>> from onnxruntime import InferenceSession

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
>>> session = InferenceSession("onnx/model.onnx")
>>> # ONNX Runtime 期望输入为 NumPy 数组
>>> inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")
>>> outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
```

所需的输出名称（如 `["last_hidden_state"]`）可以通过查看每个模型的 ONNX 配置得到。例如，对于 DistilBERT，我们有：

```python
>>> from transformers.models.distilbert import DistilBertConfig, DistilBertOnnxConfig

>>> config = DistilBertConfig()
>>> onnx_config = DistilBertOnnxConfig(config)
>>> print(list(onnx_config.outputs.keys()))
["last_hidden_state"]
```

在 🤗 Hub 上的 TensorFlow 检查点上，该过程是相同的。例如，导出纯 TensorFlow 检查点的方法如下：

```bash
python -m transformers.onnx --model=keras-io/transformers-qa onnx/
```

要导出本地存储的模型，请将模型的权重和分词器文件保存在相同目录中（例如 `local-pt-checkpoint`），然后通过将 `transformers.onnx` 包的 `--model` 参数指向所需目录来将其导出为 ONNX：

```bash
python -m transformers.onnx --model=local-pt-checkpoint onnx/
```