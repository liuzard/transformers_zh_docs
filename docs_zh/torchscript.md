<!-- 版权 2022 HuggingFace 团队。保留所有权利。

根据 Apache License，第2版 （“许可证”）获得许可；除非符合许可证，在旨在提高效率的 C + + 程序等其他程序中，你不得使用此文件。

你可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样”分发的，并不做任何明示或暗示的保证或条件。

请注意，该文件为 Markdown，但包含与我们的定义构建器相关的特定语法（类似于 MDX），可能无法在你的 Markdown 查看器中正确显示。

-->

# 导出至 TorchScript

<Tip>

这只是我们在使用 TorchScript 进行实验的初步阶段，我们仍然在探索其在可变输入大小模型中的能力。这对我们来说是一个感兴趣的焦点，我们将在即将发布的版本中深入分析，提供更多代码示例、更灵活的实现以及使用编译的 TorchScript 与基于 Python 的代码进行比较的性能基准。

</Tip>

根据[TorchScript文档](https://pytorch.org/docs/stable/jit.html)：

> TorchScript 是一种从 PyTorch 代码创建可序列化和可优化模型的方式。

PyTorch 提供了两个模块，[JIT and TRACE](https://pytorch.org/docs/stable/jit.html)，允许开发者将他们的模型导出以在其他程序中重用，例如面向效率的 C++ 程序。

我们提供了一个接口，可让你将 🤗 Transformers 模型导出到 TorchScript，以便在与基于 PyTorch 的 Python 程序不同的环境中重用它们。在这里，我们将解释如何使用 TorchScript 导出和使用我们的模型。

导出模型需要两个条件：

- 使用 `torchscript` 标志实例化模型
- 使用虚拟输入进行前向传递

这些条件意味着开发者需要注意一些细节，如下所述。

## TorchScript 标志和绑定的权重

`torchscript` 标志是必需的，因为大多数 🤗 Transformers 语言模型的 `Embedding` 层和 `Decoding` 层之间存在绑定的权重。TorchScript 不允许导出具有绑定权重的模型，因此需要在导出之前解开并克隆这些权重。

使用 `torchscript` 标志实例化的模型将它们的 `Embedding` 层和 `Decoding` 层分开，这意味着它们不应该进行后续训练。训练会导致两个层之间的不同步，导致意外的结果。

对于没有语言模型头的模型来说，情况并非如此，因为这些模型没有绑定的权重。这些模型可以在没有 `torchscript` 标志的情况下安全地导出。

## 虚拟输入和标准长度

虚拟输入用于模型的前向传递。当输入的值在层之间传播时，PyTorch 会跟踪在每个张量上执行的不同操作。然后，这些记录的操作用于创建模型的 *trace*。

trace 是相对于输入维度创建的。因此，它受到虚拟输入维度的限制，对于其他序列长度或批次大小将无法工作。在尝试不同大小时，会引发以下错误：

```
`在一个非单例维度 2 中，张量的扩展尺寸（3）必须匹配现有尺寸（7）`
```

我们建议你使用虚拟输入大小至少与在推断过程中将提供给模型的最大输入一样大进行模型的追踪。可以使用填充来填充缺失的值。但是，由于模型与较大的输入大小进行追踪，矩阵的维度也会很大，导致计算量更大。

请注意每个输入上执行的操作的总数，并在导出不同序列长度的模型时仔细监控性能。

## 在 Python 中使用 TorchScript

本节演示了如何保存和加载模型，以及如何使用 trace 进行推断。

### 保存模型

要使用 TorchScript 导出 `BertModel`，请从 `BertConfig` 类实例化 `BertModel`，然后将其保存到文件名为 `traced_bert.pt` 的磁盘上：

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch

enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    torchscript=True,
)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with *from_pretrained* you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")
```

### 加载模型

现在，你可以从磁盘上加载之前保存的 `BertModel`，即 `traced_bert.pt`，并将其用于之前初始化的 `dummy_input` 上：

```python
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```

### 使用追踪模型进行推断

通过使用追踪模型的 `__call__` 方法，可以使用追踪模型进行推断：

```python
traced_model(tokens_tensor, segments_tensors)
```

## 使用 Neuron SDK 将 Hugging Face TorchScript 模型部署到 AWS

AWS 推出了 [Amazon EC2 Inf1](https://aws.amazon.com/ec2/instance-types/inf1/) 实例系列，用于在云中进行低成本、高性能的机器学习推理。Inf1 实例由 AWS Inferentia 芯片提供支持，该芯片是专门为深度学习推理工作负载而构建的定制硬件加速器。[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#) 是用于 Inferentia 的 SDK，支持跟踪和优化 Transformers 模型以在 Inf1 上部署。Neuron SDK 提供：

1. 易于使用的 API，只需更改一行代码即可跟踪和优化 TorchScript 模型，用于云端推理。
2. 针对 [改进的成本性能](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/benchmark/)的现成性能优化。
3. 对使用 [PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html) 或 [TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/huggingface_bert/huggingface_bert.html) 构建的 Hugging Face transformers 模型的支持。

### 影响

基于 [BERT（Bidirectional Encoder Representations from Transformers）](https://huggingface.co/docs/transformers/main/model_doc/bert) 架构或其变种，例如 [distilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert) 和 [roBERTa](https://huggingface.co/docs/transformers/main/model_doc/roberta)，在 Inf1 上运行时最适用于非生成型任务，例如提取式问答、序列分类和标记分类。但是，文本生成任务仍然可以调整以在 Inf1 上运行，参考该 [AWS Neuron Marian MT 教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html)。有关可以直接在 Inferentia 上转换的模型的更多信息，请参阅 Neuron 文档的 [模型架构适配](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/models/models-inferentia.html#models-inferentia) 部分。

### 依赖项

使用 AWS Neuron 转换模型需要 [Neuron SDK 环境](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html#installation-guide)，在 [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-launching.html) 上预配置。

### 转换为 AWS Neuron 的模型

使用 [在 Python 中使用 TorchScript](torchscript.md#在-python-中使用-torchscript) 中的相同代码来跟踪 `BertModel` 来将模型转换为 AWS Neuron。导入 `torch.neuron` 框架扩展以通过 Python API 访问 Neuron SDK 的组件：

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.neuron
```

只需修改以下行：

```diff
- torch.jit.trace(model, [tokens_tensor, segments_tensors])
+ torch.neuron.trace(model, [token_tensor, segments_tensors])
```

这样就使 Neuron SDK 能够跟踪模型并对其进行优化，以在 Inf1 实例上运行。

有关 AWS Neuron SDK 功能、工具、示例教程和最新更新的更多信息，请参阅 [AWS NeuronSDK 文档](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html)。