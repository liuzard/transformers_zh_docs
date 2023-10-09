<!--版权所有 2020 年 HuggingFace 团队. 保留所有权利。

基于 Apache 许可证第 2 版 ("许可证")；在符合许可的前提下，你不得使用此文件。你可以获取此许可的副本，网址为

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，否则按"原样"分发本软件。本许可不允许你以任何方式领取、使用、复制、修改、分发、出售、许可或转让本软件中的任何权利或软件文档的任何副本。本软件由版权所有人按"原样"提供，没有明示或暗示的任何保证或条件，无论是明示或暗示的保证、条件或其他，包括非侵权、适销性或其他特定用途的适用性。有关限制在特定法律下的责任，请参阅许可证。

请注意，此文件位于 Markdown 中，但包含对我们的文档构建程序（类似于 MDX）的特定语法，可能无法在你的 Markdown 查看器中正确渲染。

-->

# 基准测试

<Tip warning={true}>

Hugging Face 的基准测试工具已经被弃用，建议使用外部的基准测试库来测量 Transformer 模型的速度和内存复杂度。

</Tip>

[[open-in-colab]]

让我们看一下如何对🤗 Transformers模型进行基准测试，最佳实践以及已有的基准测试。

可以在[这里](https://github.com/huggingface/notebooks/tree/main/examples/benchmark.ipynb)找到一个更详细解释如何对🤗 Transformers模型进行基准测试的笔记本。

## 如何对🤗 Transformers模型进行基准测试

[`PyTorchBenchmark`] 和 [`TensorFlowBenchmark`] 类允许灵活地对🤗 Transformers模型进行基准测试。基准测试类允许我们测量推断和训练的 *峰值内存使用量* 和 *所需时间*。

<Tip>

这里，推断定义为单次前向传递，训练定义为单次前向传递和后向传递。

</Tip>

[`PyTorchBenchmark`] 和 [`TensorFlowBenchmark`] 类需要相应的 [`PyTorchBenchmarkArguments`] 和 [`TensorFlowBenchmarkArguments`] 类型的对象进行实例化。[`PyTorchBenchmarkArguments`] 和 [`TensorFlowBenchmarkArguments`] 是数据类，包含其相应基准测试类所需的所有相关配置。以下示例展示了如何对类型为 *bert-base-cased* 的BERT模型进行基准测试。

<frameworkcontent>
<pt>
```py
>>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

>>> args = PyTorchBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
>>> benchmark = PyTorchBenchmark(args)
```
</pt>
<tf>
```py
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments

>>> args = TensorFlowBenchmarkArguments(
...     models=["bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> benchmark = TensorFlowBenchmark(args)
```
</tf>
</frameworkcontent>

这里，基准测试参数数据类传入了三个参数，即 `models`、`batch_sizes` 和 `sequence_lengths`。`models` 参数是必需的，并且需要一个来自[model hub](https://huggingface.co/models) 的模型标识符列表。`batch_sizes` 和 `sequence_lengths` 是列表参数，定义了对模型进行基准测试时的 `input_ids` 的大小。还有许多其他可以通过基准测试参数数据类进行配置的参数。要获取有关这些参数的更多详细信息，可以直接查阅文件 `src/transformers/benchmark/benchmark_args_utils.py`、`src/transformers/benchmark/benchmark_args.py`（用于PyTorch）和 `src/transformers/benchmark/benchmark_args_tf.py`（用于TensorFlow）。或者，可以从根目录运行以下 Shell 命令，分别打印出PyTorch和Tensorflow的所有可配置参数的描述性列表。

<frameworkcontent>
<pt>
```bash
python examples/pytorch/benchmarking/run_benchmark.py --help
```

然后，可以通过调用 `benchmark.run()` 来运行已实例化的基准测试对象。

```py
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
bert-base-uncased          8               8             0.006     
bert-base-uncased          8               32            0.006     
bert-base-uncased          8              128            0.018     
bert-base-uncased          8              512            0.088     
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
bert-base-uncased          8               8             1227
bert-base-uncased          8               32            1281
bert-base-uncased          8              128            1307
bert-base-uncased          8              512            1539
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: PyTorch
- use_torchscript: False
- framework_version: 1.4.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 08:58:43.371351
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</pt>
<tf>
```bash
python examples/tensorflow/benchmarking/run_benchmark_tf.py --help
```

然后可以通过调用 `benchmark.run()` 来运行已实例化的基准测试对象。

```py
>>> results = benchmark.run()
>>> print(results)
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
bert-base-uncased          8               8             0.005
bert-base-uncased          8               32            0.008
bert-base-uncased          8              128            0.022
bert-base-uncased          8              512            0.105
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
bert-base-uncased          8               8             1330
bert-base-uncased          8               32            1330
bert-base-uncased          8              128            1330
bert-base-uncased          8              512            1770
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: Tensorflow
- use_xla: False
- framework_version: 2.2.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:26:35.617317
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</tf>
</frameworkcontent>

默认情况下，对推断进行基准测试并测量所需时间和内存。在上面的示例输出中，前两个部分显示了与推断时间和推断内存相对应的结果。此外，在“环境信息”下的第三个部分打印出有关计算环境的所有相关信息，例如GPU类型、系统、库版本等。当在[`PyTorchBenchmarkArguments`]和[`TensorFlowBenchmarkArguments`]中添加`save_to_csv=True`参数时，这些信息可以选择保存到一个_.csv_文件中。在这种情况下，每个部分都保存在一个单独的_.csv_文件中。每个_.csv_文件的路径可以通过参数数据类进行定义。

除了通过模型标识符（例如 `bert-base-uncased`）对预训练模型进行基准测试之外，用户还可以通过任何可用的模型类对任意配置进行基准测试。在这种情况下，必须在基准测试参数中插入一系列配置，如下所示：

<frameworkcontent>
<pt>
```py
>>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig

>>> args = PyTorchBenchmarkArguments(
...     models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> config_base = BertConfig()
>>> config_384_hid = BertConfig(hidden_size=384)
>>> config_6_lay = BertConfig(num_hidden_layers=6)

>>> benchmark = PyTorchBenchmark(args, configs=[config_base, config_384_hid, config_6_lay])
>>> benchmark.run()
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length       Time in s                  
--------------------------------------------------------------------------------
bert-base                  8              128            0.006
bert-base                  8              512            0.006
bert-base                  8              128            0.018     
bert-base                  8              512            0.088     
bert-384-hid              8               8             0.006     
bert-384-hid              8               32            0.006     
bert-384-hid              8              128            0.011     
bert-384-hid              8              512            0.054     
bert-6-lay                 8               8             0.003     
bert-6-lay                 8               32            0.004     
bert-6-lay                 8              128            0.009     
bert-6-lay                 8              512            0.044
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length      Memory in MB 
--------------------------------------------------------------------------------
bert-base                  8               8             1277
bert-base                  8               32            1281
bert-base                  8              128            1307     
bert-base                  8              512            1539     
bert-384-hid              8               8             1005     
bert-384-hid              8               32            1027     
bert-384-hid              8              128            1035     
bert-384-hid              8              512            1255     
bert-6-lay                 8               8             1097     
bert-6-lay                 8               32            1101     
bert-6-lay                 8              128            1127     
bert-6-lay                 8              512            1359
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: PyTorch
- use_torchscript: False
- framework_version: 1.4.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:35:25.143267
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</pt>
<tf>
```py
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig

>>> args = TensorFlowBenchmarkArguments(
...     models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> config_base = BertConfig()
>>> config_384_hid = BertConfig(hidden_size=384)
>>> config_6_lay = BertConfig(num_hidden_layers=6)

>>> benchmark = TensorFlowBenchmark(args, configs=[config_base, config_384_hid, config_6_lay])
>>> benchmark.run()
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length       Time in s                  
--------------------------------------------------------------------------------
bert-base                  8               8             0.005
bert-base                  8               32            0.008
bert-base                  8              128            0.022
bert-base                  8              512            0.106
bert-384-hid              8               8             0.005
bert-384-hid              8               32            0.007
bert-384-hid              8              128            0.018
bert-384-hid              8              512            0.064
bert-6-lay                 8               8             0.002
bert-6-lay                 8               32            0.003
bert-6-lay                 8              128            0.0011
bert-6-lay                 8              512            0.074
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length      Memory in MB 
--------------------------------------------------------------------------------
bert-base                  8               8             1330
bert-base                  8               32            1330
bert-base                  8              128            1330
bert-base                  8              512            1770
bert-384-hid              8               8             1330
bert-384-hid              8               32            1330
bert-384-hid              8              128            1330
bert-384-hid              8              512            1540
bert-6-lay                 8               8             1330
bert-6-lay                 8               32            1330
bert-6-lay                 8              128            1330
bert-6-lay                 8              512            1540
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: Tensorflow
- use_xla: False
- framework_version: 2.2.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:38:15.487125
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</tf>
</frameworkcontent>

- 同样，这次我们测量了自定义配置的`BertModel`类的_推断时间_和_所需内存_。当决定对哪种配置进行模型训练时，这个功能特别有帮助。

  ## 基准测试最佳实践

  本节列出了在进行模型基准测试时应注意的几个最佳实践。

  - 目前，仅支持单设备基准测试。在使用GPU进行基准测试时，建议用户通过在shell中设置`CUDA_VISIBLE_DEVICES`环境变量来指定代码应在哪个设备上运行，例如在运行代码之前设置`export CUDA_VISIBLE_DEVICES=0`。
  - 选项`no_multi_processing`只应在测试和调试时设置为`True`。为确保准确的内存测量，建议通过将`no_multi_processing`设置为`True`，在单独的进程中运行每个内存基准测试。
  - 在共享模型基准测试结果时，应始终说明环境信息。由于不同的GPU设备、库版本等原因，结果可能会有很大差异，因此仅仅提供基准测试结果对社区来说并没有太大用处。

  ## 共享你的基准测试

  以前，所有可用的核心模型（当时是10个）都已经进行了_推断时间_的基准测试，涉及许多不同的设置：使用PyTorch，有无TorchScript，使用TensorFlow，有无XLA。所有这些测试都在CPU上进行（除了TensorFlow XLA）和GPU上进行。

  相关方法详见[此博客文章](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2)，结果可在[此处](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit?usp=sharing)找到。

  使用新的_基准测试_工具，与社区共享你的基准测试结果变得比以往更加容易：

- [PyTorch 基准测试结果](https://github.com/huggingface/transformers/tree/main/examples/pytorch/benchmarking/README.md)。
  - [TensorFlow 基准测试结果](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/benchmarking/README.md)。