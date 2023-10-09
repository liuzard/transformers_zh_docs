版权所有2023年HuggingFace团队保留。

根据Apache许可证2.0版（"许可证"）许可;
您除遵守许可证外，不得使用此文件。
您可以在以下网址获得许可证的副本

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则软件根据"按原样"的基础分发,
无论是明示还是暗示。
请参阅许可证以了解有关特定权限的详细信息
和限制。


# 模型训练解剖学

为了了解可以应用于改进模型训练效率的性能优化技术，
有助于熟悉在训练过程中如何利用GPU以及不同操作的计算强度如何变化。

让我们首先探索GPU利用率和模型训练运行的一个激励示例。为了演示，
我们需要安装一些库：

```bash
pip install transformers datasets accelerate nvidia-ml-py3
```

`nvidia-ml-py3`库允许我们监视Python内存中情景模型的内存使用情况。您可能熟悉终端中的`nvidia-smi`命令-此库允许直接在Python中访问相同的信息。

然后，我们创建一些虚拟数据：100到30000之间的随机令牌ID和分类器的二进制标签。
总共，我们得到512个长度为512的序列，并将它们存储在具有PyTorch格式的[`~datasets.Dataset`]中。

```py
>>> import numpy as np
>>> from datasets import Dataset

>>> seq_len, dataset_size = 512, 512
>>> dummy_data = {
...     "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
...     "labels": np.random.randint(0, 1, (dataset_size)),
... }
>>> ds = Dataset.from_dict(dummy_data)
>>> ds.set_format("pt")
```

为了打印GPU使用情况和训练运行的摘要统计信息，我们定义了两个辅助函数：

```py
>>> from pynvml import *

>>> def print_gpu_utilization():
...     nvmlInit()
...     handle = nvmlDeviceGetHandleByIndex(0)
...     info = nvmlDeviceGetMemoryInfo(handle)
...     print(f"GPU memory occupied: {info.used//1024**2} MB.")

>>> def print_summary(result):
...     print(f"Time: {result.metrics['train_runtime']:.2f}")
...     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
...     print_gpu_utilization()
```

让我们验证一下我们是否以空闲GPU内存开始：

```py
>>> print_gpu_utilization()
GPU memory occupied: 0 MB.
```

看起来不错：我们预期在加载任何模型之前，GPU内存没有被占用。如果在您的机器上不是这种情况，请确保停止使用GPU内存的所有进程。然而，并非所有的空闲GPU内存都可以被用户使用。当模型加载到GPU上时，内核也会被加载，这可能会占用1-2GB的内存。为了看到有多少，我们将一个小张量加载到GPU中，这将触发内核也被加载的过程。

```py
>>> import torch

>>> torch.ones((1, 1)).to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 1343 MB.
```

我们看到内核单独占用了1.3GB的GPU内存。现在让我们看看模型使用了多少空间。

## 加载模型

首先，我们加载`bert-large-uncased`模型。我们直接将模型权重加载到GPU上，以便我们可以检查权重本身使用了多少空间。

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 2631 MB.
```

我们可以看到模型权重单独占用了1.3GB的GPU内存。确切的数字取决于您使用的具体GPU。请注意，在更新的GPU上，模型有时可能需要更多的空间，因为权重是以优化的方式加载的，可以加速模型的使用。现在，我们还可以通过`nvidia-smi`命令行快速检查是否获得与之前相同的结果：

```bash
nvidia-smi
```

```bash
Tue Jan 11 08:58:05 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   37C    P0    39W / 300W |   2631MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      3721      C   ...nvs/codeparrot/bin/python     2629MiB |
+-----------------------------------------------------------------------------+
```

我们得到了与之前相同的数字，您还可以看到我们正在使用具有16GB内存的V100 GPU。因此，现在我们可以开始训练模型，并查看GPU内存消耗如何变化。首先，我们设置一些标准训练参数：

```py
default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
```

<Tip>

 如果您计划运行多个实验，为了正确清除实验之间的内存，请在实验之间重新启动Python内核。

</Tip>

## 原始训练的内存利用率

让我们使用[`Trainer`]训练器，以不使用任何GPU性能优化技术并且批次大小为4来训练模型：

```py
>>> from transformers import TrainingArguments, Trainer, logging

>>> logging.set_verbosity_error()

>>> training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
>>> trainer = Trainer(model=model, args=training_args, train_dataset=ds)
>>> result = trainer.train()
>>> print_summary(result)
```

```
时间：57.82
每秒样本数：8.86
GPU内存占用：14949 MB。
```

我们可以看到，即使是一个相对较小的批次大小几乎填满了GPU的整个内存。然而，更大的批次大小通常可以导致更快的模型收敛或更好的最终性能。因此，理想情况下，我们希望根据模型的需求而不是GPU的限制来调整批次大小。有趣的是，我们使用的内存远远超过了模型的大小。为了更好地理解为什么会出现这种情况，让我们看一下模型的操作和内存需求。

## 模型操作解剖学

Transformers架构包括以下3个主要操作组，按计算强度分组。

1. **张量收缩**

    线性层和多头注意力的组件都进行批量的**矩阵-矩阵乘法**。这些操作是训练transformer的计算最强大的部分。

2. **统计归一化**

    Softmax和层归一化比张量收缩更少计算强度，并涉及一个或多个**归约操作**，其结果然后通过映射应用。

3. **逐元素运算符**

    这些是剩余操作：**偏差、dropout、激活函数和残差连接**。这些是计算强度最低的操作。

这些知识在分析性能瓶颈时非常有用。

此摘要摘自[Data Movement Is All You Need：《2020年优化变压器的案例研究》](https://arxiv.org/abs/2007.00072)


## 模型内存解剖学

我们已经看到，训练模型使用的内存要比仅将模型放在GPU上多得多。这是因为在训练过程中有许多组件使用了GPU内存。GPU内存中的组件包括：

1. 模型权重
2. 优化器状态
3. 梯度
4. 用于梯度计算的前向激活保存
5. 临时缓冲区
6. 功能特定内存

使用AdamW在mixed precision下训练的典型模型每个模型参数需要18个字节加上激活内存。对于推理而言，没有优化器状态和梯度，因此可以减去这些。因此，对于mixed precision推理，每个模型参数需要6个字节，加上激活内存。

让我们看一下细节。

**模型权重：**

- 用于fp32训练的每个参数4字节
- 用于混合精度训练每个参数6字节（在内存中维护一个fp32模型和一个fp16模型）

**优化器状态：**

- 用于正常AdamW每个参数8字节（维护2个状态）
- 用于8位AdamW优化器（如[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)）每个参数2字节
- 用于带动量的SGD等优化器每个参数4字节（只维护一个状态）

**梯度**

- 用于fp32或混合精度训练的每个参数4字节（梯度始终以fp32保存）

**前向激活**

- 大小取决于许多因素，主要是序列长度、隐藏大小和批次大小。

这里有前向和反向函数传递和返回的输入和输出，以及保存用于梯度计算的前向激活。

**临时内存**

此外，还有各种临时变量，这些变量在计算完成后被释放，但是在这一时刻，这些变量可能需要额外的内存，并可能导致OOM。因此，在编码时，战略性地思考此类临时变量以及有时在不再需要时显式释放它们是至关重要的。

**功能特定内存**

然后，您的软件可能具有特殊的内存需求。例如，使用波束搜索生成文本时，软件需要维护输入和输出的多个副本。

**`forward`与`backward`执行速度**

对于卷积和线性层，反向执行比正向执行要慢大约2倍的FLOPS，这通常会导致速度较慢（有时更慢，因为反向中的尺寸 tend to be more awkward）。激活通常受带宽限制，一个激活在反向中读取的数据比正向中读取的数据多（例如，激活正向读取一次，写入一次；激活反向读取两次，gradOutput和forward的输出，并写入一次gradInput）。

正如您所看到的，有几个地方我们可以节省GPU内存或加速操作。现在，您已经了解了影响GPU利用率和计算速度的因素，请参阅[在单个GPU上进行高效训练的方法和工具](perf_train_gpu_one.md)文档页面，了解性能优化技术。