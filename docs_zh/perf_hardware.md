<!---
版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）许可；
除非符合许可，否则禁止使用此文件。
您可以在以下获得许可证的副本

     http://www.apache.org/licenses/LICENSE-2.0

请注意，此文件采用Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），在Markdown查看器中可能无法正确呈现。

-->


# 训练使用的自定义硬件

您用于运行模型训练和推理的硬件可能会对性能产生重大影响。关于GPU的深入了解，请务必查看Tim Dettmer出色的[博文](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)。

让我们看一下GPU设置的一些实用建议。

## GPU
当您训练更大的模型时，您基本上有以下三种选择：

- 更大的GPU
- 更多的GPU
- 更多的CPU和NVMe（由[DeepSpeed-Infinity](main_classes/deepspeed#nvme-support)卸载）

让我们从只有一个GPU的情况开始。

### 供电和散热

如果您购买了一款昂贵的高端GPU，请确保为其提供正确的电源和足够的散热。

**电源**：

一些高端消费级GPU卡具有2个，有时甚至3个PCI-E 8针电源插座。请确保将与插座数量相同的独立12V PCI-E 8针电缆插入卡中。不要使用同一电缆的两个分割部分（也称为猪尾电缆）。也就是说，如果GPU上有2个插槽，则希望PSU到卡的有2个PCI-E 8针电缆，而不是一个末端有2个PCI-E 8针连接器的电缆！否则，您将无法充分发挥卡的性能。

每个PCI-E 8针电源电缆都需要插入PSU侧的12V电轨，并能提供最多150W的功率。

其他一些卡可能使用PCI-E 12针连接器，这些连接器可以提供高达500-600W的功率。

低端卡可能使用6针连接器，其提供高达75W的功率。

此外，您需要拥有稳定电压的高端PSU。某些低质量的PSU可能无法为卡提供所需的稳定电压，使其以最佳状态运行。

当然，PSU还需要具备足够未使用的瓦特来供电给卡。

**散热**：

当GPU过热时，它将开始降频，无法提供完整的性能，甚至可能在过热时关闭。

很难确定在GPU严重负载时应该努力达到的最佳温度，但大致来说，在+80C以下的任何温度都是良好的，但更低的温度更好，70-75C或许是一个很好的范围。降频很可能会在84-90°C左右开始。但除了降低性能，持续的高温还可能缩短GPU的寿命。

下面让我们来看看在多个GPU时最重要的一个方面：连接性。

### 多GPU连接

如果使用多个GPU，卡之间的连接方式对总的训练时间会产生巨大影响。如果GPU在同一物理节点上，可以运行以下命令：

```
nvidia-smi topo -m
```

它会告诉您GPU的连接方式。在具有双GPU的机器上，并且这两个GPU是通过NVLink连接的，您可能会看到以下内容：

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

在另一台没有NVLink的机器上，我们可能会看到： 
```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

报告包括以下图例：

```
  X    = 自身
  SYS  = 穿越PCIe和NUMA节点之间的SMP互连的连接（例如，QPI/UPI）
  NODE = 穿越PCIe以及NUMA节点内的PCIe主机桥之间的互连的连接
  PHB  = 穿越PCIe以及PCIe主机桥（通常是CPU）的连接
  PXB  = 穿越多个PCIe桥接器的连接（而不穿越PCIe主机桥）
  PIX  = 穿越最多一个PCIe桥接器的连接
  NV#  = 穿越一组绑定的NVLink的连接
```

因此，第一个报告“NV2”告诉我们GPU之间是通过2个NVLink相互连接的，并且第二个报告“PHB”意味着我们具有典型的消费级PCIe+桥接器设置。

检查您的设置中使用的连接类型。其中一些将使卡之间的通信更快（例如NVLink），而另一些则会更慢（例如PHB）。

根据所使用的可扩展性解决方案的类型，连接速度可能会产生重大或次要影响。如果GPU需要很少进行同步，就像在DDP中一样，较慢的连接的影响将不那么重要。如果GPU需要频繁发送消息给彼此，就像在ZeRO-DP中一样，那么更快的连接速度变得非常重要，以实现更快的训练。

#### NVlink

[NVLink](https://en.wikipedia.org/wiki/NVLink)是由Nvidia开发的基于线缆的串行多通道近距离通信链接。

每一代新的NVLink都提供更快的带宽，例如以下是来自[Nvidia Ampere GA102 GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)的引用：

> 第3代NVLink®
> GA102 GPU使用NVIDIA的第三代NVLink接口，其中包括四个x4链路，
> 每个链路在两个GPU之间的每个方向上提供14.0625 GB/sec带宽。四个
> 链路在每个方向上提供56.25 GB/sec带宽，以及总共112.5 GB/sec的带宽
> 在两个GPU之间。两个RTX 3090 GPU可以使用NVLink进行SLI连接。
> （注意，不支持3路和4路SLI配置。）

因此，`nvidia-smi topo -m`输出中“NVX”报告中的较高的“X”越好。代数将取决于您的GPU架构。

我们来比较在一个小的wikitext样本上训练gpt2语言模型的执行情况。

结果如下：


| NVlink | 时间 |
| -----  | ---: |
| 是      | 101秒 |
| 否      | 131秒 |


可以看出，NVLink的训练速度快了约23%。在第二个基准测试中，我们使用`NCCL_P2P_DISABLE=1`告诉GPU不要使用NVLink。

以下是完整的基准测试代码和输出：

```bash
# 带NVLink的DDP

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# 无NVLink的DDP

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

硬件：2x TITAN RTX 24GB + 2个NVLink连接（`NV2`在`nvidia-smi topo -m`中）
软件：`pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`
