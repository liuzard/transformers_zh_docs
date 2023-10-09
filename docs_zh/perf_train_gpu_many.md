<!--版权 2022 The HuggingFace Team。版权所有。

根据Apache许可证2.0版（“许可证”）的规定，除非符合许可证的规定，否则不得使用此文件。

您可以从以下链接获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

但是，需要注意，此文件是Markdown格式的，但包含我们的文档生成器（类似于MDX的）的特定语法，可能无法在Markdown查看器中正确显示。

-->

# 在多个GPU上进行高效训练

当在单个GPU上训练速度过慢或模型权重无法适应单个GPU内存时，我们使用多个GPU进行训练。从一个GPU切换到多个GPU需要某种形式的并行处理，因为需要将工作分配给多个GPU。有几种技术可以实现并行处理，例如数据并行、张量并行或pipeline并行。然而，并不存在适用于所有情况的通用解决方案，最佳设置取决于您运行的硬件。尽管主要概念可能适用于任何其他框架，但本文重点介绍基于PyTorch的实现。

<Tip>

注意：在深入阅读以下部分（如多GPU或CPU训练）之前，建议先阅读[单个GPU部分](perf_train_gpu_one.md)中介绍的大多数策略（如混合精度训练或梯度累积），这些策略适用于一般的模型训练。

</Tip>

我们首先会详细讨论各种一维并行处理技术及其优缺点，然后再看它们如何组合成二维和三维并行处理以实现更快的训练速度，并支持更大的模型。还会介绍其他强大的替代方法。

## 概念

以下是本文稍后会详细介绍的主要概念的简要描述。

1. **数据并行（DataParallel，DP）** - 多次复制相同的设置，并为每个设置提供数据的一个片段。并行处理所有设置，并在每个训练步骤结束时进行同步。
2. **张量并行（TensorParallel，TP）** - 将每个张量分割成多个块，因此每个张量的分片都存储在其指定的GPU上。在处理过程中，每个分片在不同的GPU上独立且并行地处理，然后在步骤结束时同步结果。这被称为水平并行处理，因为分割发生在水平层面上。
3. **pipeline并行（PipelineParallel，PP）** - 将模型在垂直方向（层级级别）上划分到多个GPU上，以便只有一个或几个层位于单个GPU上。每个GPU并行处理流水线的不同阶段，并在小批次上工作。
4. **Zero Redundancy Optimizer（ZeRO）** - 与TP相似，对张量进行分片，但在前向或后向计算之前，整个张量会重新构建，因此模型不需要修改。它还支持各种卸载技术，以弥补有限的GPU内存。
5. **Sharded DDP** - 是ZeRO核心概念的另一个名称，由各种其他ZeRO实现使用。

在深入研究每个概念的细节之前，首先了解在大型基础架构上训练大型模型时的大致决策流程。

## 可扩展性策略

**⇨ 单节点/多GPU**
* 模型适用于单个GPU：

    1. DDP - 分布式DP
    2. ZeRO - 根据具体情况和配置可能更快

* 模型无法适应单个GPU：

    1. PP
    2. ZeRO
    3. TP

    使用NVLINK或NVSwitch的非常快速的节点内连接时，这三种方法几乎不会有区别，否则PP将比TP或ZeRO更快。TP的程度也可能有所不同。最好进行实验，找到特定设置的最佳方案。

    TP几乎总是在单个节点内使用。也就是说，TP size <= 每个节点的GPU数。

* 最大的层无法适应单个GPU：

    1. 如果不使用ZeRO，则必须使用TP，因为仅使用PP无法容纳。
    2. 使用ZeRO时，请参考上述“单个GPU”部分中的相同条目


**⇨ 多节点/多GPU**

* 当您具有快速的节点间连接时：

    1. ZeRO - 因为它几乎不需要对模型进行修改
    2. PP+TP+DP - 通信较少，但需要对模型进行大量更改

* 当您的节点间连接较慢且GPU内存仍然不够时：

    1. DP+PP+TP+ZeRO-1



## 数据并行处理

大多数仅具有2个GPU的用户已经通过`DataParallel`（DP）和`DistributedDataParallel`（DDP）获得了加速的训练速度，它们几乎是可以轻松使用的。这是PyTorch的内置特性。请注意，在一般情况下，建议使用DDP，因为它的维护性更好，并且适用于所有模型，而DP可能对某些模型不适用。 [PyTorch文档](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)本身推荐使用DDP。

### DP vs DDP

`DistributedDataParallel`（DDP）通常比`DataParallel`（DP）更快，但并非总是如此：
* 虽然DP基于Python线程，但DDP基于多进程 - 因此它没有Python线程的限制，如全局解释锁（GIL）
* 另一方面，GPU卡之间的连接速度较慢可能会导致DDP的实际速度较慢

以下是两种模式之间的GPU间通信开销的主要差异：

[DDP](https://pytorch.org/docs/master/notes/ddp.html):

- 在启动时，主进程将模型从gpu 0复制到其他gpu
- 然后对于每个批次：
   1. 每个gpu直接处理自己的小批量数据
   2. 在`backward`期间，一旦本地梯度准备就绪，它们就会在所有进程之间进行平均

[DP](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html):

对于每个批次：
   1. gpu 0读取批量数据，然后将小批量发送到每个gpu
   2. 将最新的模型从gpu 0复制到每个gpu
   3. 运行`forward`并将每个gpu的输出发送到gpu 0，计算损失
   4. 将来自gpu 0的损失散布到所有gpu，运行`backward`
   5. 将每个gpu的梯度发送到gpu 0并平均这些梯度

DDP每批次只执行1次通信，而DP每批次执行5次不同的数据交换。

DP通过Python线程在进程内复制数据，而DDP通过[torch.distributed](https://pytorch.org/docs/master/distributed.html)复制数据。

在DP中，gpu 0比其他gpu执行更多的工作，因此导致gpu的利用率不足。

您可以在多台机器上使用DDP，但对于DP来说不是这样。

DP和DDP还有其他差异，但与本讨论无关。

如果您想深入了解这两种模式，强烈推荐阅读这篇[文章](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)，因为它有很棒的图表，包含各种硬件的多个基准测试和分析器输出，并解释了您可能需要了解的所有细微差别。

让我们来看一个实际的基准测试：

| 类型   | NVlink | 时间 |
| :----- | -----  | ---: |
| 2:DP   | 是     | 110秒 |
| 2:DDP  | 是     | 101秒 |
| 2:DDP  | 否     | 131秒 |


分析：

在此示例中，DP比具有NVlink的DDP慢约10％，但比没有NVlink的DDP快约15％。

实际差异取决于每个GPU需要与其他GPU同步的数据量 - 同步的数据越多，慢速链接会减慢总运行时间。

以下是完整的基准测试代码和输出：

使用`NCCL_P2P_DISABLE=1`禁用了对应基准测试中的NVLink功能。

```

# DP
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 110.5948, 'train_samples_per_second': 1.808, 'epoch': 0.69}

# DDP w/ NVlink
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVlink
rm -r /tmp/test-clm; NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

硬件：2个TITAN RTX 24GB + 带有2个NVLink的NVlink（在`nvidia-smi topo -m`中为`NV2`）
软件：`pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`

## ZeRO数据并行处理

ZeRO数据并行处理（ZeRO-DP）在此[博客文章](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)中的下图中进行了描述：
![DeepSpeed-Image-1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png)

可能很难理解，但实际上这个概念非常简单。这只是通常的“DataParallel”（DP），但是，与其复制完整的模型参数、梯度和优化器状态不同，每个GPU仅存储其中的一部分。然后在运行时，当需要给定层的完整层参数时，所有GPU将同步以互相提供它们缺少的部分 - 就是这样。

考虑以下具有3层的简单模型，其中每层有3个参数：
```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```
`La`层具有`a0`、`a1`和`a2`参数。

如果我们有3个GPU，Sharded DDP (= Zero-DP) 将模型分割到3个GPU上如下所示：

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

在某种程度上，这与张量并行处理非常相似，稍后将对此进行讨论。如果您想象一下典型的DNN图表，这是因为它分割/分片了每个层的权重，与垂直模型并行处理不同。

现在，每个GPU将获得与DP相同的小批量数据：
```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

输入未经修改 - 它们认为它们将由常规模型处理。

首先，输入到达层La。

让我们专注于GPU0：x0需要a0、a1和a2参数来进行前向路径，但GPU0只有a0 - 它从GPU1接收a1，并从GPU2接收a2，将模型的所有部分集中到一起。

同时，并行处理，GPU1获得小批量x1，它只有a1，但需要a0和a2参数，因此它从GPU0和GPU2获取这些参数。

GPU2也发生同样的情况，它获取输入x2。它从GPU0和GPU1获取了a0和a1，并利用a2重构了完整的张量。

所有3个GPU都获取了重构的完整张量，并进行前向传播。

计算完成后，不再需要的数据被丢弃 - 仅在计算期间使用。重建通过预取方式进行，非常高效。

整个过程对Lb层和Lc层也会重复，并在后向Lc -> Lb -> La时重复。

对我来说，这听起来像一种高效的组合背包分配权重的策略：

1. A人负责帐篷
2. B人负责火炉
3. C人负责斧头

现在每晚他们彼此分享自己拥有的物品，并从他人那里获得自己没有的物品，早晨他们整理好自己分配的物品，然后继续前进。这就是Sharded DDP / Zero DP。

将此策略与每个人都必须自己携带帐篷、火炉和斧头的简单策略进行比较，显然后者效率更低。这就是PyTorch中的DataParallel（DP和DDP）。

在阅读有关此主题的文献时，您可能会遇到以下同义词：Sharded、Partitioned。

如果仔细观察ZeRO是如何划分模型的权重的，它看起来与稍后将会讨论的张量并行处理非常相似。这是因为它分割/分片了每个层的权重，不同于垂直模型并行处理。

实现方式：

- [DeepSpeed](https://www.deepspeed.ai/features/#the-zero-redundancy-optimizer) ZeRO-DP 的 stages 1+2+3
- [`transformers`集成](main_classes/trainer#trainer-integrations)

## 单纯的模型并行处理（垂直）和pipeline并行处理

单纯的模型并行处理（MP）是将模型的层组划分到多个GPU上的方式。机制相对简单 - 使用`to()`方法将所需的层设为所需的设备，这样，每当数据进出这些层时，数据就会切换到与层相同的设备，并保持其余部分不变。

我们将其称为垂直MP，因为如果您记得大多数模型是如何绘制的，我们是垂直地切分图层。例如，如果以下图表显示了一个具有8个层的模型：

```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        gpu0                 gpu1
```

我们只是在垂直方向上将其切成两部分，将层0-3放在GPU0上，将层4-7放在GPU1上。

现在，当数据从层0到1，从1到2和从2到3传输时，它与正常模型一样。但是当数据需要从层3传递到层4时，它需要从GPU0传输到GPU1，这就引入了通信开销。如果参与的GPU位于同一个计算节点上（例如同一物理机器），则此复制非常快速，但如果GPU位于不同的计算节点上（例如多台机器），则通信开销可能会显着增加。

然后，层4到5到6到7就像正常模型一样，当第7层完成时，通常需要将数据发送回第0层，其中包含标签（或者将标签发送到最后一层）。现在可以计算损失并使优化器发挥作用。

问题：
- 这个被称为“天真”的MP的主要缺陷是，除了一个GPU之外的所有GPU在任何给定的时刻都处于空闲状态。所以如果使用了4个GPU，它几乎等同于将单个GPU的内存扩大4倍，并忽略其他硬件。此外，还需要在设备之间复制数据的开销。因此，使用天真的MP，4张6GB的显卡可以容纳与1张24GB的显卡相同的大小，只是后者的训练速度更快，因为它没有数据复制开销。但是，举个例子，如果你有40GB的显卡并且需要适应一个45GB的模型，你可以使用4张40GB的显卡（但由于梯度和优化器状态的原因，可能勉强）。
- 共享嵌入可能需要在GPU之间来回复制。

流水线并行处理（PP）与天真的MP几乎相同，但它解决了GPU空闲问题，通过将传入的批次分成微批量并人为地创建一个流水线，允许不同的GPU同时参与计算过程。

以下是来自[GPipe论文](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)的插图，顶部是天真的MP，底部是PP：

![mp-pp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png)

从底部的图表中很容易看出，PP具有较少的空闲区域，GPU空闲的部分被称为“bubble”。

图表的两部分都显示了一个4并行的并行度。也就是说，有4个GPU参与了流水线。所以有着4个流水线阶段F0、F1、F2和F3的前向路径，然后有着反向路径B3、B2、B1和B0的返回逆序。

PP引入了一个新的超参数来调整，它是`chunks`，它定义了有多少数据块以序列方式通过同一pipeline阶段发送。例如，在底部的图表中可以看到`chunks=4`。GPU0在块0、1、2和3（F0,0、F0,1、F0,2、F0,3）上执行相同的前向路径，然后它等待其他GPU进行工作，只有当其他GPU的工作开始完成时，GPU0才开始工作，对块3、2、1和0（B0,3、B0,2、B0,1、B0,0）执行反向路径。

请注意，从概念上讲，这与梯度积累步骤（GAS）是相同的概念。PyTorch使用`chunks`，而DeepSpeed将相同的超参数称为GAS。

由于块的存在，PP引入了微批（MBS）的概念。DP将全局数据批次大小划分为小批次，因此，如果您的DP度为4，全局批次大小为1024，划分为4个大小为256的小批次（1024/4）。如果`chunks`（或GAS）的数量为32，那么我们就得到了一个微批次大小为8（256/32）。每个流水线阶段每次只处理一个微批次。

要计算DP+PP配置的全局批次大小，我们需要执行以下计算：`mbs*chunks*dp_degree`（`8*32*4=1024`）。

让我们回到图表。

当`chunks=1`时，你得到的是天真的MP，非常低效。当`chunks`值非常大时，你得到的是非常小的微批次大小，这也可能不是非常高效。因此，必须进行实验，找到能够实现最高GPU利用率的值。

虽然图表显示，在流水线的最后一个“前向”阶段必须等待“后向”完成之前，存在着一个无法并行化的“死区”的泡沫，但是找到适合`chunks`的最佳值的目的是为了实现高并发GPU利用率，从而减小泡沫的大小。

解决方案分为两组-传统的流水线API和更现代的解决方案，可以使最终用户的操作变得更简单。

传统的流水线API解决方案：
- PyTorch
- DeepSpeed
- Megatron-LM

现代解决方案：
- Varuna
- Sagemaker

传统的流水线API解决方案存在的问题：
- 必须对模型进行相当大的修改，因为Pipeline要求将模块的常规流程重写为相同的`nn.Sequential`序列，这可能需要对模型的设计进行更改。
- 目前，流水线接口非常受限制。如果在流水线的最初阶段传递了一堆python变量，那么你需要找到一个解决方法。目前，流水线接口只接受单个张量或一组张量作为输入和输出。这些张量的第一个维度必须是批次大小，因为流水线将把小批次划分为微批次。可能会在这里进行改进https://github.com/pytorch/pytorch/pull/50693
- 在pipeline阶段的条件控制流是不可能的-例如，Encoder-Decoder模型（如T5）需要特殊的解决方法来处理条件编码器阶段。
- 必须安排每个层，使得一个模型的输出成为另一个模型的输入。

我们尚未尝试过Varuna和SageMaker，但它们的论文报告称它们已经解决了上述问题列表，并且对用户模型的更改要求更小。

实现：
- [PyTorch](https://pytorch.org/docs/stable/pipeline.html)（初始支持在pytorch-1.8中，并逐步在1.9中改进，1.10更进一步）。一些[示例](https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/pipeline/pipe.py)
- [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)有一个内部实现-没有API。
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972) - 这是一个专有的解决方案，只能在AWS上使用。
- [OSLO](https://github.com/tunib-ai/oslo) - 这是基于Hugging Face Transformers实施的。

🤗 Transformers状态：截至目前为止，没有任何模型支持完全的PP。GPT2和T5模型支持天真的MP。主要的障碍是无法将模型转换为`nn.Sequential`并且所有的输入都是Tensors。这是因为目前的模型包括了很多使得转换非常复杂的功能，并且需要去除这些功能才能实现转换。

其他方法：

DeepSpeed，Varuna和SageMaker使用了[Interleaved Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html)的概念
![interleaved-pipeline-execution](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-sagemaker-interleaved-pipeline.png)

这里的泡沫（空闲时间）通过优先级后向传递来进一步减少。

Varuna试图通过使用仿真来发现最有效的调度方法，进一步改进了调度。

OSLO基于Hugging Face Transformers实施了基于流水线的并行处理，而没有进行`nn.Sequential`转换。

张量并行处理

在张量并行处理中，每个GPU仅处理张量的一个切片，并仅在需要整个张量的操作中进行整合。

在这个部分，我们使用[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)论文中的概念和图表：[Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)。

任何transformer的主要构建块都是一个完全连接的 `nn.Linear` ，后面跟着非线性激活 `GeLU`。

按照Megatron论文的表示方法，我们可以将其点积部分写作 `Y = GeLU(XA)` ，其中 `X` 和 `Y` 是输入和输出向量，`A` 是权重矩阵。

如果我们以矩阵形式看计算过程，很容易看出矩阵乘法如何在多个GPU之间进行拆分：
![Parallel GEMM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_gemm.png)

如果我们将权重矩阵 `A` 按列在`N`个GPU上拆分，并在并行进行矩阵乘法 `XA_1` 到 `XA_n`，那么我们将得到 `N` 个输出向量 `Y_1, Y_2, ..., Y_n` ，它们可以独立地输入到 `GeLU` 中：
![independent GeLU](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-independent-gelu.png)

使用这个原理，我们可以更新任意层数的MLP，而不需要在GPU之间进行任何同步，直到最后需要从分片中重构输出向量。Megatron-LM论文的作者为此提供了一个有用的插图：
![parallel shard processing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_shard_processing.png)

并行化多头注意力层更加简单，因为它们已经本质上是并行的，因为有多个独立的头！
![parallel self-attention](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_self_attention.png)

特殊考虑：张量并行处理需要非常快速的网络，因此不建议在超过一个节点上进行张量并行处理。实际上，如果一个节点有4个GPU，则最高的张量并行处理度是4。如果你需要8个张量并行处理度，你需要使用至少有8个GPU的节点。

这一部分基于更详细的[TP概述](https://github.com/huggingface/transformers/issues/10321#issuecomment-783543530)。

SageMaker将TP与DP结合起来，以获得更高效的处理。

名称替代品：
- DeepSpeed将其称为[张量切片](https://www.deepspeed.ai/features/#model-parallelism)

实现：
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)有一个内部实现，因为它非常具体。
- [parallelformers](https://github.com/tunib-ai/parallelformers)（目前只有推理）。
- [SageMaker](https://arxiv.org/abs/2111.05972) - 这是一个专有的解决方案，只能在AWS上使用。
- [OSLO](https://github.com/tunib-ai/oslo)有基于Transformers的张量并行处理实现。

🤗 Transformers状态：
- 核心：尚未在核心实现中实施
- 但是，如果你想进行推理，[parallelformers](https://github.com/tunib-ai/parallelformers)提供对大多数模型的支持。所以在这个在核心实现之前，你可以使用他们的解决方案。希望将来能支持训练模式。
- Deepspeed-Inference还通过CUDA核心实施支持了BERT、GPT-2和GPT-Neo模型，详细信息请参阅[此处](https://www.deepspeed.ai/tutorials/inference-tutorial/)

DP+PP

下面是来自DeepSpeed [流水线教程](https://www.deepspeed.ai/tutorials/pipeline/)的图表，展示了如何将DP与PP结合起来。

![dp-pp-2d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png)

在这里，重要的是要看到DP秩0看不到GPU2，而DP秩1看不到GPU3。对于DP而言，只有0号和1号GPU，它将数据传递给它们，就好像只有2个GPU一样。GPU0秘密地把它的负载转移到GPU2上，使用PP。GPU1通过将GPU3援助到自己的负载中来做同样的事情。

由于每个维度至少需要2个GPU，所以这里至少需要4个GPU。

实现：
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/tunib-ai/oslo)

🤗 Transformers状态：尚未实现

DP+PP+TP

为了获得更高效的训练，使用了一个3D并行度，其中将PP与TP和DP相结合。这可以在下图中看到。

![dp-pp-tp-3d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png)

这个图是来自一篇博客文章[3D parallelism: Scaling to trillion-parameter models](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)，这也是一篇很好的阅读材料。

由于每个维度至少需要2个GPU，所以这里至少需要8个GPU。

实现：
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - DeepSpeed还包括了更高效的DP，称为ZeRO-DP。
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/tunib-ai/oslo)

🤗 Transformers状态：尚未实现，因为我们没有PP和TP。

ZeRO DP+PP+TP

DeepSpeed的主要特点之一是ZeRO，它是DP的超大规模扩展。已经在[ZeRO Data Parallelism](#zero-data-parallelism)中讨论过它。通常它是一个独立的特性，不需要PP或TP。但它可以与PP和TP结合使用。

当ZeRO-DP与PP（和可选的TP）结合使用时，通常只启用ZeRO阶段1（优化器分片）。

虽然在流水线并行处理中理论上可以使用ZeRO阶段2（梯度分片），但它会对性能产生不好的影响。每个微批次之后需要进行额外的reduce-scatter集体操作来聚合梯度，这会增加潜在的通信开销。由于流水线并行处理使用的是小微批次，并且注重的是尝试平衡算术强度（微批次大小）和最小化pipeline泡沫（微批次数量），因此这些通信开销会带来困扰。

此外，由于层比正常情况下少，内存节省并不会很大。PP已经将梯度大小缩小了 `1/PP`，所以除了DP以外，梯度分片的节省效果并不明显。

ZeRO阶段3也不是一个好选择，原因与上述相同-它需要更多的节点间通信。

由于我们有ZeRO，另一个好处是ZeRO-Offload。由于这是阶段1，优化器状态可以卸载到CPU上。

实现：
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)和[来自BigScience的Megatron-Deepspeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)，它是前一个存储库的派生版本。
- [OSLO](https://github.com/tunib-ai/oslo)

重要论文：

- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](
https://arxiv.org/abs/2201.11990)

🤗 Transformers状态：尚未实现，因为我们没有PP和TP。

FlexFlow

[FlexFlow](https://github.com/flexflow/FlexFlow)还以稍微不同的方式解决了并行化问题。

论文：["Beyond Data and Model Parallelism for Deep Neural Networks" by Zhihao Jia, Matei Zaharia, Alex Aiken](https://arxiv.org/abs/1807.05358)

它在样本-运算符-属性-参数之间执行一种4D并行化。

**样本（Sample）**

我们以长度为512的序列为例，取10个批次。如果我们按样本维度将它们并行化到2个设备上，那么10 x 512就会变成5 x 2 x 512。

**操作符（Operator）**

如果我们进行层归一化，先计算标准差，再计算均值，然后对数据进行归一化。操作符并行化允许在计算标准差和均值时同时进行计算。所以如果我们按操作符维度将它们并行化到两个设备（cuda:0, cuda:1）上，首先我们将输入数据复制到两个设备上，然后cuda:0计算标准差，cuda:1同时计算均值。

**属性（Attribute）**

我们有10个批次，每个批次长度为512。如果我们按属性维度将它们并行化到2个设备上，10 x 512就会变成10 x 2 x 256。

**参数（Parameter）**

这类似于张量模型并行化或者朴素层级模型并行化。

![flex-flow-soap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-flexflow.jpeg)

这个框架的重要性在于它根据资源如（1）GPU/TPU/CPU与（2）RAM/DRAM与（3）快速内部连接/慢速外部连接等进行自动优化，算法决定在哪里使用并行化。

其中一个非常重要的方面是，FlexFlow是为优化具有静态和固定工作负载的DNN并行化而设计的，因为具有动态行为的模型可能在迭代中更喜欢不同的并行化策略。

因此，这个框架非常有吸引力——它在选择的集群上运行30分钟的模拟，并得出最佳策略来利用这个特定环境。如果您添加/删除/替换任何部分，它将运行并重新优化计划。然后您可以进行训练。不同的设置将有自己的定制优化。

🤗 Transformer状态：尚未集成。我们已经通过[transformers.utils.fx](https://github.com/huggingface/transformers/blob/master/src/transformers/utils/fx.py)追踪我们的模型FX，这是FlexFlow的先决条件，因此需要有人弄清楚如何使FlexFlow与我们的模型配合工作。

## 在什么时候采用哪种策略

下面是一个关于在什么时候采用哪种并行化策略的大致概述。每个列表中的第一种策略通常更快。

**⇨ 单GPU**

* 模型适合单个GPU：

    1. 正常使用

* 模型不适合单个GPU：

    1. ZeRO + 卸载CPU和可选的NVMe
    2. 如果最大层无法适应单个GPU，则添加Memory Centric Tiling（详情请参阅下文）

* 最大层无法适应单个GPU：

1. ZeRO - 启用[内存中心平铺](https://deepspeed.readthedocs.io/en/latest/zero3.html#memory-centric-tiling)（MCT）。它允许您通过自动分割层并顺序执行它们来运行任意大的层。MCT减少了在GPU上活动的参数数量，但不会影响激活内存。鉴于这种需求目前非常少见，需要用户手动覆盖`torch.nn.Linear`。

**⇨ 单节点/多GPU**

* 模型适合单个GPU：

    1. DDP - 分布式DP
    2. ZeRO - 根据情况和使用的配置可能更快

* 模型不适合单个GPU：

    1. PP
    2. ZeRO
    3. TP

    如果有NVLINK或NVSwitch的非常快的节点内连接，则这三个策略应该基本上相当，在没有这些连接的情况下，PP将比TP或ZeRO更快。TP的程度也可能有所不同。最好进行实验，找到适合您特定设置的优胜者。

    TP几乎总是在单个节点内使用。即TP大小<=每个节点的GPU数。

* 最大层无法适应单个GPU：

    1. 如果不使用ZeRO，则必须使用TP，因为单独使用PP无法适应。
    2. 对于ZeRO，参见上面"单GPU"的相同条目

**⇨ 多节点/多GPU**

* 当您具有快速的节点间连接时：

    1. ZeRO - 因为它对模型几乎没有修改要求
    2. PP+TP+DP - 通信较小，但需要对模型进行大规模更改

* 当您具有较慢的节点间连接且仍然具有较低的GPU内存时：

    1. DP+PP+TP+ZeRO-1