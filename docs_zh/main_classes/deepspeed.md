<!--
版权所有2020 The HuggingFace团队。版权所有。

根据Apache许可证第2版（“许可证”）的规定，除非符合许可证，否则禁止使用此软件。
您可以在下面的链接找到许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于基“原样”发布，无任何明示或暗示保证或条件。请参阅许可证以获取
具体约束和限制的特定语言。

⚠️请注意，此文件是以Markdown格式编写的，但包含了我们doc-builder（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正常显示。

-->

# DeepSpeed集成

[DeepSpeed](https://github.com/microsoft/DeepSpeed)实现了[ZeRO论文](https://arxiv.org/abs/1910.02054)中描述的所有内容。目前，它提供了对以下功能的完全支持：

1. 优化器状态分区（ZeRO阶段1）
2. 梯度分区（ZeRO阶段2）
3. 参数分区（ZeRO阶段3）
4. 自定义混合精度训练处理
5. 一系列基于快速CUDA扩展的优化器
6. 针对CPU和NVMe的ZeRO-Offload

[ZeRO-Offload](https://arxiv.org/abs/2101.06840)有自己的专用论文。NVMe支持在论文[ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)中进行了描述。

DeepSpeed ZeRO-2主要用于训练，因为它的特性对推理没有用处。

DeepSpeed ZeRO-3也可以用于推理，因为它允许在多个GPU上加载大型模型，这在单个GPU上是不可能的。

🤗 Transformers通过2种方式集成[DeepSpeed](https://github.com/microsoft/DeepSpeed)：

1. 通过[`Trainer`]集成核心DeepSpeed功能。这是一种一切都为您完成的集成方式-只需提供自定义配置文件或使用我们的模板，不需要做其他事情。本文档的大部分内容都集中在此功能上。
2. 如果您不使用[`Trainer`]，而是要使用自己集成了DeepSpeed的自定义Trainer，核心功能函数（例如`from_pretrained`和`from_config`）将包含DeepSpeed的关键部分集成，如ZeRO阶段3及更高级别的`zero.Init`。要使用此功能，请阅读有关[非Trainer的DeepSpeed集成](#nontrainer-deepspeed-integration)的文档。

集成的内容：

训练：

1. DeepSpeed ZeRO训练与ZeRO阶段1、2和3以及ZeRO-Infinity（CPU和NVME offload）完全兼容。

推理：

1. DeepSpeed ZeRO推理支持ZeRO阶段3和ZeRO-Infinity。它使用与训练相同的ZeRO协议，但不使用优化器和学习率调度器，只有阶段3与推理相关。有关更多详情，请参阅[zero-inference](#zero-inference)。

还有DeepSpeed推理-这是一种完全不同的技术，它使用Tensor Parallelism而不是ZeRO（即将推出）。


<a id='deepspeed-trainer-integration'></a>


## 通过Trainer集成DeepSpeed


<a id='deepspeed-installation'></a>

### 安装

通过pypi安装库：

```bash
pip install deepspeed
```

或通过`transformers`的`extras`安装：

```bash
pip install transformers[deepspeed]
```

或在[DeepSpeed的GitHub页面](https://github.com/microsoft/deepspeed#installation)查找更多详细信息和[高级安装](https://www.deepspeed.ai/tutorials/advanced-install/)。

如果您仍然在构建方面遇到困难，请首先确保阅读[CUDA扩展安装说明](trainer#cuda-extension-installation-notes)。

如果您没有预先构建扩展，并且依赖于运行时构建它们，即使尝试了上述所有解决方案仍然无法成功，则下一步尝试应该是在安装它们之前构建模块。

要进行DeepSpeed的本地构建：

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

如果您打算使用NVMe offload，您还需要在上述指令中包括`DS_BUILD_AIO=1`（同时还需在系统范围内安装*libaio-dev*）。

编辑`TORCH_CUDA_ARCH_LIST`以插入您打算使用的GPU卡的架构代码。假设您的所有卡都是相同的，您可以通过以下方式获取架构：

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

因此，如果您获得`8, 6`，那么请使用`TORCH_CUDA_ARCH_LIST="8.6"`。如果您有多个不同的卡，可以像这样列出所有卡：`TORCH_CUDA_ARCH_LIST="6.1;8.6"`。

如果您需要在多台机器上使用相同的设置，请下载二进制轮子：

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
python setup.py build_ext -j8 bdist_wheel
```

它将生成类似于`dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`的东西，您现在可以在本地或任何其他机器上安装它为`pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`。

再次，请确保调整`TORCH_CUDA_ARCH_LIST`以适应目标架构。

您可以在[此处](https://developer.nvidia.com/cuda-gpus)找到NVIDIA GPU的完整列表以及其对应的**计算能力**（在此上下文中与架构相同）。

您可以使用以下命令检查PyTorch构建时所使用的架构：

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

以下是如何找出已安装的GPU之一的架构的示例。例如，对于GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
print(torch.cuda.get_device_properties(torch.device('cuda')))"
```

如果输出结果是：

```bash
_CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)
```

则您知道此卡的架构是`8.6`。

您也可以完全不使用`TORCH_CUDA_ARCH_LIST`，然后构建程序将自动查询构建所使用GPU的架构。这可能与目标机器上的GPU匹配，也可能不匹配，这就是明确指定所需架构的最佳方式。

如果尝试了所有建议的解决办法后仍然遇到构建问题，请继续使用[Deepspeed](https://github.com/microsoft/DeepSpeed/issues)的GitHub Issue。


<a id='deepspeed-multi-gpu'></a>

### 多GPU部署

要部署DeepSpeed集成，请调整[`Trainer`]命令行参数，包括一个新的参数`--deepspeed ds_config.json`，其中`ds_config.json`是DeepSpeed配置文件，如[此处](https://www.deepspeed.ai/docs/config-json/)所述。文件命名由您决定。
建议使用DeepSpeed的`add_config_arguments`实用程序来向您的代码中添加必要的命令行参数。
有关更多信息，请参阅[DeepSpeed的参数解析](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing)文档。

您可以在此处继续使用pytorch启动器：

```bash
torch.distributed.run --nproc_per_node=2 your_program.py <normal cl args> --deepspeed ds_config.json
```

或使用`deepspeed`提供的启动器：

```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json
```

如您所见，这些参数不同，但对于大多数需求，其中任何一个都可以工作。有关多个节点和GPU配置的详细信息，请参见[此处](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)。

当您使用`deepspeed`启动器并且希望使用所有可用的GPU时，可以省略`--num_gpus`标志。

下面是在DeepSpeed上使用所有可用GPU运行`run_translation.py`的示例：

```bash
deepspeed examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

请注意，在DeepSpeed文档中，您可能会看到`--deepspeed --deepspeed_config ds_config.json`-即两个DeepSpeed相关的参数，但为了简单起见，并且已经有太多参数要处理，我们将两者合并为一个参数。

有关一些实际用法示例，请参见[此处](https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400)的内容。


<a id='deepspeed-one-gpu'></a>

### 单GPU部署

要使用单个GPU部署DeepSpeed，请调整以下[`Trainer`]命令行参数：

```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

这与多个GPU的情况几乎相同，但在这里我们通过`--num_gpus=1`明确告诉DeepSpeed仅使用一个GPU。默认情况下，DeepSpeed会使用给定节点上的所有GPU。如果开始时只有1个GPU，则不需要此参数。以下文档中讨论了启动器选项。

为什么要使用只有一个GPU的DeepSpeed？

1. 它具有ZeRO-offload功能，可以将一些计算和内存委派给主机的CPU和RAM，从而为模型的需求留下更多的GPU资源-例如更大的批次大小，或启用通常无法容纳的非常大的模型。
2. 它提供了智能的GPU内存管理系统，最小化内存碎片化，这样再次可以适应更大的模型和数据批次。

虽然我们将在接下来的节中详细讨论配置，但要在具有一个GPU的DeepSpeed上获得巨大改进的关键是在配置文件中至少有以下配置：

```json
{
  "zero_optimization": {
     "stage": 2,
     "offload_optimizer": {
         "device": "cpu",
         "pin_memory": true
     },
     "allgather_partitions": true,
     "allgather_bucket_size": 2e8,
     "reduce_scatter": true,
     "reduce_bucket_size": 2e8,
     "overlap_comm": true,
     "contiguous_gradients": true
  }
}
```

它启用了优化器卸载和其他一些重要功能。您可以根据需要尝试不同的缓冲区大小，有关详细讨论，请参见下面的讨论。

有关此类部署的实际用法示例，请参见[此处](https://github.com/huggingface/transformers/issues/8771#issuecomment-759176685)。

您还可以尝试使用描述文件中进一步解释的带有CPU和NVMe offload的ZeRO-3。


<a id='deepspeed-multi-node'></a>

### 多节点部署

本节中的信息不仅适用于DeepSpeed集成，还适用于任何多节点程序。但是DeepSpeed始终提供了一个`deepspeed`启动器，它比其他启动器更容易使用，除非您处于SLURM环境中。

在本节的持续时间中，让我们假设您有2个拥有8个GPU的节点。您可以通过`ssh hostname1`访问第一个节点，通过`ssh hostname2`访问第二个节点，并且两者必须能够通过本地ssh在没有密码的情况下相互访问。当然，您需要将这些主机（节点）名称重新命名为您使用的实际主机名称。

#### torch.distributed.run启动器


例如，要使用`torch.distributed.run`，您可以执行以下操作：

```bash
python -m torch.distributed.run --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

您必须ssh到每个节点并在每个节点上运行相同的命令！不用着急，启动器会等待直到两个节点同步。

有关更多信息，请参见[torchrun](https://pytorch.org/docs/stable/elastic/run.html)。顺便说一句，这也是几个PyTorch版本前替代了`torch.distributed.launch`的启动器。


#### deepspeed启动器

要改用`deepspeed`启动器，您首先必须创建一个`hostfile`文件：

```
hostname1 slots=8
hostname2 slots=8
```
然后您可以启动它：

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
your_program.py <normal cl args> --deepspeed ds_config.json
```

与`torch.distributed.run`启动器不同，`deepspeed`将自动在两个节点上启动此命令！

有关更多信息，请参见[资源配置（多节点）](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)。


#### 在SLURM环境中启动

在SLURM环境中，可以使用以下方法。以下是一个slurm脚本`launch.slurm`，您需要根据特定的SLURM环境对其进行调整。

```bash
#SBATCH --job-name=test-nodes        # 名称
#SBATCH --nodes=2                    # 节点数
#SBATCH --ntasks-per-node=1          # 至关重要 - 每个节点上只有1个任务分发！
#SBATCH --cpus-per-task=10           # 每个任务的核数
#SBATCH --gres=gpu:8                 # GPU数
#SBATCH --time 20:00:00              # 最大执行时间（HH:MM:SS）
#SBATCH --output=%x-%j.out           # 输出文件名

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

只需安排它运行：
```bash
sbatch launch.slurm
```

`srun`会负责同时在所有节点上启动程序。


#### 使用非共享文件系统

默认情况下，DeepSpeed假设多节点环境使用共享存储。如果不是这种情况，而且每个节点只能看到本地文件系统，则需要调整配置文件，包含以下设置的[`checkpoint`_section](https://www.deepspeed.ai/docs/config-json/#checkpoint-options)：

```json
{
  "checkpoint": {
    "use_node_local_storage": true
  }
}
```

另外，您还可以使用[`Trainer`]的`--save_on_each_node`参数，上述配置将自动为您添加。

### 在笔记本中的部署
作为脚本运行笔记本单元格的问题在于没有常规的`deepspeed`启动器可以依赖，因此在某些设置下，我们不得不模拟该过程。

如果您只使用一个GPU，以下是在笔记本中调整训练代码以使用DeepSpeed的方法：

```python
# DeepSpeed要求即使在只使用一个进程时也需要一个分布式环境
# 这在笔记本中模拟了一个启动器
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# 现在继续正常操作，并传递deepspeed配置文件
training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

注意：`...`代表您传递给函数的正常参数。

如果您要使用多个GPU，您必须使用多进程环境才能使DeepSpeed工作。也就是说，您必须使用启动器来实现这个目的，而不能通过模拟开始部分中介绍的分布式环境来完成。

如果您想要在笔记本中在当前目录中即时创建配置文件，可以使用以下专用单元格：

```python no-style
%%bash
cat <<'EOT' > ds_config_zero3.json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT
```

如果训练脚本位于普通文件中而不是笔记本单元格中，您可以从单元格中使用shell正常启动`deepspeed`。例如，要使用`run_translation.py`启动它，您可以使用以下命令：

```python no-style
!git clone https://github.com/huggingface/transformers
!cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...
```

或者使用`%%bash`魔术，您可以编写一个多行代码供shell程序运行：

```python no-style
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```

在这种情况下，您不需要使用开始部分中提供的任何代码。

注意：虽然`%%bash`魔术很方便，但目前它会缓冲输出，因此在进程完成之前您不会看到日志。

`stage3_max_live_parameters`是在任何给定时间内保留在GPU上的完整参数的上限。"重复使用距离"是一个衡量参数将来何时再次使用的度量标准，我们使用`stage3_max_reuse_distance`来决定是丢弃参数还是保留参数。如果一个参数将在不久的将来再次使用（小于`stage3_max_reuse_distance`），那么我们将保留它以减少通信开销。当您启用激活检查点时，这非常有用，其中我们对单层进行前向重计算和反向传递，并希望在前向重计算中保留参数直到反向传递。

以下配置值取决于模型的隐藏大小：

- `reduce_bucket_size`：`hidden_size*hidden_size`
- `stage3_prefetch_bucket_size`：`0.9 * hidden_size * hidden_size`
- `stage3_param_persistence_threshold`：`10 * hidden_size`

因此，将这些值设置为`auto`，[`Trainer`]将自动分配推荐的值。当然，您也可以显式设置这些值。

`stage3_gather_16bit_weights_on_model_save`在保存模型时启用模型fp16权重合并。对于大型模型和多个GPU，这是一项昂贵的操作，无论是在内存还是速度方面。如果您计划恢复训练，目前需要它。请注意，将来会有更新，消除此限制并提供更灵活的功能。

如果您从ZeRO-2配置迁移，请注意，ZeRO-3不使用`allgather_partitions`，`allgather_bucket_size`和`reduce_scatter`配置参数。如果您在配置文件中保留这些参数，它们将被忽略。

- `sub_group_size`：`1e9`

`sub_group_size`控制在优化器步骤期间参数更新的粒度。参数分组到`sub_group_size`大小的存储桶中，并且每个存储桶都逐个更新。当与ZeRO-Infinity中的NVMe卸载一起使用时，`sub_group_size`控制模型状态在优化器步骤期间从NVMe移入和移出CPU内存的粒度。这可防止在极大型模型的情况下CPU内存不足。

当不使用NVMe卸载时，您可以将`sub_group_size`保留为默认值*1e9*。在以下情况下，您可能希望更改其默认值：

1. 优化器步骤时遇到OOM：减小`sub_group_size`以减少临时缓冲区的内存使用
2. 优化器步骤花费很长时间：增加`sub_group_size`以提高由于增加数据缓冲区而导致的带宽利用率。

#### ZeRO-0配置

请注意，我们将第0和第1阶段放在最后，因为它们很少使用。

阶段0是禁用所有分片类型，仅使用DeepSpeed作为DDP。您可以使用以下方法启用它：

```json
{
    "zero_optimization": {
        "stage": 0
    }
}
```

这将完全禁用ZeRO，而您无需更改其他任何内容。

#### ZeRO-1配置

第1阶段是第2阶段减去梯度分片。您可以尝试使用以下方法来稍微加快速度，只在优化器状态中进行分片：

```json
{
    "zero_optimization": {
        "stage": 1
    }
}
```

### NVMe支持

通过使用NVMe内存可以扩展GPU和CPU内存，ZeRO-Infinity允许训练规模非常大的模型。由于智能划分和平铺算法，每个GPU在卸载过程中需要发送和接收非常少量的数据，因此现代NVMe被证明适合为训练过程提供总共更大的内存池。ZeRO-Infinity需要启用ZeRO-3。

以下配置示例启用了将优化器状态和参数同时卸载到NVMe：

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
}
```

您可以选择同时卸载优化器状态和参数到NVMe，或者只卸载它们中的一个，或者都不卸载。例如，如果您有大量的CPU内存可用，可以只卸载到CPU内存，因为它的速度更快（提示："device": "cpu"）。

这是卸载[优化器状态](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading)和[参数](https://www.deepspeed.ai/docs/config-json/#parameter-offloading)的完整文档。

确保`nvme_path`实际上是一个NVMe，因为它可以与常规硬盘或固态硬盘一起使用，但速度要慢得多。快速可扩展的训练是针对现代NVMe传输速度设计的（按照当前编写时，最大读取速度约为3.5GB / s，写入速度约为3GB / s）。

要找出最佳的`aio`配置块，您必须在目标设置上运行基准测试，详细信息请参见[此处](https://github.com/microsoft/DeepSpeed/issues/998)。

#### ZeRO-2与ZeRO-3性能进行比较

如果在其他所有配置保持不变的情况下，ZeRO-3可能比ZeRO-2慢，因为前者需要收集模型权重，并且比ZeRO-2执行的操作更多。如果ZeRO-2满足您的需求，并且您不需要在几个GPU之间扩展，那么可以选择使用ZeRO-2。重要的是要了解，ZeRO-3可以以更高的可扩展性为代价提供更高的性能。

可以调整ZeRO-3配置，使其性能更接近于ZeRO-2：

- 将`stage3_param_persistence_threshold`设置为一个非常大的值-大于最大的参数值，例如`6 * hidden_size * hidden_size`。这将使参数保留在GPU上。
- 关闭`offload_params`，因为ZeRO-2没有该选项。

即使您不更改`stage3_param_persistence_threshold`，只要将`offload_params`关闭，性能可能会显着提高。当然，这些更改将影响您可以训练的模型的大小。因此，这些更改可让您在可扩展性和速度之间进行权衡，具体取决于您的需求。

#### ZeRO-2示例

这是一个完整的ZeRO-2自动配置文件`ds_config_zero2.json`：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

这是一个完整的手动设置的ZeRO-2配置文件，主要是为了让您看到典型值的外观，但我们强烈建议使用其中具有多个`auto`设置的值。

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
```

#### ZeRO-3示例

这是一个完整的ZeRO-3自动配置文件`ds_config_zero3.json`：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

这是一个完整的手动设置的ZeRO-3配置文件，主要是为了让您看到典型值的外观，但我们强烈建议使用其中具有多个`auto`设置的值。

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 0.94e6,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
```

#### 如何选择最佳性能的ZeRO阶段和卸载方式

现在您知道有所有这些不同的阶段了。如何决定要使用其中哪个阶段呢？本部分将尝试回答这个问题。

通常，以下情况适用：

- 从速度角度来看（左边比右边快）

阶段0（DDP）> 阶段1 > 阶段2 > 阶段2 + 卸载 > 阶段3 > 阶段3 + 卸载

- 从GPU内存使用率来看（右边比左边更高效）

阶段0（DDP）< 阶段1 < 阶段2 < 阶段2 + 卸载 < 阶段3 < 阶段3 + 卸载

因此，当您希望获得最快的执行速度，同时适应最小数量的GPU时，可以按照以下流程进行操作。我们从最快的方法开始，如果发生GPU OOM，然后转到更低速的方法，但使用更少的GPU内存。依此类推。

首先将批次大小设置为1（您始终可以使用渐进累积进行任何所需的有效批次大小）。

1. 启用`--gradient_checkpointing 1`（HF Trainer）或直接`model.gradient_checkpointing_enable()`- 如果发生OOM，则
2. 尝试首先使用ZeRO阶段2。如果发生OOM，则
3. 尝试使用ZeRO阶段2 + `offload_optimizer`。如果发生OOM，则
4. 切换到ZeRO阶段3。如果发生OOM，则
5. 将`offload_param`设置为`cpu`。如果发生OOM，则
6. 将`offload_optimizer`设置为`cpu`。如果发生OOM，则

7. 如果仍然无法适应批次大小为1，请检查各种默认值，并在可能的情况下将其降低。例如，如果使用`generate`并且不使用宽的搜索束，将其变为更窄，因为它会消耗大量内存。

8. 绝对使用半精度而不是fp32 - 因此在Ampere及更高的GPU上使用bf16，在较旧的GPU架构上使用fp16。

9. 如果仍然发生OOM，可以添加更多硬件或启用ZeRO-Infinity-将`offload_param`和`offload_optimizer`切换到`nvme`。您需要确保它是一个非常快速的nvme。作为一个轶事，我曾经能够在一个小型GPU上推断BLOOM-176B，但速度非常慢。但它确实可以工作！

当您的批次大小为1时，没有发生OOM，请测量有效吞吐量。

接下来，尝试增加批次大小，尽可能大，因为批次大小越大，GPU执行的效率越高，因为它们在乘以矩阵时表现最佳，而这些矩阵都非常大。

现在性能优化游戏开始。您可以关闭一些卸载功能或者降低 ZeRO 阶段，并增加/减少批大小，然后再测量有效吞吐量。反复洗涤直到满意。

不要花太多时间在上面，但是如果您要开始一个为期 3 个月的训练，确保在此过程中花几天时间找到最有效的吞吐量设置。这样你的训练成本将是最低的，训练速度也会更快。在当前快节奏的机器学习世界中，如果你多花一个月的时间来训练某个东西，很可能就会错过一个绝佳的机会。当然，这只是我的观察分享，无论如何，我都不想催促你。在开始训练 BLOOM-176B 之前，我花了 2 天时间进行了这个过程，并且能够将吞吐量从 90 提高到 150 TFLOPs！这个努力帮助我们节约了一个多月的训练时间。

这些注意事项主要是为训练模式编写的，但大部分适用于推理模式。例如，在推理期间，渐变检查点是无效操作，因为它只在训练期间有用。此外，我们发现，如果您正在进行多 GPU 推理，并且未使用 [DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/)，[Accelerate](https://huggingface.co/blog/bloom-inference-pytorch-scripts)应该提供更优秀的性能。

其他与性能相关的快速注释:
- 如果您从头开始训练某个东西，请尝试使张量的形状可被 16 整除（例如隐藏大小）。对于批大小，请至少尝试使其可被 2 整除。如果要从 GPU 中获得更高的性能，可以尝试硬件特定的[波和瓷砖量化](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/)整除方式。

### 激活检查点或渐变检查点

激活检查点和渐变检查点是两个相互独立的术语，指的是同一方法。这非常令人困惑，但情况就是这样。

渐变检查点允许您在 GPU 内存和速度之间进行权衡，它可以克服 GPU OOM 或增加批大小，从而通常可以获得更好的性能。

HF 变换器模型不知道 DeepSpeed 的激活检查点，因此，如果您尝试在 DeepSpeed 配置文件中启用该功能，将不会发生任何事情。

因此，您有两种方法可以利用此非常有益的功能：

1. 如果要使用 HF 变换器模型，可以使用 `model.gradient_checkpointing_enable()` 或在 HF Trainer 中使用 `--gradient_checkpointing`，它将自动为您启用此功能。在那里使用了 `torch.utils.checkpoint`。
2. 如果您自己编写了模型，并且想使用 DeepSpeed 的激活检查点，则可以使用[此处](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html)规定的 API。您还可以使用 HF 变换器建模代码并将`torch.utils.checkpoint` 替换为 DeepSpeed 的 API。后者更加灵活，因为它允许您将前向激活卸载到 CPU 内存，而不是重新计算它们。

### 优化器和调度器

只要不启用 `offload_optimizer`，就可以混合使用 DeepSpeed 和 HuggingFace 的调度器和优化器，除了使用 HuggingFace 调度器和 DeepSpeed 优化器的组合之外:

| 组合       | HF 调度器 | DS 调度器 |
| HF 优化器 | 是          | 是          |
| DS 优化器 | 否           | 是          |

可以使用非 DeepSpeed 优化器，只要它具有 CPU 和 GPU 实现（不包括 LAMB）。

优化器必须通过[此处](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters)进行配置。DeepSpeed 的主要优化器是 Adam、AdamW、OneBitAdam 和 Lamb。这些优化器已经经过全面测试，因此建议使用。它还可以从 `torch` 导入其他优化器。如果不在配置文件中配置 `optimizer` 条目，则 [`Trainer`] 将自动将其设置为 `AdamW`，并使用提供的值或默认值设置以下命令行参数: `--learning_rate`、`--adam_beta1`、`--adam_beta2`、`--adam_epsilon` 和 `--weight_decay`。

以下是自动配置的 `AdamW` 的示例:

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}
```

请注意，命令行参数将设置配置文件中的值。这样就有了一个定义值的唯一来源，并且避免了例如在不同位置设置学习率为不同值时难以找到的错误。命令行的规则优先。被覆盖的值有:

- `lr` 使用 `--learning_rate` 的值
- `betas` 使用 `--adam_beta1` 和 `--adam_beta2` 的值
- `eps` 使用 `--adam_epsilon` 的值
- `weight_decay` 使用 `--weight_decay` 的值

因此，请记住在命令行上调整共享超参数。

您还可以显式地设置值:

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": 0.001,
         "betas": [0.8, 0.999],
         "eps": 1e-8,
         "weight_decay": 3e-7
       }
   }
}
```

但是，您需要自己同步 [`Trainer`] 命令行参数和 DeepSpeed 配置文件。

如果要使用其他未列出的优化器，必须将其添加到顶级配置中。

```json
{
   "zero_allow_untested_optimizer": true
}
```

与 `AdamW` 类似，您可以配置其他官方支持的优化器。只需记住这些优化器可能具有不同的配置值。例如，对于 Adam，您将希望 `weight_decay` 在`0.01` 左右。

此外，当与卸载一起使用时，使用 Deepspeed 的 CPU Adam 优化器时效果最好。如果要在卸载时使用其他优化器，自 `deepspeed==0.8.3` 以来，您还需要添加:

```json
{
   "zero_force_ds_cpu_optimizer": false
}
```
到顶级配置。

#### 优化器

DeepSpeed 的主要优化器是 Adam、AdamW、OneBitAdam 和 Lamb。这些已经与 ZeRO 进行了彻底测试，因此建议使用。它也可以从 `torch` 导入其他优化器。完整的文档在[这里](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters)。

如果未在配置文件中配置 `optimizer` 条目，则 [`Trainer`] 将自动将其设置为 `AdamW`，并使用提供的值或默认值设置以下命令行参数: `--learning_rate`、`--adam_beta1`、`--adam_beta2`、`--adam_epsilon` 和 `--weight_decay`。

以下是自动配置的 `AdamW` 的示例:

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}
```

请注意，[`Trainer`] 参数将设置配置文件中的值。这样就有了一个定义值的唯一来源，并且避免了例如在不同位置设置学习率为不同值时难以找到的错误。命令行优先。被覆盖的值为：

- `lr` 的值为 `--learning_rate`
- `betas` 的值为 `--adam_beta1 --adam_beta2`
- `eps` 的值为 `--adam_epsilon`
- `weight_decay` 的值为 `--weight_decay`

因此，请记住在命令行上调整共享超参数。

您还可以显式设置这些值：

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": 0.001,
         "betas": [0.8, 0.999],
         "eps": 1e-8,
         "weight_decay": 3e-7
       }
   }
}
```

但是，您需要自己同步 [`Trainer`] 命令行参数和 DeepSpeed 配置文件。

如果要使用其他未列出的优化器，则必须将它们添加到顶级配置中：

```json
{
   "zero_allow_untested_optimizer": true
}
```

类似于 `AdamW`，您可以配置其他官方支持的优化器。只需记住这些优化器可能具有不同的配置值。例如，对于 Adam，您可能希望 `weight_decay` 在 `0.01` 附近。

另外，当与卸载一起使用时，使用 Deepspeed 的 CPU Adam 优化器效果最好。如果想要使用其他的卸载器，例如 `deepspeed==0.8.3` 时，还需要添加以下配置：

```json
{
   "zero_force_ds_cpu_optimizer": false
}
```

#### 调度器

DeepSpeed 支持 `LRRangeTest`、`OneCycle`、`WarmupLR` 和 `WarmupDecayLR` 学习率调度器。完整文档在[这里](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters)。

以下是 DeepSpeed 和 🤗 Transformers 之间调度器的重叠部分：

- `WarmupLR` 通过 `--lr_scheduler_type constant_with_warmup`。
- `WarmupDecayLR` 通过 `--lr_scheduler_type linear`。这也是 `--lr_scheduler_type` 的默认值，因此，如果不配置调度器，这是默认的配置。

如果不在配置文件中配置 `scheduler` 条目，则 [`Trainer`] 将使用 `--lr_scheduler_type`、`--learning_rate` 和 `--warmup_steps` 或 `--warmup_ratio` 的值配置 🤗 Transformers 版本。

以下是自动配置的 `WarmupLR` 的示例:

```json
{
   "scheduler": {
         "type": "WarmupLR",
         "params": {
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

由于使用了 "auto"，[`Trainer`] 参数将在配置文件中设置正确的值。这样就有了一个定义值的唯一来源，并且避免了例如在不同位置设置学习率为不同值时难以找到的错误。命令行优先。设置的值为：

- `warmup_min_lr` 的值为 `0`。
- `warmup_max_lr` 的值为 `--learning_rate`。
- `warmup_num_steps` 的值为如果提供了 `--warmup_steps`，则使用该值。否则，将使用 `--warmup_ratio` 乘以训练步骤的数量，并向上取整。
- `total_num_steps` 的值为 `--max_steps` 的值，否则在运行时根据环境、数据集的大小和其他命令行参数自动推导出来（`WarmupDecayLR` 需要）。

当然，您可以接管配置值中的任何一个或多个，并自行设置：

```json
{
   "scheduler": {
         "type": "WarmupLR",
         "params": {
             "warmup_min_lr": 0,
             "warmup_max_lr": 0.001,
             "warmup_num_steps": 1000
         }
     }
}
```

但是，您需要自己同步 [`Trainer`] 命令行参数和 DeepSpeed 配置。

例如，对于 `WarmupDecayLR`，可以使用以下条目:

```json
{
   "scheduler": {
         "type": "WarmupDecayLR",
         "params": {
             "last_batch_iteration": -1,
             "total_num_steps": "auto",
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

它将在加载时设置 `total_num_steps`、`warmup_max_lr`、`warmup_num_steps` 和 `total_num_steps`。

### fp32 精度

Deepspeed 支持完全的 fp32 和 fp16 混合精度。

由于 fp16 混合精度需要的内存更少，速度更快，所以您唯一不希望使用的情况是当您使用的模型在此训练模式下表现不佳时。通常，当模型没有在 fp16 混合精度下进行预训练时，就会发生这种情况（例如，bf16 预训练模型通常会发生这种情况）。这样的模型可能会溢出或下溢，导致损失为 `NaN`。如果是这种情况，您将希望使用完全的 fp32 模式，并通过显式禁用默认的 fp16 混合精度模式来禁用它:

```json
{
    "fp16": {
        "enabled": false,
    }
}
```

如果使用 Ampere 架构的 GPU，从 pytorch 1.7 版本开始，默认情况下会自动切换为使用更高效的 tf32 格式进行某些操作，但结果仍然是 fp32。有关详细信息和基准测试，请参见[TensorFloat-32(TF32) on Ampere devices](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)。文档中包含有关如何禁用此自动转换的说明，如果出于某种原因你不想使用它。

使用 🤗 Trainer，您可以使用 `--tf32` 启用它，或使用 `--tf32 0` 或 `--no_tf32` 禁用它。默认情况下，PyTorch 使用默认值。

```json
{
    "bf16": {
        "enabled": "auto"
    }
}
```

bf16 的动态范围与 fp32 相同，因此不需要有损失区。

当使用 `--bf16` 或 `--bf16_full_eval` 命令行参数时，启用此模式。

您还可以显式启用/禁用此模式：

```json
{
    "bf16": {
        "enabled": true
    }
}
```

提示:

截至 `deepspeed==0.6.0`，bf16 支持是新的和实验性的。

如果您在训练时使用[梯度累积](#gradient-accumulation)，并启用了 bf16，您需要注意，它将以 bf16 累积梯度，这可能不是您想要的，因为此格式的精度较低，可能会导致有损累积。

正在努力解决此问题，并提供一个选项来使用更高精度的 `dtype`（fp16 或 fp32）。

### NCCL 集合

有一个 `dtype` 是训练制度，还有一个单独的 `dtype` 用于通信集合，如各种缩减和收集/分散操作。

所有收集/分散操作都使用与数据相同的 `dtype`，因此，如果您正在使用 bf16 训练制度，则以 bf16 进行收集。收集是一个非损失操作。

各种缩减操作可能会非常有损，例如当梯度在多个 GPU 上进行平均时，如果通信是在 fp16 或 bf16 上执行的，则结果很可能会有损-因为在低精度下添加多个数字时，结果不是精确的。特别是在使用 bf16 时更加如此，因为它的精度低于 fp16。通常情况下，fp16 已经足够好，因为平均梯度通常非常小。因此，默认情况下，在半精度训练中使用 fp16 作为缩减操作的默认值。但是，您对此功能有完全的控制，并且如果选择，可以添加一些额外的开销，并确保在累计完成后将其累积到半精度 `dtype` 中，直到结果准备好后才降级到您正在训练的半精度“dtype”。

为了覆盖默认值，您只需添加一个新的配置条目：

```json
{
    "communication_data_type": "fp32"
}
```

截至撰写本文时，有效值是 "fp16"、"bfp16" 和 "fp32"。

注意：stage zero 3 中有一个与 bf16 comm dtype 相关的错误，在 `deepspeed==0.8.1` 中已经修复。

### 自动混合精度

您可以使用 pytorch-like AMP 方法或 apex-like 方法来使用自动混合精度：

### fp16

要配置带有 fp16（float16）的 pytorch-like AMP 模式，请设置：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

[`Trainer`] 将根据 `args.fp16_backend` 的值和 `args.fp16_opt_level` 的值自动启用或禁用此模式。

当传递 `--fp16 --fp16_backend amp --fp16_opt_level 01` 命令行参数时，将启用此模式。

您还可以显式配置此模式：

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

但是，您需要自己同步 [`Trainer`] 的命令行参数和 DeepSpeed 的配置文件。

在[这里](https://www.deepspeed.ai/docs/config-json/#fp16-training-options)进行了更详细的说明。

### bf16

如果希望使用 bf16（bfloat16）而不是 fp16，则可以使用以下配置部分：

```json
{
    "bf16": {
        "enabled": "auto"
    }
}
```

bf16 与 fp32 具有相同的动态范围，因此不需要有损补。

当传递 `--bf16` 或 `--bf16_full_eval` 命令行参数时，启用此模式。

您还可以显式启用/禁用此模式：

```json
{
    "bf16": {
        "enabled": true
    }
}
```

提示：

截至 `deepspeed==0.6.0`，bf16 支持是新的和实验性的。

如果您在训练时使用[梯度累积](#gradient-accumulation)，并启用了 bf16，您需要注意，它将以 bf16 累积梯度，这可能不是您想要的，因为此格式的精度较低，可能会导致有损累积。

正在努力解决此问题，并提供一个选项来使用更高精度的 `dtype`（fp16 或 fp32）。

### NCCL 集合

有一个 `dtype` 是训练制度，还有一个单独的 `dtype` 用于通信集合，如各种缩减和收集/分散操作。

所有收集/分散操作都使用与数据相同的 `dtype`，因此，如果您正在使用 bf16 训练制度，则以 bf16 进行收集-收集是一个非损失操作。

各种缩减操作可能会非常有损，例如当梯度在多个 GPU 上进行平均时，如果通信是在 fp16 或 bf16 上执行的，则结果很可能会有损。因为当以低精度相加多个数字时，结果不是精确的。更重要的是在使用 bf16 时。因为 bf16 的精度低于 fp16。通常情况下，fp16 已经足够好，因为平局梯度通常非常小。因此，默认情况下，半精度训练使用 fp16 作为缩减操作的默认设置。<button>但是您对此功能有完全的控制，并且如果您选择，您可以添加一些额外的开销并确保在累积完成后将其累积到您正在训练的半精度“dtype”。``

要覆盖默认值，只需添加一个新的配置条目：

```json
{
    "communication_data_type": "fp32"
}
```

在撰写本文时，有效值为 "fp16"、"bfp16" 和 "fp32"。

注意:自`deepspeed==0.8.1` 以来修复了 bf16 comm dtype 的错误。



```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 2 GPUs per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   76.74GB |   2.71GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   76.74GB |   2.71GB | offload_param=none, offload_optimizer=cpu , zero_init=0
   52.29GB |  43.46GB | offload_param=none, offload_optimizer=none, zero_init=1
  133.57GB |  43.46GB | offload_param=none, offload_optimizer=none, zero_init=0
```

You can see with 2 GPUs, you need around 2.7GB for each GPU. That increases to around 43.5GB when offloading to GPUs and no CPU offload is performed.

Again, this is just an estimate and you should experiment with different settings to find the best tradeoff between cost and speed for your specific use case.

```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'
[...]
估计为params、optim状态和gradients需要内存的参数：
HW:设置为1个节点，每个节点2个GPU。
SW:模型总参数数为2783M，最大层参数为65M。
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   2.84GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   2.84GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.74GB |  23.58GB | offload_param=none, offload_optimizer=none, zero_init=1
   31.11GB |  23.58GB | offload_param=none, offload_optimizer=none, zero_init=0
```

因此，您需要使用2个32GB或更高的GPU来运行此模型，且不进行CPU卸载。

有关完整信息，请参阅[memory estimators](https://deepspeed.readthedocs.io/en/latest/memory.html)。

### 提交问题

在报告中，请始终包括以下内容：

1. 完整的Deepspeed配置文件
2. 如果您使用[`Trainer`]，请包括命令行参数；如果您使用[`TrainingArguments`]自己进行Trainer设置，请不要包括[`TrainingArguments`]的dump，因为其中有数十个与问题无关的条目。
3. 运行以下命令后的输出：
```bash
python -c 'import torch; print(f"torch: {torch.__version__}")'
python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'
```
4. 如果可能，提供一个可以在其上重现该问题的Google Colab笔记本的链接。
5. 除非无法，否则请始终使用我们可以使用的标准数据集而不是自定义数据集。

6. 如果可能，请尝试使用现有的[示例](https://github.com/huggingface/transformers/tree/main/examples/pytorch)之一重现该问题。

需要考虑的问题：

- Deepspeed往往不是问题的原因。

  一些已归档的问题证明与Deepspeed无关。即，一旦将Deepspeed从设置中删除，问题仍然存在。

  因此，如果不是绝对明显是Deepspeed相关的问题，即您可以看到存在异常和DeepSpeed模块涉及到的问题，请首先在没有Deepspeed的设置中重新测试您的设置。只有在问题仍然存在的情况下才提到Deepspeed并提供所有所需的详细信息。

- 如果您确定问题是DeepSpeed核心中的问题而不是集成部分，请直接在[Deepspeed](https://github.com/microsoft/DeepSpeed/)上提交Issue。如果您不确定，请不要担心，任何一个Issue跟踪器都可以，一旦您发布问题，我们将找到解决方法并将您重定向到另一个Issue跟踪器（如果需要）。

### 故障排除

#### 在启动时，`deepspeed`进程无回溯地被杀死

如果`deepspeed`进程在启动时被无回溯地杀死，这通常意味着程序尝试分配的CPU内存超过了系统或进程允许分配的CPU内存，因此操作系统内核杀死了该进程。这是因为您的配置文件很可能同时配置了`offload_optimizer`和`offload_param`将其转移到了`cpu`。如果您有NVMe，如果在ZeRO-3下运行，可以尝试将其分流到NVMe。可以使用以下方法来[估计为特定模型需要多少内存](https://deepspeed.readthedocs.io/en/latest/memory.html)。

#### 训练和/或评估/预测损失为NaN

在将以bf16混合精度模式预训练的模型用于不带混合精度的fp16下时，经常会发生损失为NaN的情况。大多数基于TPU并且通常是谷歌发布的模型都属于此类别（例如，几乎所有基于t5的模型）。在这种情况下，解决方案是要么使用fp32，要么使用如果您的硬件支持（TPU、Ampere GPU或更新版本）时使用bf16。

另一个问题可能与使用fp16有关。当配置以下部分时：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

并且您在日志中看到Deepspeed报告如下`OVERFLOW!`的情况：

```
0%|                                                                                                                             | 0/189 [00:00<?, ?it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
  1%|▌                                                                                                                    | 1/189 [00:00<01:26,  2.17it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072.0
  1%|█▏
 [...]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 14%|████████████████▌                                                                                                   | 27/189 [00:14<01:13,  2.21it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▏                                                                                                  | 28/189 [00:14<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▊                                                                                                  | 29/189 [00:15<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
[...]
```

这意味着Deepspeed损失缩放器无法找到一个可以克服损失溢出的缩放系数。

（此日志已进行优化，以便更易读）

在这种情况下，您通常需要提高`initial_scale_power`的值。将其设置为`"initial_scale_power": 32`通常可以解决该问题。

### 注意事项

- Deepspeed可以与PyTorch [`Trainer`]一起工作，但无法与TF [`TFTrainer`]一起工作。
- 虽然DeepSpeed有一个可pip安装的PyPI软件包，但强烈建议从[源代码](https://github.com/microsoft/deepspeed#installation)进行安装，以便最好地匹配您的硬件，并且如果您需要启用某些功能（如1-bit Adam），在pypi分发中无法使用。
- 您不必使用[`Trainer`]来与Deepspeed和🤗 Transformers一起使用-您可以使用任何模型与自己的训练器，并且您将不得不根据[Deepspeed集成说明](https://www.deepspeed.ai/getting-started/#writing-deepspeed-models)来调整后者的设置。

## 使用非Trainer的Deepspeed集成

当不使用[`Trainer`]时，[`~integrations.HfDeepSpeedConfig`]用于将Deepspeed集成到🤗 Transformers核心功能中。唯一的需要是处理Deepspeed ZeRO-3参数聚合并在`from_pretrained`调用期间自动将模型分割到多个GPU上。其他所有操作都需要您自己完成。

当使用[`Trainer`]时，所有操作都会自动处理。

当不使用[`Trainer`]时，为了有效地部署DeepSpeed ZeRO-3，您必须在实例化模型之前实例化[`~integrations.HfDeepSpeedConfig`]对象，并将该对象保持活动状态。

如果您使用Deepspeed ZeRO-1或ZeRO-2，则根本不需要使用`HfDeepSpeedConfig`。

例如，对于预训练模型：

```python
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import deepspeed

ds_config = {...}  # deepspeed配置对象或文件的路径
# 必须在实例化模型之前运行以检测zero 3
dschf = HfDeepSpeedConfig(ds_config)  # 保持此对象的活动状态
model = AutoModel.from_pretrained("gpt2")
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

或者对于非预训练模型：

```python
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel, AutoConfig
import deepspeed

ds_config = {...}  # deepspeed配置对象或文件的路径
# 必须在实例化模型之前运行以检测zero 3
dschf = HfDeepSpeedConfig(ds_config)  # 保持此对象的活动状态
config = AutoConfig.from_pretrained("gpt2")
model = AutoModel.from_config(config)
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

请注意，如果您不使用[`Trainer`]集成，则完全由您自己负责。基本上按照[Deepspeed](https://www.deepspeed.ai/)网站上的文档操作。此外，必须显式配置配置文件-无法使用`"auto"`值，必须使用实际值。

## HfDeepSpeedConfig

[[autodoc]] integrations.HfDeepSpeedConfig
    - all

### 自定义Deepspeed ZeRO推理

以下示例演示了如何在不使用[`Trainer`]时进行Deepspeed ZeRO推理，当无法将模型装入单个GPU中时。该解决方案包括使用额外的GPU和/或将GPU内存卸载到CPU内存中。

需要了解的重要细微之处是，ZeRO的设计方式允许在每个GPU上并行处理不同的输入。

示例具有大量注释，并以自我记录方式进行了说明。

确保：

1. 如果您有足够的GPU内存，请禁用CPU卸载（因为会减慢处理速度）
2. 如果您拥有Ampere或更高版本的GPU，请启用bf16以加快速度。如果您没有这样的硬件，只要不使用以bf16混合精度预训练的任何模型（例如大多数t5模型），您可以启用fp16。这些模型通常在fp16中溢出，并显示垃圾输出。

```python
#!/usr/bin/env python

# 此脚本演示了在无法将模型装入单个GPU中时如何在推理模式下使用Deepspeed ZeRO。
#
# 1. 使用1个带CPU卸载的GPU
# 2. 或者使用多个GPU
#
# 首先您需要安装deepspeed：pip install deepspeed
#
# 这里我们使用3B "bigscience/T0_3B"模型，它需要大约15GB的GPU RAM-因此可以使用1个较大的或2个较小的GPU来处理它。或者，一个小型的GPU和大量的CPU内存。
#
# 要使用更大的模型，比如需要大约50GB的"bigscience/T0"，除非您拥有一个80GB的GPU，否则需要使用2-4个GPU。然后您可以根据需要调整该脚本以处理更多的GPU。
#
# 提供的deepspeed配置还激活了CPU内存卸载，因此，如果您有大量可用的CPU内存，并且不介意减慢速度，应该可以加载通常不适应单个GPU的模型。如果您有足够的GPU内存，如果您不想进行CPU卸载，那么程序将运行得更快-因此禁用该部分。
#
# 要在1个gpu上部署：
#
# deepspeed --num_gpus 1 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# 要在2个gpu上部署：
#
# deepspeed --num_gpus 2 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=2 t0.py


from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免关于tokenizers并行性的警告

# 分布式设置
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = "bigscience/T0_3B"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# 批处理大小必须可被world_size整除，但可以大于world_size
train_batch_size = 1 * world_size

# ds_config 注释：
#
# - 如果您使用的是Ampere或更高版本的GPU，请启用bf16-这将以混合精度运行并且速度更快。
#
# - 对于旧一些的GPU，您可以启用fp16，但仅使用未经bf16预训练的模型-例如，所有官方的t5模型都是经过bf16预训练的。
#
# - 将offload_param.device设置为"none"或完全删除`offload_param`部分，如果您不- 想进行CPU卸载
#
# - 如果使用`offload_param`，您可以手动微调stage3_param_persistence_threshold以控制应保留在GPU上的参数数量- 值越大，卸载的尺寸越小
#
# 有关Deepspeed配置的详细信息，请参见
# https://huggingface.co/docs/transformers/main/main_classes/deepspeed

# 为了保持与.json的一致性使用相同的格式，只是它在true/false上使用小写
# fmt: off
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
# fmt: on

# 下一行指示transformers在调用模型的`from_pretrained`方法时，使用deepspeed.zero.Init直接在多个gpu上对模型进行分区。
#
# **必须在加载模型AutoModelForSeq2SeqLM.from_pretrained(model_name)之前运行此行**
#
# 否则，模型将首先以常规方式加载，仅在前向时分区，这样会更低效，并且在CPU内存很少的情况下可能会失败
dschf = HfDeepSpeedConfig(ds_config)  # 保持此对象的活动状态

# 现在可以加载模型。
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 初始化Deepspeed ZeRO并仅存储引擎对象
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # 推理模式

# Deepspeed ZeRO可以在每个GPU上处理不相关的输入。因此，对于2个gpu，您可以同时处理2个输入。
# 如果只有一个要处理的输入，则需要同时将相同的字符串传递给两个gpu
# 如果只有一个GPU，那么您只有rank 0。
rank = torch.distributed.get_rank()
if rank == 0:
    text_in = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
elif rank == 1:
    text_in = "Is this review positive or negative? Review: this is the worst restaurant ever"

将下面这句话翻译成中文，格式是markdown，<>里面的保留原文，也不要添加额外的内容：

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, synced_gpus=True)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"rank{rank}:\n   in={text_in}\n  out={text_out}")
```

将其保存为`t0.py`并运行：

```
$ deepspeed --num_gpus 2 t0.py
rank0:
   in=Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy
  out=Positive
rank1:
   in=Is this review positive or negative? Review: this is the worst restaurant ever
  out=negative
```

这是一个非常基本的示例，你需要根据自己的需求进行调整。

### `generate`细微差别

使用ZeRO Stage-3和多个GPU时，必须通过调用`generate(..., synced_gpus=True)`来同步GPU。如果不这样做，如果某个GPU在其他GPU之前完成生成，则整个系统将发生挂起，因为其他GPU将无法从停止生成的GPU接收权重分片。

从`transformers>=4.28`开始，如果未显式指定`synced_gpus`，则如果检测到以下条件，它将自动设置为`True`。但是，如果需要，仍然可以覆盖`synced_gpus`的值。

## 测试Deepspeed集成

如果您提交的PR涉及DeepSpeed集成，请注意我们的CircleCI PR CI设置没有GPU，因此我们只在另一个CI夜间运行需要GPU的测试。因此，如果您在PR中得到一个绿色的CI报告，并不意味着DeepSpeed测试通过。

要运行DeepSpeed测试，请至少运行以下命令：

```
RUN_SLOW=1 pytest tests/deepspeed/test_deepspeed.py
```

如果更改了任何建模或pytorch示例代码，请运行模型库测试。以下命令将运行所有DeepSpeed测试：

```
RUN_SLOW=1 pytest tests/deepspeed
```

## 主要DeepSpeed资源

- [项目的GitHub](https://github.com/microsoft/deepspeed)
- [使用文档](https://www.deepspeed.ai/getting-started/)
- [API文档](https://deepspeed.readthedocs.io/en/latest/index.html)
- [博客文章](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

论文：

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)

最后，请记住，HuggingFace的[`Trainer`]只集成了DeepSpeed，因此如果您在使用DeepSpeed时遇到任何问题或疑问，请在[DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/issues)上提交问题。