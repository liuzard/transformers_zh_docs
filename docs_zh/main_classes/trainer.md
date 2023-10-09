版权所有2020年The HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”），除非符合许可以外，否则你不得使用此文件。
你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或经书面同意，否则根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证的特定语言，以获得许可证下的特定语言的权限和限制的说明。

⚠️请注意，此文件在Markdown中，但包含我们的文档生成器特定语法（类似于MDX），在你的Markdown查看器中可能无法正确呈现。

# Trainer

[`Trainer`]类为PyTorch的大多数标准用例提供了完整的训练API。它在大多数[示例脚本](https://github.com/huggingface/transformers/tree/main/examples)中使用。

在实例化你的[`Trainer`]之前，请创建一个[`TrainingArguments`]来在训练期间访问所有自定义点。

该API支持在多个GPU/TPU上进行分布式训练，通过[NVIDIA Apex](https://github.com/NVIDIA/apex)和PyTorch的Native AMP进行混合精度训练。

[`Trainer`]包含支持上述功能的基本训练循环。要注入自定义行为，可以对其进行子类化并覆盖以下方法：

- **get_train_dataloader** —— 创建训练DataLoader。
- **get_eval_dataloader** —— 创建评估DataLoader。
- **get_test_dataloader** —— 创建测试DataLoader。
- **log** —— 记录监视训练的各种对象的信息。
- **create_optimizer_and_scheduler** —— 设置优化器和学习率调度程序（如果在init时未传递）。请注意，你也可以分别子类化或覆盖`create_optimizer`和`create_scheduler`方法。
- **create_optimizer** —— 设置优化器（如果在init时未传递）。
- **create_scheduler** —— 设置学习率调度程序（如果在init时未传递）。
- **compute_loss** - 计算批处理训练输入的损失。
- **training_step** —— 执行训练步骤。
- **prediction_step** —— 执行评估/测试步骤。
- **evaluate** —— 运行评估循环并返回指标。
- **predict** —— 在测试集上返回预测结果（如果可用，则包括指标）。

<Tip warning={true}>

[`Trainer`]类针对🤗Transformers模型进行了优化，并在使用其他模型时可能具有令人惊讶的行为。当在自己的模型上使用它时，请确保：

- 你的模型始终返回元组或[`~utils.ModelOutput`]的子类。
- 如果提供了`labels`参数并且将损失返回为元组的第一个元素（如果你的模型返回元组），则你的模型可以计算损失。
- 你的模型可以接受多个标签参数（在[`TrainingArguments`]中使用`label_names`来指示它们的名称给[`Trainer`]），但这些参数中没有一个应命名为`"label"`。

</Tip>

以下是如何自定义[`Trainer`]以使用加权损失的示例（在训练集不平衡时非常有用）：

```python
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # 正向传递
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # 计算自定义损失（假设有3个具有不同权重的标签）
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

另一种自定义PyTorch [`Trainer`]的训练循环行为的方法是使用[callbacks](callback)，它们可以检查训练循环状态（用于进度报告、在TensorBoard或其他ML平台上记录日志等）并做出决策（例如提前停止）。


## Trainer

[[autodoc]] Trainer
    - all

## Seq2SeqTrainer

[[autodoc]] Seq2SeqTrainer
    - evaluate
    - predict

## TrainingArguments

[[autodoc]] TrainingArguments
    - all

## Seq2SeqTrainingArguments

[[autodoc]] Seq2SeqTrainingArguments
    - all

## Checkpoints

默认情况下，[`Trainer`]会将所有检查点保存在你设置的`output_dir`中的子文件夹`checkpoint-xxx`中，其中xxx是训练的步骤。

可以通过使用以下调用[`Trainer.train`]来从检查点恢复训练：

- `resume_from_checkpoint=True` ——将从最新的检查点恢复训练
- `resume_from_checkpoint=checkpoint_dir` ——将从指定目录中的特定检查点恢复训练

此外，当使用`push_to_hub=True`时，你还可以将检查点轻松保存在模型中心（Model Hub）中。默认情况下，保存在中间检查点中的所有模型都保存在不同的提交中，但不保存优化器状态。你可以根据你的[`TrainingArguments`]适应[`hub-strategy`]值来执行以下操作之一：

- `"checkpoint"`：最新的检查点也会被推送到一个名为last-checkpoint的子文件夹中，方便你使用`trainer.train(resume_from_checkpoint="output_dir/last-checkpoint")`继续训练。
- `"all_checkpoints"`：所有检查点都按照它们在输出文件夹中出现的方式进行推送（因此你将获得一个检查点文件夹，每个文件夹在最终存储库中对应一个文件夹）


## 日志记录

默认情况下，[`Trainer`]的主进程使用`logging.INFO`，并且如果有的话，副本使用`logging.WARNING`。

可以通过[`TrainingArguments`]的参数来覆盖这些默认设置，以使用5个`logging`级别中的任何一个：

- `log_level` —— 用于主进程
- `log_level_replica` —— 用于副本

此外，如果[`TrainingArguments`]的`log_on_each_node`设置为`False`，则仅主节点将使用其主进程的日志级别设置，所有其他节点将使用副本的日志级别设置。

请注意，[`Trainer`]将在其[`Trainer.__init__`]中为每个节点单独设置`transformers`的日志级别。因此，如果需要在创建[`Trainer`]对象之前使用其他`transformers`功能，请尽早设置此项（请参阅下一个示例）。

以下是如何在应用程序中使用此功能的示例：

```python
[...]
logger = logging.getLogger(__name__)

# 设置日志记录
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# 根据节点设置主代码和其使用的模块的相同日志级别
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

然后，如果你只想在主节点上看到警告，并且所有其他节点不打印任何可能重复的警告，你可以按以下方式运行：

```bash
my_app.py ... --log_level warning --log_level_replica error
```

在多节点环境中，如果还不希望日志在每个节点的主进程中重复，请将上述更改为：

```bash
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0
```

然后，只有第一个节点的主进程将以“warning”级别记录日志，其他节点上的所有进程以及其他节点上的所有进程将以“error”级别记录日志。

如果你需要应用程序尽可能尽静，请执行以下操作：

```bash
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

（如果在多节点环境下，添加`--log_on_each_node 0`）


## 随机性

从[`Trainer`]生成的检查点中恢复训练时，会尽一切努力将_python_、_numpy_和_pytorch_的随机数生成器状态恢复到保存检查点时的状态，以使“停止和恢复”类型的训练尽量接近非停止训练。

然而，由于各种默认的非确定性pytorch设置，这可能无法完全实现。如果需要完全确定性，请参阅[控制随机源](https://pytorch.org/docs/stable/notes/randomness)。正如文档中所解释的，有一些使事物确定的设置（例如`torch.backends.cudnn.deterministic`）可能会减慢速度，因此默认情况下无法执行此操作，但是你可以在需要时启用它们。

## 特定GPU的选择

让我们讨论如何告诉程序使用哪个GPU以及以什么顺序使用它们。

当使用[`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)仅使用GPU子集时，只需指定要使用的GPU数量即可。例如，如果你有4个GPU，但希望使用前2个GPU，可以执行以下操作：

```bash
python -m torch.distributed.launch --nproc_per_node=2  trainer-program.py ...
```

如果你安装了[`accelerate`](https://github.com/huggingface/accelerate)或[`deepspeed`](https://github.com/microsoft/DeepSpeed)，你也可以使用以下命令执行相同操作：

```bash
accelerate launch --num_processes 2 trainer-program.py ...
```

```bash
deepspeed --num_gpus 2 trainer-program.py ...
```

你不需要使用Accelerate或Deepspeed整合功能来使用这些启动器。


到目前为止，你可以告诉程序要使用多少个GPU。现在让我们讨论如何选择特定的GPU并控制其顺序。

以下环境变量可帮助你控制要使用的GPU及其顺序。

**`CUDA_VISIBLE_DEVICES`**

如果你有多个GPU，并且只想使用其中1个或几个GPU，请将环境变量`CUDA_VISIBLE_DEVICES`设置为要使用的GPU列表。

例如，假设你有4个GPU：0、1、2和3。要仅在物理GPU 0和2上运行，可以执行以下操作：

```bash
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch trainer-program.py ...
```

因此，现在pytorch将只看到2个GPU，其中物理GPU 0和2分别映射到cuda:0和cuda:1。

你甚至可以更改它们的顺序：

```bash
CUDA_VISIBLE_DEVICES=2,0 python -m torch.distributed.launch trainer-program.py ...
```

这样，物理GPU 0和2将分别映射到cuda:1和cuda:0。

上述示例都是针对`DistributedDataParallel`使用模式的，但同样的方法也适用于[`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)：
```bash
CUDA_VISIBLE_DEVICES=2,0 python trainer-program.py ...
```

要模拟没有GPU的环境，只需将此环境变量设置为空值，例如：

```bash
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

与任何其他环境变量一样，你当然可以导出它们，而不是将它们添加到命令行，例如：

```bash
export CUDA_VISIBLE_DEVICES=0,2
python -m torch.distributed.launch trainer-program.py ...
```

但是，由于设置环境变量后可能会忘记并不理解为何使用了错误的GPU，因此这种方法可能会令人困惑。因此，按照此部分的大多数示例中显示的方式，最好仅针对特定的运行在同一命令行中设置环境变量。

**`CUDA_DEVICE_ORDER`**

还有一个额外的环境变量`CUDA_DEVICE_ORDER`，用于控制物理设备的排序方式。有两个选择：

1. 按PCIe总线ID排序（与`nvidia-smi`的顺序匹配）- 这是默认设置。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

2. 按GPU计算能力排序

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
```

大多数情况下，你不需要关心这个环境变量，但如果你的设置不平衡，既有较新也有较旧的GPU插入在这样的方式下，以便较慢的旧卡首先出现，这将非常有帮助。解决方法之一是交换卡。但是，如果你无法交换卡（例如，如果设备的冷却受到影响），则将`CUDA_DEVICE_ORDER=FASTEST_FIRST`设置为始终首先放置较新和较快的卡。不过这可能会有点令人困惑，因为`nvidia-smi`仍然会按照PCIe顺序报告它们。

交换顺序的另一种解决方案是使用：

```bash
export CUDA_VISIBLE_DEVICES=1,0
```
在此示例中，我们使用了2个GPU，但是当然，对于你的计算机上的所有GPU都适用相同的情况。

此外，如果你设置了此环境变量，则最好将其设置在`〜/ .bashrc`文件或其他启动配置文件中，并忘记它。


## Trainer集成

[`Trainer`]已扩展以支持可以大大提高训练时间并适应更大模型的库。

目前，它支持第三方解决方案[DeepSpeed](https://github.com/microsoft/DeepSpeed)和[PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)，它们实现了论文[ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)的部分内容。

截至写作时，此提供的支持是新的且实验性的。虽然对DeepSpeed和PyTorch FSDP的支持是主动的，并且我们欢迎与此相关的问题，但我们不再支持FairScale的集成，因为它已集成在PyTorch主分支中（请参阅[PyTorch FSDP集成](#pytorch-fully-sharded-data-parallel)）。

确切的位置可能因系统而异，但`/usr/local/cuda-10.2' 是许多Unix系统上最常见的路径。当CUDA正确设置并添加到`PATH`环境变量时，可以通过执行来找到安装位置:

```bash
which nvcc
```

如果你没有在系统范围内安装CUDA，请先安装CUDA。你可以使用你喜欢的搜索引擎找到安装说明。例如，如果你使用的是Ubuntu，你可能想搜索: [ubuntu cuda 10.2 安装](https://www.google.com/search?q=ubuntu+cuda+10.2+install)。

#### 可能的问题＃2

另一个可能的常见问题是你可能在系统范围内安装了多个CUDA工具包。例如，你可能拥有:

```bash
/usr/local/cuda-10.2
/usr/local/cuda-11.0
```

在这种情况下，你需要确保`PATH`和`LD_LIBRARY_PATH`环境变量包含正确的CUDA版本的路径。通常，软件包安装程序会将这些设置为最后安装的版本。如果遇到以下问题，即使你在系统范围内安装了相关的CUDA版本，构建程序仍然无法找到正确的CUDA版本，这意味着你需要调整上述2个环境变量。

首先，你可以查看它们的内容:

```bash
echo $PATH
echo $LD_LIBRARY_PATH
```

这样你就可以了解其中的内容。

`LD_LIBRARY_PATH`可能为空。

`PATH`列出了可执行文件的位置，而`LD_LIBRARY_PATH`则是共享库的位置的查找路径。在这两种情况下，较早的路径条目优先于较晚的路径条目。`:` 用于分隔多个条目。

现在，要告诉构建程序在哪里找到特定的CUDA工具包，请将要列在最前面的所需路径插入:

```bash
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

请注意，我们没有覆盖现有的值，而是在其前面插入值。

当然，根据版本号和实际情况调整完整路径。检查你分配的目录是否确实存在。`lib64`子目录是各种CUDA `.so`对象（如`libcudart.so`）所在的位置，它们的命名可能不同，但如果存在，可以调整它以反映你的实际情况。

#### 可能的问题＃3

某些旧版本的CUDA可能拒绝使用更新的编译器进行构建。例如，你可能有`gcc-9`，但它需要`gcc-7`。

有各种方法可以解决此问题。

如果你可以安装最新的CUDA工具包，它通常应支持较新的编译器。

或者，你可以在已安装的编译器之外再安装较低版本的编译器，或者你可能已经安装了较低版本的编译器，但它不是默认版本，因此构建系统无法找到它。如果已经安装了`gcc-7`，但构建系统抱怨找不到它，可以尝试以下操作:

```bash
sudo ln -s /usr/bin/gcc-7  /usr/local/cuda-10.2/bin/gcc
sudo ln -s /usr/bin/g++-7  /usr/local/cuda-10.2/bin/g++
```

这里，我们创建了一个指向`/usr/local/cuda-10.2/bin/gcc`的指向`gcc-7`的符号链接，因为`/usr/local/cuda-10.2/bin/`应在`PATH`环境变量中（请参阅之前问题的解决方案），所以它应该找到`gcc-7`（和`g++7`），然后构建将成功。

正如总是要注意的那样，请确保修改示例中的路径以符合你的情况。


### PyTorch完全分片数据并行

为了加速在较大批次大小上训练庞大的模型，可以使用完全分片的数据并行模型。这种数据并行范式通过分片优化器状态，梯度和参数来实现更多的数据和更大的模型的适配。要了解更多信息和好处，请查看[完全分片数据并行博客](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)。我们已经将最新的PyTorch完全分片数据并行（FSDP）训练功能集成到了其中。你只需要通过配置文件启用它。

**FSDP支持的必需PyTorch版本**：PyTorch Nightly（或者在发布后阅读本文之后是1.12.0）是唯一支持带有FSDP激活的模型保存的版本。

**使用方法**：

- 确保你已经添加了分布式启动程序`-m torch.distributed.launch --nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE`，如果你还没有使用它。
- **分片策略**：
  - FULL_SHARD: 将优化器状态 + 梯度 + 模型参数分片到数据并行工作进程/GPU。为此，请将 `--fsdp full_shard` 添加到命令行参数中。
  - SHARD_GRAD_OP: 将优化器状态 + 梯度 分片到数据并行工作进程/GPU。 对此，请将 `--fsdp shard_grad_op` 添加到命令行参数中。
  - NO_SHARD: 不分片。 对此，请将 `--fsdp no_shard` 添加到命令行参数中。
- 要将参数和梯度卸载到CPU，请将 `--fsdp "full_shard offload"` 或 `--fsdp "shard_grad_op offload"` 添加到命令行参数中。
- 要自动使用`default_auto_wrap_policy`递归包装层，请将 `--fsdp "full_shard auto_wrap"` 或 `--fsdp "shard_grad_op auto_wrap"`添加到命令行参数中。
- 要同时启用CPU卸载和自动包装，请将 `--fsdp "full_shard offload auto_wrap"` 或 `--fsdp "shard_grad_op offload auto_wrap"` 添加到命令行参数中。
- 剩余的FSDP配置通过 `--fsdp_config <path_to_fsdp_config.json>` 传递。它可以是FSDP json配置文件的位置（例如，`fsdp_config.json`），也可以是已经加载的json文件作为`dict`。
  - 如果启用了自动包装，请在配置文件中指定`fsdp_transformer_layer_cls_to_wrap`。如果未指定，默认值为`model._no_split_modules`（如果可用）。这指定了要包装的变压器层类名（区分大小写），例如，[`BertLayer`]，[`GPTJBlock`]，[`T5Block`] ....。这很重要，因为共享权重的子模块（例如，嵌入层）不应该分布在不同的FSDP包装单元中。使用此策略，将为每个包含多头注意力的块以及几个MLP层的块进行包装。其余的层，包括共享的嵌入层，方便地包装在同一个最外层的FSDP单元中。因此，请用于基于变压器的模型。
  - 对于基于大小的自动包装策略，请在配置文件中添加 `fsdp_min_num_params` 。它指定了自动包装的FSDP的最小参数数。
  - 可以在配置文件中指定`fsdp_backward_prefetch`。它控制何时预提取下一组参数。`backward_pre`和`backward_pos`是可用选项。有关更多信息，请参阅`torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch`
  - 可以在配置文件中指定`fsdp_forward_prefetch`。它控制何时预提取下一组参数。如果设置为"True"，则FSDP在执行前向传递时会显式预取下一个即将到来的all-gather。
  - 可以在配置文件中指定`limit_all_gathers`。如果设置为"True"，FSDP会显式同步CPU线程，以防止过多的并行all-gathers。
  - 可以在配置文件中指定`activation_checkpointing`。如果设置为"True"，FSDP激活检查点是一种通过在反向传递期间清除某些层的激活并在需要时重新计算它们的技术。事实上，这是以时间换空间的方式，减少了内存使用。

**需要注意的重要事项**
- 它与`generate`不兼容，因此与所有seq2seq/clm脚本（翻译/摘要/clm等）中的`--predict_with_generate`也不兼容。请参考问题[#21667](https://github.com/huggingface/transformers/issues/21667)

### PyTorch/XLA完全分片数据并行

对于所有TPU用户，好消息！PyTorch/XLA现在支持FSDP。支持所有最新的完全分片数据并行（FSDP）训练。
有关更多信息，请参见[使用FSDP扩展PyTorch模型的规模化](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)和[PyTorch/XLA实现FSDP](https://github.com/pytorch/xla/tree/master/torch_xla/distributed/fsdp)
所有你需要做的就是通过配置将其启用。

**FSDP支持的必需PyTorch/XLA版本**：>=2.0

**使用方法**：

与 `Trainer` 集成了 MPS 后端，因此如果你对 MPS 后端的使用有任何问题或疑问，请务必提出问题[PyTorch GitHub](https://github.com/pytorch/pytorch/issues).


## 使用Trainer在Mac上进行加速PyTorch训练

在PyTorch v1.12发布中，开发者和研究人员可以利用Apple silcion GPU进行更快速的模型训练。这样一来，在Mac上就能够进行本地机器学习工作流程，如原型设计和微调。PyTorch在Metal Performance Shaders（MPS）中使用Apple Silicon芯片作为后端实现了这一功能，可以通过新的"`mps"`设备来使用。这将使用MPS图形框架和MPS提供的优化内核来映射计算图和基元。有关更多信息，请参阅[在Mac上加速PyTorch训练的介绍](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)和[MPS后端](https://pytorch.org/docs/stable/notes/mps.html)。 

**优点**：
- 使用户可以在本地训练更大型的网络模型或批次大小
- 减少了数据检索延迟，并且由于统一内存架构，该GPU可以直接访问完整的存储器存储。因此，改善了端到端的性能。
- 减少了云端开发的成本或额外本地GPU的需求。

**先决条件**：要安装具有mps支持的torch，请按照这篇很好的中文文章进行操作：[在Mac上给PyTorch提供GPU加速](https://writing.natpr.studio/post/2021-10-25-m1pytorch/)

**用法**：
通常情况下，如果可用，`mps`设备将默认使用，就像使用`cuda`设备一样。因此，用户无需采取任何操作。

以下是在使用Apple Silicon GPU运行官方的Glue文本分类任务的示例命令（从根目录）：

```bash
export TASK_NAME=mrpc

python examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
```

**需要注意的是**

- MPS尚未实现某些PyTorch操作，使用这些操作将会抛出错误。解决方法之一是设置环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1`，这将使用CPU执行这些操作。但仍会引发UserWarning警告。
- `gloo` 和 `nccl` 是MP的后端，则这些后端在 `mps` 设备上无法使用。目前，仅可以使用单个`mps`设备。

最后，请记住，🤗 `Trainer` 只集成了 MPS 的后端，因此如果你在使用 MPS 后端时遇到任何问题或疑问，请在 [PyTorch GitHub](https://github.com/pytorch/pytorch/issues) 上提交问题。


## 使用加速启动程序和Trainer

加速现在为Trainer提供支持。在用户的角度来看，可以期望以下内容：
- 用户可以继续使用Trainer的迭代，如FSDP、DeepSpeed通过Trainer参数，无需进行任何更改。
- 现在用户可以将加速启动程序与Trainer一起使用（推荐）。

使用加速启动程序和Trainer的步骤：
1. 确保已安装🤗 加速，否则你无法使用`Trainer`。如果没有，执行 `pip install accelerate` 。你还可以更新加速的版本： `pip install accelerate --upgrade` 。
2. 运行 `accelerate config` 并填写问卷。以下是加速配置的示例：
   a. DDP多节点多GPU配置:
    ```yaml
    compute_environment: LOCAL_MACHINE                                                                                             
    distributed_type: MULTI_GPU                                                                                                    
    downcast_bf16: 'no'
    gpu_ids: all
    machine_rank: 0 #change rank as per the node
    main_process_ip: 192.168.20.1
    main_process_port: 9898
    main_training_function: main
    mixed_precision: fp16
    num_machines: 2
    num_processes: 8
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```

b. FSDP配置：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

c. DeepSpeed配置指向一个文件：

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /home/user/configs/ds_zero3_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

d. 使用accelerate插件的DeepSpeed配置：

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 0.7
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

3. 使用除accelerate配置或launcher参数外的其他参数运行Trainer脚本。以下是使用上述FSDP配置运行`run_glue.py`脚本的示例。

```bash
cd transformers

accelerate launch \
./examples/pytorch/text-classification/run_glue.py \
--model_name_or_path bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

4. 你也可以直接使用`accelerate launch`命令参数。上面的示例将对应以下命令:

```bash
cd transformers

accelerate launch --num_processes=2 \
--use_fsdp \
--mixed_precision=bf16 \
--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP  \
--fsdp_transformer_layer_cls_to_wrap="BertLayer" \
--fsdp_sharding_strategy=1 \
--fsdp_state_dict_type=FULL_STATE_DICT \
./examples/pytorch/text-classification/run_glue.py
--model_name_or_path bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

有关更多信息，请参阅🤗加速CLI指南：[启动你的🤗加速脚本](https://huggingface.co/docs/accelerate/basic_tutorials/launch)。

移动的部分：

[<a href="./deepspeed#deepspeed-trainer-integration">DeepSpeed</a><a id="deepspeed"></a> | <a href="./deepspeed#deepspeed-installation">安装</a><a id="installation"></a> | <a href="./deepspeed#deepspeed-multi-gpu">使用多个GPU部署</a><a id="deployment-with-multiple-gpus"></a> | <a href="./deepspeed#deepspeed-one-gpu">使用一个GPU部署</a><a id="deployment-with-one-gpu"></a> | <a href="./deepspeed#deepspeed-notebook">Notebook部署</a><a id="deployment-in-notebooks"></a> | <a href="./deepspeed#deepspeed-config">配置</a><a id="configuration"></a> | <a href="./deepspeed#deepspeed-config-passing">传递配置</a><a id="passing-configuration"></a> | <a href="./deepspeed#deepspeed-config-shared">共享配置</a><a id="shared-configuration"></a> | <a href="./deepspeed#deepspeed-zero">ZeRO</a><a id="zero"></a> | <a href="./deepspeed#deepspeed-zero2-config">ZeRO-2配置</a><a id="zero-2-config"></a> | <a href="./deepspeed#deepspeed-zero3-config">ZeRO-3配置</a><a id="zero-3-config"></a> | <a href="./deepspeed#deepspeed-nvme">NVMe支持</a><a id="nvme-support"></a> | <a href="./deepspeed#deepspeed-zero2-zero3-performance">ZeRO-2与ZeRO-3性能</a><a id="zero-2-vs-zero-3-performance"></a> | <a href="./deepspeed#deepspeed-zero2-example">ZeRO-2示例</a><a id="zero-2-example"></a> | <a href="./deepspeed#deepspeed-zero3-example">ZeRO-3示例</a><a id="zero-3-example"></a> | <a href="./deepspeed#deepspeed-optimizer">优化器</a><a id="optimizer"></a> | <a href="./deepspeed#deepspeed-scheduler">调度器</a><a id="scheduler"></a> | <a href="./deepspeed#deepspeed-fp32">fp32精度</a><a id="fp32-precision"></a> | <a href="./deepspeed#deepspeed-amp">自动混合精度</a><a id="automatic-mixed-precision"></a> | <a href="./deepspeed#deepspeed-bs">批次大小</a><a id="batch-size"></a> | <a href="./deepspeed#deepspeed-grad-acc">梯度累积</a><a id="gradient-accumulation"></a> | <a href="./deepspeed#deepspeed-grad-clip">梯度裁剪</a><a id="gradient-clipping"></a> | <a href="./deepspeed#deepspeed-weight-extraction">提取模型权重</a><a id="getting-the-model-weights-out"></a>]