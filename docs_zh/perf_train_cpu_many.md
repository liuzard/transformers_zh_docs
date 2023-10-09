<!--版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证Version 2.0（“许可证”）许可使用本文件；您除非符合许可证要求，否则不得使用本文件。您可以从以下地址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面约定，依据许可证分发的软件是基于“按原样”提供的，无论是明示还是暗示的，不提供任何担保或陈述，也不提供任何条件的保证。请注意，本文件采用Markdown格式，但包含我们文档生成工具的特定语法（类似于MDX），在您的Markdown查看器中可能无法正确渲染。

-->

# 多CPU高效训练

当单个CPU训练速度太慢时，我们可以使用多个CPU。本指南主要介绍基于PyTorch的DDP方法，可以高效实现分布式CPU训练。

## 适用于PyTorch的Intel® oneCCL绑定

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL)（集体通信库）是一个用于高效分布式深度学习训练的库，实现了如allreduce、allgather、alltoall等集合通信功能。有关oneCCL的更多信息，请参阅[oneCCL文档](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)和[oneCCL规范](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)。

模块`oneccl_bindings_for_pytorch`（版本1.12之前的名称为`torch_ccl`）实现了PyTorch C10D ProcessGroup API，并且可以作为外部ProcessGroup动态加载，目前仅适用于Linux平台。

请查看[oneccl_bind_pt](https://github.com/intel/torch-ccl)获取更详细的信息。

### 安装适用于PyTorch的Intel® oneCCL绑定:

以下Python版本的Wheel文件可用：

| 扩展版本 | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :------: | :--------: | :--------: | :--------: | :--------: | :---------: |
|  1.13.0  |            |     √      |     √      |     √      |      √      |
| 1.12.100 |            |     √      |     √      |     √      |      √      |
|  1.12.0  |            |     √      |     √      |     √      |      √      |
|  1.11.0  |            |     √      |     √      |     √      |      √      |
|  1.10.0  |     √      |     √      |     √      |     √      |             |

```
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```
其中`{pytorch_version}`应为您的PyTorch版本，例如1.13.0。
请参阅[oneccl_bind_pt安装方法](https://github.com/intel/torch-ccl)获取更多方法。
oneCCL和PyTorch的版本必须匹配。

<Tip warning={true}>

oneccl_bindings_for_pytorch 1.12.0 的预构建Wheel文件与 PyTorch 1.12.1 不兼容（适用于 PyTorch 1.12.0）。
PyTorch 1.12.1 可与 oneccl_bindings_for_pytorch 1.12.100 配合使用。

</Tip>

## Intel® MPI库
使用这个基于标准的MPI实现，在Intel®体系结构上提供灵活、高效和可扩展的集群通信。这个组件是Intel® oneAPI HPC Toolkit的一部分。

oneccl_bindings_for_pytorch和MPI工具集同时安装。使用之前需要设置环境变量。

对于Intel® oneCCL >= 1.12.0
```
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

对于版本< 1.12.0的Intel® oneCCL
```
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

#### IPEX安装:

IPEX为CPU训练提供了Float32和BFloat16的性能优化，请参考[单CPU部分](perf_train_cpu.md)。

以下“Trainer中的使用方法”以Intel® MPI库中的mpirun为例。

## Trainer中的使用方法
要在Trainer中启用多CPU分布式训练并使用ccl后端，用户应在命令参数中添加**`--ddp_backend ccl`**。

我们以[question-answering示例](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)为例进行说明。

下面的命令将启用在一个Xeon节点上使用2个进程进行训练，每个进程在一个套接字上运行。可以根据需要调整OMP_NUM_THREADS/CCL_WORKER_COUNT变量以实现最佳性能。
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=127.0.0.1
 mpirun -n 2 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex
```
下面的命令将在两个Xeon节点（node0和node1）上总共使用四个进程进行训练，其中node0为主进程，ppn（每个节点的进程数）设置为2，每个套接字上运行一个进程。可以根据需要调整OMP_NUM_THREADS/CCL_WORKER_COUNT变量以实现最佳性能。

在node0上，您需要创建一个包含每个节点的IP地址的配置文件（例如hostfile），并将该配置文件路径作为参数传递。
```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```
现在，在node0上运行以下命令，将在node0和node1中启用4DDP，并使用BF16自动混合精度：
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
 mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex \
 --bf16
```
