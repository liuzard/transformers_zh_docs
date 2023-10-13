<!--
版权所有2022 HuggingFace团队。 保留所有权利。

根据Apache许可证第2.0版（“许可证”），你除非遵守许可证，否则不得使用此文件。
你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则依据许可证分发的软件是基于“原样”原则，即没有任何形式的明示或暗示的保证或条件。关于证本身的湿提示，无论是明示或暗示的，包括但不限于对适销性、特定目的的适用性和无侵权的保证。详细信息请参阅许可证。

请注意，此文件为Markdown格式，但包含特定于我们doc-builder的语法（类似于MDX），这在你的Markdown查看器中可能无法正确呈现。
-->

# 在单个GPU上进行高效训练的方法和工具

本指南演示了你可以使用的实用技术，以通过优化内存利用率、加快训练速度或两者兼顾来提高模型训练的效率。如果你想了解在训练期间如何利用GPU，请首先参考[模型训练解剖](model_memory_anatomy.md)概念指南。本指南侧重于实用技术。

<Tip>

如果你可以访问具有多个GPU的计算机，则这些方法仍然有效，并且你还可以利用在[多GPU部分](perf_train_gpu_many.md)中概述的其他方法。

</Tip>

在训练大型模型时，应同时考虑以下两个方面：

* 数据吞吐量/训练时间
* 模型性能

最大化吞吐量（样本/秒）可以降低训练成本。通常，这通过尽可能多地利用GPU并将其填充到其极限来实现。如果所需的批次大小超出了GPU内存的限制，则可以使用内存优化技术（例如渐变累积）来帮助解决内存问题。

但是，如果首选的批次大小适合内存，那么就没有理由应用内存优化技术，因为它们可能会减慢训练速度。仅仅因为可以使用大批量大小，并不意味着必须使用。作为超参数调整的一部分，你应确定哪个批次大小产生最佳结果，然后相应地优化资源。本指南中涉及的方法和工具可以根据它们对训练过程的影响进行分类：

| 方法/工具                                                   | 提高训练速度      | 优化内存利用率            |
|:-----------------------------------------------------------|:------------------------|:-----------------------------|
| [选择批量大小](#选择批量大小)                               | 是                     | 是                          |
| [渐变累积](#渐变累积)                                      | 否                     | 是                          |
| [渐变检查点](#渐变检查点)                                  | 否                     | 是                          |
| [混合精度训练](#混合精度训练)                              | 是                     | （否）                       |
| [选择优化器](#选择优化器)                                   | 是                     | 是                          |
| [数据预加载](#数据预加载)                                     | 是                     | 否                          |
| [DeepSpeed Zero](#deepspeed-zero)                          | 否                     | 是                          |
| [torch.compile](#使用-torchcompile)                       | 是                     | 否                           |

<Tip>

注意：当使用小型模型和大批次大小进行混合精度时，将节省一些内存，但在使用大型模型和小批次大小时，内存使用量将更大。

</Tip>

你可以组合上述方法以获得累积效果。无论你是使用[`Trainer`]训练模型还是编写纯PyTorch循环，都可以使用这些技术。，在后一种情况下，你可以使用🤗Accelerate[配置这些优化。

如果这些方法无法获得足够的收益，你可以尝试以下选项：
* [查看使用高效软件预构建构建自定义Docker容器](#高效软件预构建)
* [考虑使用混合专家（MoE）模型](#专家混合)
* [将模型转换为BetterTransformer以利用PyTorch本机注意力](#使用pytorch本机注意力)

最后，即使在切换到像A100这样的服务器级GPU之后，如果以上所有方法仍然不足够，请考虑切换到多GPU设置。所有这些方法在多GPU设置中仍然有效，此外，你还可以利用在[多GPU部分](perf_train_gpu_many.md)中概述的其他并行技术。

## 选择批量大小

为了实现最佳性能，请首先确定适当的批量大小。建议使用大小为2^N的批量大小和输入/输出神经元计数。通常，它是8的倍数，但可能更高，具体取决于所使用的硬件和模型的数据类型。

有关参考，请查看NVIDIA关于全连接层输入/输出神经元计数（涉及GEMM（通用矩阵乘法））的[建议](
https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features)和[批量大小](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size)。

张量核心要求根据数据类型和硬件来定义乘数。例如，对于fp16数据类型，推荐使用8的倍数，除非使用的是A100 GPU，此时请使用64的倍数。

对于较小的参数，请考虑[维度量化效应](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization)。这是平铺发生的地方，正确的乘数可以显着加快速度。

## 渐变累积

**渐变累积**方法旨在通过小批次逐步计算渐变，而不是一次性为整个批次计算渐变。该方法通过反复在更小的批次中执行模型的前向和后向传播、并累积梯度，在此过程中积累足够数量的梯度后，执行模型的优化步骤。通过使用渐变累积，可以将**有效批次大小**增加到GPU的内存容量所限制的范围之外。然而，需要注意的是，渐变累积引入的附加前向和后向传播可能会减慢训练过程。

可以通过向[`TrainingArguments`]添加`gradient_accumulation_steps`参数来启用渐变累积：

```py
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
```

在上面的示例中，有效的批次大小为4。

或者，使用🤗Accelerate对训练循环进行全面控制。在本指南的[further down](#using-accelerate)中查找🤗Accelerate示例。

虽然建议尽可能充分利用GPU的使用率，但高数量的渐变累积步骤可能导致训练减速更明显。考虑以下示例。假设`per_device_train_batch_size=4`而没有渐变累积时达到了GPU的限制。如果你想使用大小为64的批次进行训练，请勿将`per_device_train_batch_size`设置为1，并将`gradient_accumulation_steps`设置为64。相反，保持`per_device_train_batch_size=4`，并设置`gradient_accumulation_steps=16`。这样可以获得相同的有效批次大小，同时更好地利用可用的GPU资源。

有关更多信息，请参阅以下批处理大小和渐变累积基准[RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004392537)和[A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1005033957)。

## 渐变检查点

即使将批次大小设置为1并使用渐变累积，一些大型模型仍可能遇到内存问题。这是因为还有其他组件也需要内存存储。

在前向传播过程中保存所有激活以在反向传播中计算梯度可能会导致显着的内存开销。抛弃这些激活并在反向传播期间需要时重新计算它们的替代方法会引入计算开销并减慢训练过程。

**渐变检查点**在这两种方法之间提供了一种妥协方案，并在计算图中保存了选定的激活元素，因此只需重新计算梯度所需的激活元素的一部分。有关渐变检查点的详细解释，请参阅[这篇精彩的文章](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)。

要在[`Trainer`]中启用渐变检查点，请将相应的标志传递给[`TrainingArguments`]：

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)
```

或者，使用🤗Accelerate - 在本指南的较远处查找🤗Accelerate示例。

<Tip>

尽管渐变检查点可能提高内存效率，但训练速度会减慢约20%。

</Tip>

## 混合精度训练

**混合精度训练**是一种通过使用较低精度的数值格式来优化训练模型的计算效率的技术。传统上，大多数模型使用32位浮点精度（fp32或float32）来表示和处理变量。然而，并非所有变量都需要此高精度级别才能获得准确的结果。通过将某些变量的精度降低到较低的数值格式（如16位浮点或半精度，fp16或float16），我们可以加快计算过程。由于在这种方法中，一些计算使用半精度进行，而一些计算仍然使用全精度进行，所以该方法被称为混合精度训练。

大多数情况下，混合精度训练是通过使用fp16（16位浮点数）数据类型来实现的，但某些GPU架构（如Ampere架构）提供了bf16和tf32（CUDA内部数据类型）数据类型。查看[NVIDIA博客](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)以了解这些数据类型之间的区别。

### fp16

混合精度训练的主要优势来自于在半精度（fp16）中保存激活函数的能力。尽管渐变也是以半精度计算的，但在优化步骤之前会将其转换回全精度，因此在这里不会节省内存。尽管混合精度训练可以加快计算速度，但它可能会导致使用的GPU内存增加，特别是对于较小的批次大小。这是因为模型现在在GPU上同时以16位和32位精度存在（在GPU上原始模型的1.5倍）。

要启用混合精度训练，请将`fp16`标志设置为`True`：

```py
training_args = TrainingArguments(per_device_train_batch_size=4, fp16=True, **default_args)
```

如果你更喜欢使用🤗Accelerate，请在本指南的进一步使用[ further in this guide](#using-accelerate)找到🤗Accelerate示例。

### BF16

如果你可以使用Ampere或更新的硬件，可以使用bf16进行混合精度训练和评估。尽管bf16的精度比fp16更差，但动态范围更大。在fp16中，你可以拥有的最大数字为`65535`，而超过该数字的任何数字都将导致溢出。bf16数字可以达到`3.39e+38`（！），与fp32大致相同-因为两者都使用了8位来表示数值范围。

你可以使用以下命令在🤗Trainer中启用BF16：

```python
training_args = TrainingArguments(bf16=True, **default_args)
```
### TF32

Ampere硬件使用一种被称为tf32的神奇数据类型。它具有与fp32相同的数字范围（8位），但是精度为23位（与fp16相同）而不是19位 (与fp16相同).。它是“神奇”的，是因为你可以使用与平常使用的fp32训练和/或推理代码相同的代码，并通过启用tf32支持，可以获得高达3倍的吞吐量改进。需要做的就是在代码中添加以下内容：

```
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

CUDA将自动切换到使用tf32而不是使用fp32（假设使用的GPU是Ampere系列）。

根据[NVIDIA研究](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)，绝大多数机器学习训练工作负载以tf32训练与fp32相同的困惑度和收敛。如果你已经使用fp16或bf16混合精度，则它也可以提高吞吐量。

你可以在🤗Trainer中启用此模式：

```python
TrainingArguments(tf32=True, **default_args)
```

<Tip>

无法直接通过`tensor.to(dtype=torch.tf32)`访问tf32，因为它是内部CUDA数据类型。你需要`torch >= 1.7`才能使用tf32数据类型。

</Tip>

有关tf32与其他精度的更多信息，请参见以下基准测试：
[RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803)和
[A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1004543189)。

## 优化器选择

用于训练变换器模型的最常用优化器是Adam或AdamW（带有权重衰减的Adam）。Adam通过存储先前梯度的滚动平均值实现良好的收敛；然而，它增加了约模型参数数量的内存占用。为了解决这个问题，你可以使用替代的优化器。例如，如果你安装了[NVIDIA/apex](https://github.com/NVIDIA/apex)，`adamw_apex_fused`将为你提供所有支持的AdamW优化器中的最快训练体验。

[`Trainer`]集成了各种优化器，可以直接使用：`adamw_hf`、`adamw_torch`、`adamw_torch_fused`、`adamw_apex_fused`、`adamw_anyprecision`、`adafactor`或`adamw_bnb_8bit`。可以通过第三方实现插入更多优化器。

让我们更详细地看一下替代AdamW优化器的两个选择：
1.`adafactor` 可在[`Trainer`]中使用
2. `adamw_bnb_8bit` 在Trainer中也可用，但以下是提供的第三方整合。


对比而言，对于一个3B参数模型，如“t5-3b”：
* 一个标准的AdamW优化器需要24GB的GPU内存，因为它对每个参数使用了8字节（8*3 => 24GB）
* Adafactor优化器需要超过12GB。它对每个参数使用略多于4字节，即4*3，还有一些额外的空间。
* 如果所有优化器状态都被量化，8bit BNB量化优化器只会使用6GB的内存。（2*3）。

### Adafactor

Adafactor不会为每个权重矩阵中的每个元素保留滚动平均值。相反，它保留了聚合信息（逐行和逐列的滚动平均值之和），从而显著减少了其占用空间。然而，与Adam相比，在某些情况下，Adafactor可能收敛较慢。

你可以通过在[`TrainingArguments`]中设置`optim="adafactor"`来切换到Adafactor：

```py
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adafactor", **default_args)
```

结合其他方法（梯度积累、梯度检查点和混合精度训练），可以在保持吞吐量的同时实现最多3倍的改进！然而，正如前面提到的，Adafactor的收敛性可能比Adam差。

### 8位Adam

与Adafactor不同，8位Adam保留完整状态并对其进行量化。量化意味着以较低的精度存储状态，并仅在优化时进行解量化。这类似于混合精度训练的思路。

要使用`adamw_bnb_8bit`，你只需要在[`TrainingArguments`]中设置`optim="adamw_bnb_8bit"`：

```py
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adamw_bnb_8bit", **default_args)
```

然而，为了示范目的，我们还可以使用第三方实现的8位优化器，看看如何集成该优化器。

首先，按照GitHub [repo](https://github.com/TimDettmers/bitsandbytes)中的安装指南安装实现8位Adam优化器的`bitsandbytes`库。

接下来，你需要初始化优化器。这涉及两个步骤：
* 首先，将模型的参数分组为两组-一组应用权重衰减，另一组不应用权重衰减。通常，偏差和层规范化参数不会应用权重衰减。
* 然后进行一些参数处理，以使用先前使用的AdamW优化器相同的参数。

```py
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)

decay_parameters = get_parameter_names(model, [nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": training_args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer_kwargs = {
    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    "eps": training_args.adam_epsilon,
}
optimizer_kwargs["lr"] = training_args.learning_rate
adam_bnb_optim = bnb.optim.Adam8bit(
    optimizer_grouped_parameters,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    lr=training_args.learning_rate,
)
```

最后，将自定义优化器作为参数传递给`Trainer`：

```py
trainer = Trainer(model=model, args=training_args, train_dataset=ds, optimizers=(adam_bnb_optim, None))
```

结合其他方法（梯度积累、梯度检查点和混合精度训练），你可以期望获得约3倍的内存改进，甚至比使用Adafactor时具有稍高的吞吐量。

### multi_tensor

pytorch-nightly引入了`torch.optim._multi_tensor`，可以显著加快大量小特征张量的优化器速度。它最终将成为默认选项，但如果你想提前尝试它，请查看此GitHub [issue](https://github.com/huggingface/transformers/issues/9965)。

## 数据预加载

实现出色的训练速度的一个重要要求是能够以GPU能够处理的最大速度馈送数据。默认情况下，所有操作都在主进程中进行，可能无法快速从磁盘读取数据，从而造成瓶颈，导致GPU利用率不高。通过配置以下参数来减少瓶颈：

- `DataLoader(pin_memory=True, ...)` - 确保数据预加载到CPU上的固定内存中，通常会导致从CPU到GPU内存的传输速度大大提高。
- `DataLoader(num_workers=4, ...)` - 派生多个工作线程以更快地预加载数据。在训练过程中，观察GPU利用率统计数据；如果离100％有一定差距，请尝试增加工作线程数。当然，问题可能不在这里，因此增加工作线程数不一定会带来更好的性能。

使用[`Trainer`]时，对应的[`TrainingArguments`]参数是：`dataloader_pin_memory`（默认为`True`）和`dataloader_num_workers`（默认为`0`）。

## DeepSpeed ZeRO

DeepSpeed是一个与🤗Transformers和🤗Accelerate集成的开源深度学习优化库。它提供了一系列功能和优化，旨在改进大规模深度学习训练的效率和可扩展性。

如果你的模型适合于单个GPU并且有足够的空间来放置较小的批次大小，则不需要使用DeepSpeed，因为它只会使事情变慢。然而，如果模型无法适应单个GPU，或者无法放置较小的批次，则可以利用DeepSpeed的ZeRO + CPU Offload或NVMe Offload来处理更大的模型。在这种情况下，你需要单独[安装库](main_classes/deepspeed#installation)，然后遵循一个配置文件并启动DeepSpeed的指南：

* 对于DeepSpeed与[`Trainer`]的完整指南，请查阅[相应的文档](main_classes/deepspeed) ，特别是[单个GPU的部署部分](main_classes/deepspeed#deployment-with-one-gpu)。要在笔记本中使用DeepSpeed，需要进行一些调整；请查阅[对应指南](main_classes/deepspeed#deployment-in-notebooks)。
* 如果你更喜欢使用🤗Accelerate，请参考[🤗Accelerate DeepSpeed指南](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)。

## 使用torch.compile

PyTorch 2.0引入了一个新的编译函数，它不需要对现有的PyTorch代码进行任何修改，但可以通过添加一行代码来优化你的代码：`model = torch.compile(model)`。

如果使用[`Trainer`]，你只需要在[`TrainingArguments`]中传递`torch_compile`选项：

```python
training_args = TrainingArguments(torch_compile=True, **default_args)
```

`torch.compile`使用Python的帧评估API来自动从现有的PyTorch程序中创建图形。在捕获图形之后，可以部署不同的后端将图形降到优化引擎。你可以在[PyTorch文档](https://pytorch.org/get-started/pytorch-2.0/)中找到更多详细信息和基准测试。

`torch.compile`具有不断增长的后端列表，可以通过调用`torchdynamo.list_backends()`找到。每个后端都有其可选的依赖项。

通过在[`TrainingArguments`]中指定要使用的后端的方式选择要使用的后端。最常用的几个后端是：

**调试后端**：
* `dynamo.optimize("eager")` - 使用PyTorch运行提取的GraphModule，这对于调试TorchDynamo问题非常有用。
* `dynamo.optimize("aot_eager")` - 使用AotAutograd而没有编译器的AotAutograd's提取的前向和后向图的PyTorch eager运行。这对于调试非常有用，不太可能带来速度提升。

**训练和推断后端**：
* `dynamo.optimize("inductor")` - 使用具有AotAutograd和cudagraphs的TorchInductor后端，通过利用codegened Triton内核平衡地训练每个专家的门控函数来进行训练。[了解更多](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
* `dynamo.optimize("nvfuser")` - 使用TorchScript的nvFuser。[了解更多](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_nvfuser")` - 使用AotAutograd的nvFuser。[了解更多](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_cudagraphs")` - 使用AotAutograd的cudagraphs。[了解更多](https://github.com/pytorch/torchdynamo/pull/757)

**仅推断后端**：
* `dynamo.optimize("ofi")` - 使用Torchscript的optimize_for_inference。[了解更多](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
* `dynamo.optimize("fx2trt")` - 使用Nvidia TensorRT进行推断优化。[了解更多](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
* `dynamo.optimize("onnxrt")` - 使用ONNXRT进行CPU/GPU上的推断。[了解更多](https://onnxruntime.ai/)
* `dynamo.optimize("ipex")` - 使用IPEX进行CPU上的推断。[了解更多](https://github.com/intel/intel-extension-for-pytorch)

要使用`torch.compile`与🤗Transformers的示例，请查看本文档中关于使用最新的PyTorch 2.0功能[Fine-tuning a BERT model for Text Classification using the newest PyTorch 2.0 features]的[博客文章](https://www.philschmid.de/getting-started-pytorch-2-0-transformers)。

## 使用🤗Accelerate

通过[🤗Accelerate](https://huggingface.co/docs/accelerate/index)，你可以使用以上方法，并完全控制训练循环，实质上可以使用纯粹的PyTorch编写循环，只需进行一些细微的修改。

假设你已经将[`TrainingArguments`]中的方法组合如下：

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)
```

使用🤗Accelerate的完整示例训练循环只有几行代码：

```py
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

accelerator = Accelerator(fp16=training_args.fp16)
model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

model.train()
for step, batch in enumerate(dataloader, start=1):
    loss = model(**batch).loss
    loss = loss / training_args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % training_args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

首先，我们将数据集包装在[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)中。然后，我们可以通过调用模型的[`~PreTrainedModel.gradient_checkpointing_enable`]方法来启用梯度检查点。在初始化[`Accelerator`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator)时，我们可以指定是否使用混合精度训练，并且它将在[`prepare`]调用中为我们处理。在[`prepare`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.prepare)调用期间，数据加载器也将在使用多个GPU时分布在工作进程中。我们从之前示例中使用相同的[8位优化器](#8-bit-adam)。

最后，我们可以添加主要的训练循环。请注意，`backward`调用是由🤗Accelerate处理的。我们还可以看到梯度累积的工作原理：我们将损失归一化，因此在累积结束时得到平均值，并且一旦我们进行足够的步骤，就进行优化。

在🤗Accelerate中，通过少量的代码即可实现这些优化技术，并且具有更灵活的训练循环。要了解所有功能的完整文档，请查看[Accelerate文档](https://huggingface.co/docs/accelerate/index)。

## 高效的软件预构建

PyTorch的[pip和conda构建](https://pytorch.org/get-started/locally/#start-locally)已经预先构建了cuda toolkit，这足以运行PyTorch，但如果你需要构建cuda扩展，则不足够。

有时候可能需要额外的努力来预先构建某些组件。例如，如果你使用的是不预先编译的库（如`apex`），可能需要额外的努力。在其他情况下，找到如何在系统范围内安装正确的cuda toolkit可能很复杂。为了解决这些情况，PyTorch和NVIDIA发布了新版本的NGC Docker容器，其中已经预先构建了所有内容。你只需将程序安装在其中，它就可以直接运行。

如果你想调整pytorch源代码和/或进行新的定制构建，这种方法也很有用。
要找到所需的Docker映像版本，请查看[PyTorch发布说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)，选择最新的月度版本之一。进入所需版本的发布说明，检查环境组件是否符合你的需求（包括NVIDIA驱动程序要求！），然后在该文档的顶部转到相应的NGC页面。如果因某种原因迷失方向，请查看[所有PyTorch NGC图像的索引](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)。

接下来，请按照下载和部署Docker映像的说明进行操作。

## 专家组合

一些最近的论文报道，结合专家组合（Mixture of Experts，MoE）技术将Transformer模型中的训练速度提高了4-5倍，并实现了更快的推断。

由于发现更多的参数可以带来更好的性能，此技术允许将参数数量提高一个数量级，而不增加训练成本。

在这种方法中，每个FFN层被一个MoE层取代，该MoE层由许多专家组成，具有根据输入标记在序列中的位置平衡训练的门控函数。

![MoE Transformer 2x block](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perf-moe-transformer.png)

（来源：[GLAM](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)）

可以在下面列出的论文中找到详细的信息和比较表格：

- ["Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"](https://arxiv.org/abs/2101.03961) by Ben-Zaken et al.
- ["GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"](https://arxiv.org/abs/2006.16668) by Fedus et al.
- ["MixNMatch: Training a Convolutional Neural Network with a Switchable Mixture of Experts"](https://arxiv.org/abs/2002.03598) by Aghajanyan et al.
- ["Scalable Mixture Models for Deep Learning"](https://arxiv.org/abs/2010.09161) by Chen et al.

以上是翻译结果，仅供参考。

这种方法的主要缺点是需要大量的GPU内存，几乎比其密集等效模型多一个数量级。有多种蒸馏和方法被提出来解决这种更高的内存需求。

然而，存在直接的权衡，你可以使用少量具有2-3倍较小基础模型的专家，而不是数十或数百个专家，从而得到一个5倍较小的模型，适度提高训练速度，同时适度提高内存需求。

大多数相关论文和实现都是基于Tensorflow/TPUs的：

- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [GLaM: Generalist Language Model (GLaM)](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)

对于PyTorch，DeepSpeed也构建了一个：[DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596)，[Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/) - 博客文章：[1](https://www.microsoft.com/en-us/research/blog/deepspeed-powers-8x-larger-moe-model-training-with-high-performance/)，[2](https://www.microsoft.com/en-us/research/publication/scalable-and-efficient-moe-training-for-multitask-multilingual-models/)以及具有大型基于transformer的自然语言生成模型的特定部署：[博客文章](https://www.deepspeed.ai/news/2021/12/09/deepspeed-moe-nlg.html)，[Megatron-Deepspeed分支](Thttps://github.com/microsoft/Megatron-DeepSpeed/tree/moe-training)。

## 使用PyTorch原生注意力和Flash Attention

PyTorch 2.0发布了一个原生的[`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA)，
它允许使用融合的GPU内核，例如[内存高效注意力](https://arxiv.org/abs/2112.05682)和[闪存注意力](https://arxiv.org/abs/2205.14135)。

在安装了[`optimum`](https://github.com/huggingface/optimum)软件包之后，可以替换相关的内部模块以使用PyTorch的原生注意力：

```python
model = model.to_bettertransformer()
```

一旦转换完成，可以像往常一样训练模型。

<Tip warning={true}>

如果没有提供`attention_mask`，PyTorch原生的`scaled_dot_product_attention`运算符只能调度到Flash Attention。

默认情况下，在训练模式下，BetterTransformer集成会**放弃对掩码的支持，并且只能用于不需要批量训练的填充掩码的训练**。例如，这适用于掩码语言建模或因果语言建模。BetterTransformer不适合在需要填充掩码的任务上微调模型。

</Tip>

阅读这篇[博文](https://pytorch.org/blog/out-of-the-box-acceleration/)，了解更多关于SDPA的加速和节省内存的信息。