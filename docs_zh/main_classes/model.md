# 模型

基类 [`PreTrainedModel`], [`TFPreTrainedModel`], 和 [`FlaxPreTrainedModel`] 实现了加载/保存模型的常用方法，可以从本地文件或目录加载模型，或从库提供的预训练模型配置中加载模型（从HuggingFace的AWS S3存储库下载）。

[`PreTrainedModel`] 和 [`TFPreTrainedModel`] 还实现了一些对所有模型都通用的方法，包括：

- 在词汇表中添加新词时调整输入标记嵌入的大小
- 对模型的注意力头进行修剪

每个模型通用的其他方法定义在[`~modeling_utils.ModuleUtilsMixin`]（用于PyTorch模型）和[`~modeling_tf_utils.TFModuleUtilsMixin`]（用于TensorFlow模型）中，或针对文本生成，还有[`~generation.GenerationMixin`]（用于PyTorch模型），[`~generation.TFGenerationMixin`]（用于TensorFlow模型）和[`~generation.FlaxGenerationMixin`]（用于Flax/JAX模型）。

## PreTrainedModel

[[autodoc]] PreTrainedModel
    - push_to_hub
    - all

<a id='from_pretrained-torch-dtype'></a>

### 加载大型模型

在Transformers 4.20.0中，[`~PreTrainedModel.from_pretrained`] 方法已经进行了重新设计，以适应使用[Accelerate](https://huggingface.co/docs/accelerate/big_modeling)加载大型模型。这需要 Accelerate >= 0.9.0 和 PyTorch >= 1.9.0。不再是创建完整的模型，然后在其中加载预训练的权重（这将占用内存中模型大小的两倍，一半用于随机初始化的模型，一半用于权重），而是可以选择创建一个空模型壳，然后在加载预训练权重时才实例化其参数。

可以通过 `low_cpu_mem_usage=True` 来激活该选项。模型首先在Meta设备上（用空的权重）创建，然后将状态字典加载到其中（对于分片检查点而言，逐片加载）。这样，使用的最大 RAM 仅为模型的完整大小。

```py
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", low_cpu_mem_usage=True)
```

此外，如果模型无法完全适配内存（目前仅适用于推理），您可以直接将模型放置在不同的设备上。通过使用 `device_map="auto"`，Accelerate 将确定在哪一个设备上放置每个图层，以最大化使用速度最快的设备（GPU），并将剩余部分卸载到CPU，甚至硬盘上（如果您没有足够的GPU内存或CPU内存）。即使模型分布在多个设备上，它也会正常运行。

当传递 `device_map` 时，`low_cpu_mem_usage` 会自动设置为 `True`，所以不需要指定它：

```py
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

您可以通过查看其 `hf_device_map` 属性来了解模型如何在设备上分割：

```py
t0pp.hf_device_map
```

```python out
{'shared': 0,
 'decoder.embed_tokens': 0,
 'encoder': 0,
 'decoder.block.0': 0,
 'decoder.block.1': 1,
 'decoder.block.2': 1,
 'decoder.block.3': 1,
 'decoder.block.4': 1,
 'decoder.block.5': 1,
 'decoder.block.6': 1,
 'decoder.block.7': 1,
 'decoder.block.8': 1,
 'decoder.block.9': 1,
 'decoder.block.10': 1,
 'decoder.block.11': 1,
 'decoder.block.12': 1,
 'decoder.block.13': 1,
 'decoder.block.14': 1,
 'decoder.block.15': 1,
 'decoder.block.16': 1,
 'decoder.block.17': 1,
 'decoder.block.18': 1,
 'decoder.block.19': 1,
 'decoder.block.20': 1,
 'decoder.block.21': 1,
 'decoder.block.22': 'cpu',
 'decoder.block.23': 'cpu',
 'decoder.final_layer_norm': 'cpu',
 'decoder.dropout': 'cpu',
 'lm_head': 'cpu'}
```

您还可以按照相同的格式编写自己的设备映射（一个将图层名称映射到设备的字典）。它应该将模型的所有参数映射到特定设备，但如果该层完全位于同一设备上，则不必详细说明一个层的所有子模块放在哪个设备上。例如，以下设备映射对T0pp来说是有效的（只要您具有GPU内存）：

```python
device_map = {"shared": 0, "encoder": 0, "decoder": 1, "lm_head": 1}
```

减少模型内存影响的另一种方法是以较低的精度数据类型（如 `torch.float16`）实例化模型，或使用下面描述的直接量化技术。

### 模型实例化数据类型

在PyTorch中，模型通常以 `torch.float32` 格式进行实例化。如果尝试加载权重为fp16的模型，则可能会遇到问题，因为它需要两倍的内存。为了克服这个限制，可以使用 `torch_dtype` 参数显式传递所需的 `dtype`：

```python
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype=torch.float16)
```

或者，如果要始终以最佳的内存模式加载模型，则可以使用特殊值 `"auto"`，然后 `dtype` 将从模型的权重自动派生：

```python
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype="auto")
```

从头开始实例化的模型也可以通过以下方式告知要使用的 `dtype`：

```python
config = T5Config.from_pretrained("t5")
model = AutoModel.from_config(config)
```

由于PyTorch的设计限制，此功能仅适用于浮点数字类型。

## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

[[autodoc]] TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

[[autodoc]] FlaxPreTrainedModel
    - push_to_hub
    - all

## 推送到 Hub

[[autodoc]] utils.PushToHubMixin

## 分片检查点

[[autodoc]] modeling_utils.load_sharded_checkpoint