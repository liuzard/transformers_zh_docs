<!--版权2023年HuggingFace团队。版权所有。

根据Apache许可证，第2版（“许可证”）获得许可;除非符合许可证的规定，否则你将不能使用此文件。
你可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，否则根据许可证分发的软件是以“按原样”分发的，
不附带任何明示或暗示的保证或条件。请参阅许可证以获取
特定语言下授权的权限和限制。

⚠️请注意，此文件采用Markdown格式，但包含我们文档构建器（类似于MDX）的特定语法，可能在你的Markdown查看器中无法正确呈现。

-->

# 量化🤗Transformers模型

## `AutoGPTQ`集成

🤗Transformers已经集成了`optimum` API，用于对语言模型执行GPTQ量化。你可以在8、4、3甚至2个比特中加载和量化模型，而性能下降很小，推理速度更快！这是由大多数GPU硬件支持的。

要了解有关量化模型的更多信息，请查看：
- [GPTQ](https://arxiv.org/pdf/2210.17323.pdf)论文
- `optimum` [指南](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)上的GPTQ量化
- [`AutoGPTQ`](https://github.com/PanQiWei/AutoGPTQ)库用作后端

### 要求

要运行下面的代码，你需要安装以下要求：

- 安装最新的`AutoGPTQ`库
`pip install auto-gptq`

- 安装最新的`optimum`源码
`pip install git+https://github.com/huggingface/optimum.git`

- 安装最新的`transformers`源码
`pip install git+https://github.com/huggingface/transformers.git`

- 安装最新的`accelerate`库
`pip install --upgrade accelerate`

请注意，GPTQ集成目前仅支持文本模型，对于视觉、语音或多模式模型，你可能会遇到意外行为。

### 加载和量化模型

GPTQ是一种量化方法，在使用量化模型之前需要进行权重校准。如果你想从头开始量化transformers模型，生成量化模型可能需要一些时间（对于`facebook/opt-350m`模型在Google colab上大约需要5分钟）。

因此，有两种不同的情况需要使用GPTQ量化模型。第一种情况是加载已经由其他用户量化的模型，这些模型可以在Hub上找到；第二种情况是量化自己的模型并保存或将其推送到Hub，以便其他用户也可以使用。

#### GPTQ配置

为了加载和量化模型，你需要创建一个[`GPTQConfig`]。你需要传递`bits`的数量、一个`dataset`用于校准量化以及模型的`tokenizer`以准备数据集。

```python
model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
```

请注意，你可以将自己的数据集作为字符串列表传递。然而，强烈建议使用GPTQ论文中的数据集。
```python
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
quantization = GPTQConfig(bits=4, dataset = dataset, tokenizer=tokenizer)
```

#### 量化

你可以使用`from_pretrained`来量化模型，并设置`quantization_config`。

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=gptq_config)
```
请注意，你将需要一个GPU来量化模型。我们将模型放到CPU上，然后将模块来回移动到GPU上以进行量化。

如果你想在使用CPU隐藏时最大限度地使用GPU，可以设置`device_map = "auto"`。
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)
```
请注意，目前不支持磁盘映射。此外，如果由于数据集而内存不足，你可能需要在`from_pretained`中传递`max_memory`。请查阅此[指南](https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map)了解有关`device_map`和`max_memory`的更多信息。

<Tip warning={true}>
目前，GPTQ量化仅适用于文本模型。此外，根据硬件的不同，量化过程可能需要很长时间（175B模型 = 使用NVIDIA A100的4个GPU小时）。如果不是，你可以在GitHub上提交需求。
</Tip>

### 将量化模型推送到🤗Hub

你可以像将任何🤗模型推送到Hub一样推送量化模型，使用`push_to_hub`方法。量化配置将被保存并与模型一起推送。

```python
quantized_model.push_to_hub("opt-125m-gptq")
tokenizer.push_to_hub("opt-125m-gptq")
```

如果你想在本地保存量化模型，也可以使用`save_pretrained`：
```python
quantized_model.save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")
```

请注意，如果你使用`device_map`对模型进行了量化，请确保在保存之前将整个模型移动到其中一个GPU或`cpu`。
```python
quantized_model.to("cpu")
quantized_model.save_pretrained("opt-125m-gptq")
```

### 从🤗Hub加载量化模型

你可以使用`from_pretrained`从Hub加载量化模型。
通过检查模型配置对象中是否存在属性`quantization_config`，确保推送的权重已经进行了量化。

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq")
```

如果你想更快地加载模型，而且不会分配额外的内存，请在量化模型上使用`device_map`参数。请确保已安装`accelerate`库。
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto")
```

### 用于更快推断的快速推断内核

对于4比特模型，你可以使用快速推断内核以提高推断速度。它默认为激活状态。你可以通过在[`GPTQConfig`]中传递`disable_exllama`来更改此行为。这将覆盖在配置中存储的量化配置。请注意，只能覆盖与内核相关的属性。此外，如果要使用exllama内核，需要将整个模型放在gpu上。

```py
import torch
gptq_config = GPTQConfig(bits=4, disable_exllama=False)
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config = gptq_config)
```

请注意，目前仅支持4比特模型。此外，如果要微调具有PEFT的量化模型，建议禁用exllama内核。

#### 对量化模型进行微调

在Hugging Face生态系统中正式支持adapter后，你可以微调已使用GPTQ进行量化的模型。
有关详细信息，请参阅`peft`[库](https://github.com/huggingface/peft)。

### 示例演示

查看Google Colab的[notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing)以了解如何使用GPTQ量化模型以及如何使用peft对量化模型进行微调。

### GPTQConfig

[[autodoc]] GPTQConfig


## `bitsandbytes`集成

🤗Transformers与`bitsandbytes`上最常用的模块紧密集成。只需几行代码，你就可以以8位精度加载你的模型。
自“0.37.0”版本以来，`bitsandbytes`已经支持大多数GPU硬件。

了解有关量化方法的详细信息，请查看[LLM.int8()](https://arxiv.org/abs/2208.07339)论文，或有关此合作的[博文](https://huggingface.co/blog/hf-bitsandbytes-integration)。

从“0.39.0”版本开始，你可以使用4位量化加载支持`device_map`的任何模型，从而提供FP4数据类型。

如果你想量化自己的pytorch模型，请查看🤗加速库的[此文档](https://huggingface.co/docs/accelerate/main/zh/usage_guides/quantization)。 

以下是使用`bitsandbytes`集成可以实现的功能

### 一般用法

你可以通过在调用[`~PreTrainedModel.from_pretrained`]方法时使用`load_in_8bit`或`load_in_4bit`参数，将模型量化为8位精度。只要你的模型支持使用🤗加速加载并包含`torch.nn.Linear`层即可。这对于任何模态也适用。

```python
from transformers import AutoModelForCausalLM

model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_8bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_4bit=True)
```

默认情况下，所有其他模块（如`torch.nn.LayerNorm`）将转换为`torch.float16`，但是如果要更改它们的`dtype`，可以覆盖`torch_dtype`参数：

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM

>>> model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_8bit=True, torch_dtype=torch.float32)
>>> model_8bit.model.decoder.layers]lstm(12/987)(
                               lstm_feedfo1r
```


### 使用FP4量化

#### 要求

在运行下面的代码片段之前，请确保已安装以下要求。

- 最新的`bitsandbytes`库
`pip install bitsandbytes>=0.39.0`

- 安装最新的`accelerate`
`pip install --upgrade accelerate`

- 安装最新的`transformers`
`pip install --upgrade transformers`

#### 提示和最佳实践

- **高级用法：**请参考[此Google Colab笔记本](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf)，了解4位量化的高级用法和所有可能的选项。

- **使用`batch_size = 1`进行更快的推理：**自`bitsandbytes`的`0.40.0`版本以来，如果`batch_size = 1`，你可以获得快速推理的好处。查看[这些发布说明](https://github.com/TimDettmers/bitsandbytes/releases/tag/0.40.0)，确保你的版本大于`0.40.0`，以便从早期开始无缝地使用此功能。

- **训练：**根据[QLoRA论文](https://arxiv.org/abs/2305.14314)，对于训练4位基本模型（例如使用LoRA适配器），应使用`bnb_4bit_quant_type='nf4'`。

- **推理：**对于推理，`bnb_4bit_quant_type`对性能没有太大影响。但是，为了与模型的权重保持一致，请确保使用相同的`bnb_4bit_compute_dtype`和`torch_dtype`参数。

#### 在4位中加载大模型

通过在调用`.from_pretrained`方法时使用`load_in_4bit=True`，你可以将内存使用减少约4倍（大致）。

```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
```

<Tip warning={true}>

请注意，一旦模型已经以4位加载，当前无法将量化的权重推送到Hub上。请注意，尚不支持对4位权重进行训练。但是，你可以使用4位模型来训练额外的参数，这将在下一节中介绍。

</Tip>

### 在8位中加载大模型

通过在调用`.from_pretrained`方法时使用`load_in_8bit=True`参数，你可以将内存需求减小约一半。

```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)
```

然后，像通常使用[`PreTrainedModel`]一样使用你的模型。

你可以使用`get_memory_footprint`方法检查模型的内存占用情况。

```python
print(model.get_memory_footprint())
```

通过此集成，我们能够在较小的设备上加载大模型并顺利运行。

<Tip warning={true}>

请注意，一旦模型已经以8位加载，目前无法将量化的权重推送到Hub上，除非使用最新的`transformers`和`bitsandbytes`。请注意，尚不支持对8位权重进行训练。但是，你可以使用8位模型来训练额外的参数，这将在下一节中介绍。
此外，请注意，`device_map`是可选的，但是将`device_map = 'auto'`设置为推理是最好的，因为它将有效地将模型分配到可用资源。

</Tip>

#### 高级用例

在这里，我们将介绍你可以使用FP4量化执行的一些高级用例

##### 更改计算数据类型

计算数据类型用于更改计算过程中要使用的数据类型。例如，隐藏状态可以是`float32`，但是可以使用bf16进行计算以加快速度。默认情况下，计算数据类型设置为`float32`。

```python
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

##### 使用NF4（Normal Float 4）数据类型

你还可以使用NF4数据类型，该数据类型是针对使用正态分布初始化的权重而设计的新的4位数据类型。运行以下代码：

```python
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

##### 使用嵌套量化进行更节省内存的推理

我们还建议用户使用嵌套量化技术。这样可以节省更多的内存，而不会有任何额外的性能——根据我们的实证观察，这使得在NVIDIA-T4 16GB上使用`sequence_length = 1024`、`batch_size = 1`和`gradient accumulation_steps = 4`微调llama-13b模型成为可能。

```python
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
```


### 将量化模型推送到🤗Hub

你可以通过简单地使用`push_to_hub`方法将量化模型推送到Hub上。这将首先推送量化配置文件，然后推送量化模型权重。
确保使用 `bitsandbytes>0.37.2`（在撰写本文时，我们在`bitsandbytes==0.38.0.post1`上进行了测试），以便能够使用此功能。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model.push_to_hub("bloom-560m-8bit")
```

<Tip warning={true}>

强烈鼓励将8位模型推送到Hub上以适应大型模型。这将使社区能够受益于内存占用的减少，例如在Google Colab上加载大型模型。

</Tip>

### 从🤗Hub加载量化模型

你可以使用`from_pretrained`方法从Hub加载量化模型。确保推送的权重已经被量化，检查模型配置对象中是否存在`quantization_config`属性。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```
请注意，在这种情况下，你不需要指定`load_in_8bit=True`参数，但是你需要确保已安装了`bitsandbytes`和`accelerate`。
还要注意，`device_map`是可选的，但是设置`device_map = 'auto'`对于推断来说是首选的，因为它将在可用资源上高效地分派模型。

### 高级用例

本节旨在为希望在加载和运行8位模型之外探索更多功能的高级用户提供。

#### 在`cpu`和`gpu`之间进行卸载

其中一个高级用例是能够加载模型并在`CPU`和`GPU`之间分派权重。请注意，在CPU上分派的权重**不会**转换为8位，而是保持为`float32`。此功能适用于希望适应非常大的模型并在GPU和CPU之间分派模型的用户。

首先，从`transformers`中加载[`BitsAndBytesConfig`]并将属性`llm_int8_enable_fp32_cpu_offload`设置为`True`：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

假设你想要加载`bigscience/bloom-1b7`模型，并且你的GPU RAM刚好足以容纳整个模型，除了`lm_head`。因此，按如下所示编写自定义的`device_map`：
```python
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}
```

然后按如下所示加载模型：
```python
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

就是这样！享受你的模型吧！

#### 调整`llm_int8_threshold`

你可以调整`llm_int8_threshold`参数来更改异常值的阈值。"异常值"是大于某个阈值的隐藏状态值。 
这对应于`LLM.int8()`论文中描述的用于异常值检测的异常值阈值。任何超过此阈值的隐藏状态值都将被视为异常值，并且对这些值的操作将以fp16进行。这些值通常服从正态分布，即大多数值位于[-3.5, 3.5]范围内，但对于大型模型来说，某些异常系统异常值的分布可能完全不同。这些异常值通常在区间[-60, -6]或[6, 60]中。对于绝对值在~5范围内的值，Int8量化效果良好，但是超过该范围后，性能损失显著。一个很好的默认阈值是6，但对于不稳定的模型（小模型，微调）可能需要较低的阈值。
这个参数可以影响模型的推断速度。我们建议根据你的使用情况调整此参数，找到最合适的参数。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_threshold=10,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### 跳过某些模块的转换

某些模型有多个模块，这些模块不需要转换为8位以确保稳定性。例如，Jukebox模型有几个应该跳过的`lm_head`模块。可以使用`llm_int8_skip_modules`进行调整。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### 在8位加载的模型上进行微调

有了Hugging Face生态系统中的官方适配器支持，你可以微调已以8位加载的模型。
这使你可以在单个Google Colab中微调大型模型，例如`flan-t5-large`或`facebook/opt-6.7b`。请查看[`peft`](https://github.com/huggingface/peft)库以获取更多详细信息。

请注意，在加载用于训练的模型时不需要传递`device_map`。它会自动将你的模型加载到GPU上。如果需要，你还可以将设备映射设置为特定设备（例如`cuda:0`，`0`，`torch.device('cuda:0')`）。请注意，`device_map=auto`应仅用于推断。

### BitsAndBytesConfig

[[autodoc]] BitsAndBytesConfig

## 使用🤗`optimum` 进行量化

请参阅[Optimum文档](https://huggingface.co/docs/optimum/index)以了解`optimum`支持的量化方法，并查看这些方法是否适用于你的用例。