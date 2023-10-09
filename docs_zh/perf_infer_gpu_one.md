<!--版权所有2022年The HuggingFace团队。 保留所有权利。

根据Apache许可证第2.0版（“许可证”）授权；您除非遵守此许可证，否则不得使用此文件。
您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样” BASIS 没有任何明示或暗示的担保或条件。请参阅有关许可证的详细信息

⚠️请注意，此文件是Markdown格式的，但包含针对我们的文档生成器的特定语法（类似于MDX）。因此在您的Markdown查看器中可能无法正常呈现。 -->

# 在单个GPU上进行高效推理

除了本指南外，还可以在 [使用单个GPU进行训练指南](perf_train_gpu_one.md) 和 [在CPU上进行推理指南](perf_infer_cpu.md) 中找到相关信息。

## BetterTransformer

[BetterTransformer](https://huggingface.co/docs/optimum/bettertransformer/overview) 将 🤗 Transformers 模型转换为使用 PyTorch 本地的快速路径执行，该执行调用了优化的核函数，如 Flash Attention。  

BetterTransformer 还支持文本、图像和音频模型的在单个和多个GPU上进行更快的推理。

<Tip>

Flash Attention 仅适用于使用 fp16 或 bf16 数据类型的模型。在使用 BetterTransformer 之前，请确保将模型转换为适当的 dtype。
  
</Tip>

### 编码器模型

使用 PyTorch 本地的 [`nn.MultiHeadAttention`](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/) 注意力快速路径，名为 BetterTransformer，可以通过 [🤗 Optimum 库](https://huggingface.co/docs/optimum/bettertransformer/overview) 中的集成与 Transformers 结合使用。

PyTorch 的注意力快速路径通过内核融合和使用 [嵌套张量](https://pytorch.org/docs/stable/nested.html) 来加速推理。详细的基准测试可以在 [此博文](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2) 中找到。

在安装 [`optimum`](https://github.com/huggingface/optimum) 包之后，在推理过程中使用 Better Transformer，可以通过调用[`~PreTrainedModel.to_bettertransformer`]来替换相关的内部模块：

```python
model = model.to_bettertransformer()
```

方法[`~PreTrainedModel.reverse_bettertransformer`]可以使模型恢复到原始的建模方式，应该在保存模型之前使用，以便使用规范的 transformers 建模方式：

```python
model = model.reverse_bettertransformer()
model.save_pretrained("saved_model")
```

请查看此 [博文](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2) 以了解有关使用 `BetterTransformer` API 针对编码器模型可以做些什么的更多信息。

### 解码器模型

对于文本模型，特别是基于解码器的模型（如 GPT、T5、Llama 等），BetterTransformer API 将所有注意力操作转换为使用 [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention)（SDPA）运算符的操作（此运算符仅在 PyTorch 2.0 及更高版本中可用）。

要将模型转换为 BetterTransformer：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
# 将模型转换为 BetterTransformer
model.to_bettertransformer()

# 用于训练或推理
```

SDPA 也可以在内部调用 [Flash Attention](https://arxiv.org/abs/2205.14135) 的核函数。要启用 Flash Attention 或检查它在给定环境（硬件、问题规模）中是否可用，请使用 [`torch.backends.cuda.sdp_kernel`](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel) 作为上下文管理器：

```diff
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16).to("cuda")
# 将模型转换为 BetterTransformer
model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

+ with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

如果你看到一个带有回溯的错误，该错误提示为：

```bash
RuntimeError: No available kernel.  Aborting execution.
```

请尝试使用 PyTorch 每夜版，该版本的 Flash Attention 可能具有更广的覆盖范围：

```bash
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

或确保您的模型正确转换为 float16 或 bfloat16 字符串


请查看这个 [详细博文](https://pytorch.org/blog/out-of-the-box-acceleration/) 以了解如何使用 `BetterTransformer` + SDPA API 来获取更多功能。

## 使用 FP4 half-precision 混合精度进行推理的 `bitsandbytes` 集成

您可以安装 `bitsandbytes` 并从中受益，以便在 GPU 上轻松压缩模型。 使用 FP4 量化，与原始全精度版本相比，可以将模型大小减小多达 8 倍。 请查看下面如何开始。

<Tip>

请注意，此功能也可以在多个 GPU 设置中使用。

</Tip>

### 要求 [[要求-用于-fp4-half-precision-推理]]

- 最新的 `bitsandbytes` 库
`pip install bitsandbytes>=0.39.0`

- 从源代码安装最新的 `accelerate`
`pip install git+https://github.com/huggingface/accelerate.git`

- 从源代码安装最新的 `transformers`
`pip install git+https://github.com/huggingface/transformers.git`

### 运行 FP4 模型 - 单 GPU 设置 - 快速入门

您可以通过运行以下代码快速在单个 GPU 上运行 FP4 模型：

```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
```
请注意，`device_map` 是可选的，但在推理时设置 `device_map = 'auto'` 是推荐的，因为它将模型有效地分派到可用资源上。

### 运行 FP4 模型 - 多 GPU 设置

将混合 4 位模型加载到多个 GPU 中的方法与单个 GPU 设置相同（与单 GPU 设置相同的命令）：
```py
model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
```
但是你可以使用 `accelerate` 来控制每个 GPU 上要分配的 GPU 内存。使用 `max_memory` 参数，如下所示：

```py
max_memory_mapping = {0: "600MB", 1: "1GB"}
model_name = "bigscience/bloom-3b"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_4bit=True, max_memory=max_memory_mapping
)
```
在此示例中，第一个 GPU 将使用 600MB 的内存，第二个 GPU 将使用 1GB。

### 高级用法

有关此方法的更高级用法，请参阅 [量化](main_classes/quantization) 文档页面。

## 使用 Int8 混合精度矩阵分解的 `bitsandbytes` 集成

<Tip>

请注意，此功能也可以在多个 GPU 设置中使用。

</Tip>

从论文 [`LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale`](https://arxiv.org/abs/2208.07339)，我们支持在几行代码中进行 Hugging Face 集成。该方法通过 8 位张量核心（fp16 权重为 2 倍，fp32 权重为 4 倍）操作半精度的耍杂技来提供结果性能的方法。

![HFxbitsandbytes.png](https://cdn-uploads.huggingface.co/production/uploads/1659861207959-62441d1d9fdefb55a0b7d12c.png)

Int8 混合精度矩阵分解通过将矩阵乘法分解为两个流进行操作：（1）在 fp16 中进行的系统化特征异常值流矩阵乘法（0.01%），（2）int8 矩阵乘法操作的常规流（99.9%）。借助该方法，可以在不损失预测性能的情况下进行适用于非常大模型的 int8 推理。有关该方法的更多详细信息，请查看[论文](https://arxiv.org/abs/2208.07339) 或我们的[关于集成的博文](https://huggingface.co/blog/hf-bitsandbytes-integration)。

![MixedInt8.gif](https://cdn-uploads.huggingface.co/production/uploads/1660567469965-62441d1d9fdefb55a0b7d12c.gif)

请注意，您需要 GPU 才能运行混合 8 位模型，因为核已编译为仅适用于 GPU。确保您有足够的 GPU 内存来存储模型的四分之一（如果您的模型权重为半精度，则是二分之一）之前，使用此功能。

以下是一些提示，以帮助您使用此模块，或者按照 [Google colab 的演示](#colab-demos) 进行演示。

### 要求 [[requirements-for-int8-mixedprecision-matrix-decomposition]]

- 如果您的 `bitsandbytes<0.37.0`，请确保您在支持 8 位张量核心的 NVIDIA GPU 上运行（图灵、安培或更新架构 - 例如 T4、RTX20s、RTX30s、A40-A100）。对于 `bitsandbytes>=0.37.0`，应支持所有 GPU。
- 通过运行以下命令来安装正确的 `bitsandbytes` 版本：
`pip install bitsandbytes>=0.31.5`
- 安装 `accelerate`
`pip install accelerate>=0.12.0`

### 运行混合 Int8 模型 - 单 GPU 设置

在安装所需库之后，加载混合 8 位模型的方法如下所示：

```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
```

对于文本生成，我们建议：

* 使用模型的 `generate()` 方法而不是 `pipeline()` 函数。尽管使用 `pipeline()` 函数可以进行推理，但它不针对混合 8 位模型进行优化，因此与使用 `generate()` 方法相比，速度较慢。此外，一些采样策略，如具有核心序列的采样，在混合 8 位模型的 `pipeline()` 函数中不受支持。
* 将所有输入都放在与模型相同的设备上。

以下是一个简单的示例：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

### 运行混合 Int8 模型 - 多 GPU 设置

在多个 GPU 中加载混合 8 位模型的方法与单个 GPU 设置相同（与单 GPU 设置相同的命令）：
```py
model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
```

但是，您可以使用 `accelerate` 来控制要在每个 GPU 上分配的 GPU 内存。使用 `max_memory` 参数，如下所示：

```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping
)
```
在此示例中，第一个 GPU 将使用 1GB 内存，第二个 GPU 将使用 2GB。

### Google Colab 演示

使用此方法，您可以在以前无法在 Google Colab 上运行的模型上进行推理。
查看在 Google Colab 上运行 T5-11b（42GB 的 fp32）的演示！使用 8 位量化：

[![在 Colab 中打开：T5-11b 演示](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing)

或者查看 BLOOM-3B 的演示：

[![在 Colab 中打开：BLOOM-3b 演示](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing)

## 高级用法：FP4（或Int8）和BetterTransformer 混合

您可以组合上述不同的方法，以获得最佳的模型性能。例如，您可以在 FP4 混合精度推理 + flash attention 中使用 BetterTransformer：

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=quantization_config)

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
