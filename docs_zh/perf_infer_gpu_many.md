<!--版权所有 2022 The HuggingFace Team。保留所有权利。

根据Apache许可证2.0版（“许可证”）许可；除非符合许可证，否则不得使用此文件。您可以获取许可证的副本，网址为

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，否则根据许可证分发的软件是按照
“按原样” BASIS提供的，没有任何明示或默示的保证或条件。请看许可证的要求

⚠️ 请注意，本文件在Markdown中，但包含我们的文档构建器的特定语法（类似于MDX），可能无法
在您的Markdown查看器中正确呈现。

-->

# Efficient Inference on a Multiple GPUs

本文档包含有关如何在多个GPU上进行高效推理的信息。
<Tip>

注意：多GPU设置可以使用在[single GPU section](perf_infer_gpu_one.md)中描述的大部分策略。不过，您必须了解一些简单的技术，以便更好地使用。

</Tip>

## BetterTransformer

[BetterTransformer](https://huggingface.co/docs/optimum/bettertransformer/overview) 将🤗 Transformers模型转换为使用基于PyTorch的快速执行路径（底层调用优化的内核，如Flash Attention）。

BetterTransformer还支持更快的文本、图像和音频模型的单个GPU和多个GPU推理。

<Tip>

Flash Attention只能用于使用fp16或bf16 dtype的模型。在使用BetterTransformer之前，请确保将模型转换为适当的dtype。
  
</Tip>

### 解码器模型

对于文本模型，特别是解码器模型（GPT、T5、Llama等），BetterTransformer API将所有注意力操作转换为使用[`torch.nn.functional.scaled_dot_product_attention`运算符](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) (SDPA) ，该运算符仅在PyTorch 2.0及以上版本中可用。

要将模型转换为BetterTransformer：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
# 转换模型为BetterTransformer
model.to_bettertransformer()

# 用于训练或推理
```

SDPA还可以在底层调用[Flash Attention](https://arxiv.org/abs/2205.14135)内核。要启用Flash Attention或检查在给定设置（硬件、问题大小）中是否可用，可以使用`torch.backends.cuda.sdp_kernel`作为上下文管理器：

```diff
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to("cuda")
# 转换模型为BetterTransformer
model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

+ with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

如果出现带有跟踪信息（traceback）的错误提示如下：

```bash
RuntimeError: No available kernel.  Aborting execution.
```

尝试使用PyTorch的夜版（nightly version），该版本可能对Flash Attention有更广泛的覆盖范围：

```bash
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

阅读此[博文](https://pytorch.org/blog/out-of-the-box-acceleration/)以了解有关通过BetterTransformer + SDPA API可以实现的更多内容。

### 编码器模型

对于编码器模型的推理，BetterTransformer将编码器层的前向调用分派给等效的[`torch.nn.TransformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)，该层将执行编码器层的快速路径实现。

由于`torch.nn.TransformerEncoderLayer`的快速路径不支持训练，因此会将其调度到`torch.nn.functional.scaled_dot_product_attention`，后者不利用嵌套张量，但可以使用Flash Attention或Memory-Efficient Attention融合内核。

有关BetterTransformer性能的更多详细信息，请参见此[博文](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2)，您可以在此[博文](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/)中了解有关编码器模型的BetterTransformer信息。


## 高级用法：混合FP4（或Int8）和BetterTransformer

您可以结合上述不同的方法来获得模型的最佳性能。例如，可以将FP4混合精度推理+flash attention与BetterTransformer一起使用：

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