<!--版权 2023 年 HuggingFace 团队。版权所有。

根据 Apache 许可证第 2 版 (the "License")，除非符合许可证的规定，否则你不得使用此文件。
你可以在以下位置获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，依照许可证分发的软件是基于"按原样"的基础上提供的，不附带任何担保或条件。请参阅许可证以了解许可证的具体语言和限制。

⚠️ 请注意，该文件是 Markdown 格式的，但包含了我们的文档生成器的特定语法（类似于 MDX），你的 Markdown 阅读器可能无法正确渲染。-->

# RWKV

## 概述

RWKV 模型是在 [此存储库](https://github.com/BlinkDL/RWKV-LM) 中提出的。

它对传统的 Transformer 注意力机制进行了微调，使其线性化。这样，该模型可以用作循环网络：将时间戳 0 和时间戳 1 的输入一起传入，与在时间戳 0 传入输入，然后在时间戳 1 传入输入并附带时间戳 0 的状态的效果相同（参见下面的示例）。

相比之下，RWKV 模型比常规 Transformer 更高效，并且可以处理任意长度的句子（即使模型在训练时使用的上下文长度是固定的）。

此模型由 [sgugger](https://huggingface.co/sgugger) 贡献。
原始代码可在 [此处](https://github.com/BlinkDL/RWKV-LM) 找到。

使用作为 RNN 的示例：

```py
import torch
from transformers import AutoTokenizer, RwkvConfig, RwkvModel

model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

inputs = tokenizer("This is an example.", return_tensors="pt")
# Feed everything to the model
outputs = model(inputs["input_ids"])
output_whole = outputs.last_hidden_state

outputs = model(inputs["input_ids"][:, :2])
output_one = outputs.last_hidden_state

# Using the state computed on the first inputs, we will get the same output
outputs = model(inputs["input_ids"][:, 2:], state=outputs.state)
output_two = outputs.last_hidden_state

torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5)
```

如果你希望确保模型在检测到 `'\n\n'` 时停止生成，则建议使用以下停止准则：

```python 
from transformers import StoppingCriteria

class RwkvStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [187,187], eos_token_id = 537):
        self.eos_sequence = eos_sequence
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_2_ids = input_ids[:,-2:].tolist()
        return self.eos_sequence in last_2_ids


output = model.generate(inputs["input_ids"], max_new_tokens=64, stopping_criteria = [RwkvStoppingCriteria()])
```

## RwkvConfig

[[autodoc]] RwkvConfig


## RwkvModel

[[autodoc]] RwkvModel
    - forward

## RwkvLMHeadModel

[[autodoc]] RwkvForCausalLM
    - forward

## Rwkv 注意力和循环公式

在传统的自回归 Transformer 中，注意力机制的公式如下：

$$O = \hbox{softmax}(QK^{T} / \sqrt{d}) V$$

其中 \\(Q\\)、\\(K\\) 和 \\(V\\) 是形状为 `seq_len x hidden_size` 的矩阵，分别称为查询(query)、键(key)和值(value)（实际上它们是具有批次维度和注意力头维度的更大的矩阵，但我们只关心最后两个维度，即进行矩阵乘法的那两个维度，所以为了简单起见，我们只考虑这两个维度）。矩阵乘积 \\(QK^{T}\\) 的形状为 `seq_len x seq_len`，我们可以用它与 \\(V\\) 进行矩阵乘法，得到与其他矩阵具有相同形状的输出 \\(O\\)。

用其值代替 softmax，可得：

$$O_{i} = \frac{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}} V_{j}}{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}}}$$

需要注意的是，对于 \\(j > i\\) 对应的 \\(QK^{T}\\) 条目进行了掩码 (求和止于 j)，因为注意力不能看到未来的标记（只能看到过去的标记）。

相比之下，RWKV 注意力的公式如下：

$$O_{i} = \sigma(R_{i}) \frac{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}} V_{j}}{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}}}$$

其中 \\(R\\) 是作者称为 receptance 的新矩阵，\\(K\\) 和 \\(V\\) 仍然是键和值（\\(\sigma\\) 是 sigmoid 函数）。\\(W\\) 是一个代表标记位置的新向量，由以下公式给出：

$$W_{0} = u \hbox{  and  } W_{k} = (k-1)w \hbox{ for } k \geq 1$$

其中 \\(u\\) 和 \\(w\\) 是称为 `time_first` 和 `time_decay` 的可学习参数。分子和分母都可以递归表示。将它们命名为 \\(N_{i}\\) 和 \\(D_{i}\\)，我们有：

$$N_{i} = e^{u + K_{i}} V_{i} + \hat{N}_{i} \hbox{  where  } \hat{N}_{i} = e^{K_{i-1}} V_{i-1} + e^{w + K_{i-2}} V_{i-2} \cdots + e^{(i-2)w + K_{1}} V_{1}$$

因此 \\(\hat{N}_{i}\\)（在代码中称为 `numerator_state`）满足以下条件：

$$\hat{N}_{0} = 0 \hbox{  and  } \hat{N}_{j+1} = e^{K_{j}} V_{j} + e^{w} \hat{N}_{j}$$

以及

$$D_{i} = e^{u + K_{i}} + \hat{D}_{i} \hbox{  where  } \hat{D}_{i} = e^{K_{i-1}} + e^{w + K_{i-2}} \cdots + e^{(i-2)w + K_{1}}$$

因此 \\(\hat{D}_{i}\\)（在代码中称为 `denominator_state`）满足以下条件：

$$\hat{D}_{0} = 0 \hbox{  and  } \hat{D}_{j+1} = e^{K_{j}} + e^{w} \hat{D}_{j}$$

为了数值稳定性的考虑，实际上使用了稍微复杂一些的递归公式，因为我们不想计算大数的指数。通常，softmax 不像原样计算，而是对所有项中的最大项的指数进行除法，分子和分母分别除以最大项的指数：

$$\frac{e^{x_{i}}}{\sum_{j=1}^{n} e^{x_{j}}} = \frac{e^{x_{i} - M}}{\sum_{j=1}^{n} e^{x_{j} - M}}$$

其中 \\(M\\) 是所有 \\(x_{j}\\) 的最大值。因此，除保存分子状态（\\(\hat{N}\\)）和分母状态（\\(\hat{D}\\)）外，我们还要跟踪指数项中遇到的所有项的最大值。因此，实际上使用了以下公式：

$$\tilde{N}_{i} = e^{-M_{i}} \hat{N}_{i} \hbox{  and  } \tilde{D}_{i} = e^{-M_{i}} \hat{D}_{i}$$

分别由以下递归公式定义：

$$\tilde{N}_{0} = 0 \hbox{  and  } \tilde{N}_{j+1} = e^{K_{j} - q} V_{j} + e^{w + M_{j} - q} \tilde{N}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

和

$$\tilde{D}_{0} = 0 \hbox{  and  } \tilde{D}_{j+1} = e^{K_{j} - q} + e^{w + M_{j} - q} \tilde{D}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

然后 \\(M_{j+1} = q\\)。有了这些，我们可以计算出

$$N_{i} = e^{u + K_{i} - q} V_{i} + e^{M_{i}} \tilde{N}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

和

$$D_{i} = e^{u + K_{i} - q} + e^{M_{i}} \tilde{D}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

从而最终得到

$$O_{i} = \sigma(R_{i}) \frac{N_{i}}{D_{i}}$$