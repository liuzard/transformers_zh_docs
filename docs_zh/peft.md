<!--版权所有2023 The HuggingFace团队。版权所有。
根据Apache许可证，第2.0版（“许可证”）获得许可；除非你遵守许可证，否则你不得使用此文件。
你可以在以下位置获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则按原样分发软件是基于“按原样”方式分发的，
不附带任何明示或暗示的担保或条件。详细了解许可证中的限制和条件。
⚠️ 请注意，此文件以Markdown格式编写，但包含我们 doc-builder 的特殊语法（类似于 MDX），这可能在你的 Markdown 视图器中无法正确呈现。-->

# 使用🤗PEFT加载adapters

[[open-in-colab]]

[参数高效微调（PEFT）](https://huggingface.co/blog/peft) 方法在微调期间冻结预训练模型的参数，并在其之上添加少量可训练参数（adapters）。adapters用于学习特定于任务的信息。这种方法已经被证明在使用更低的计算资源时可以非常节省内存，同时产生与完全微调模型相当的结果。

使用PEFT训练的adapters通常比完整模型小一个数量级，这使得分享、存储和加载它们非常方便。

<div class="flex flex-col justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
  <figcaption class="text-center">存储在Hub上的OPTForCausalLM模型的adapters权重仅为~6MB，而模型权重的完整大小可以达到~700MB。</figcaption>
</div>

如果你想了解有关🤗PEFT库的更多信息，请查看[文档](https://huggingface.co/docs/peft/index)。

## 设置

首先，通过安装🤗PEFT来开始：

```bash
pip install peft
```

如果你想尝试全新的功能，可以考虑从源代码安装库：

```bash
pip install git+https://github.com/huggingface/peft.git
```

## 支持的PEFT模型

🤗Transformers原生支持一些PEFT方法，这意味着你可以加载本地或Hub上存储的adapters权重，并使用少量代码运行或训练它们。支持以下方法：

- [低秩adapters](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [IA3](https://huggingface.co/docs/peft/conceptual_guides/ia3)
- [AdaLoRA](https://arxiv.org/abs/2303.10512)

如果你想使用其他PEFT方法（如提示学习或提示调整）或了解有关🤗PEFT库的一般信息，请参阅文档。

## 加载PEFTadapters

要从🤗transformers加载和使用PEFTadapters模型，请确保Hub仓库或本地目录包含`adapter_config.json`文件和adapters权重，如上图所示。然后，你可以使用`AutoModelFor`类加载PEFTadapters模型。例如，要为因果语言模型加载PEFTadapters模型：

1. 指定PEFT模型ID
2. 将其传递给[`AutoModelForCausalLM`]类

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
```


>你可以使用`AutoModelFor`类或基本模型类，如 `OPTForCausalLM` 或 `LlamaForCausalLM`来加载PEFTadapters。


你还可以通过调用`load_adapter`方法来加载PEFTadapters：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
peft_model_id = "ybelkada/opt-350m-lora"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
```

## 以8位或4位加载

`bitsandbytes`集成支持8位和4位精度数据类型，对于加载大型模型非常有用，因为它节省了内存（有关详细信息，请参阅`bitsandbytes`集成[指南](./quantization#bitsandbytes-integration)）。将`load_in_8bit`或`load_in_4bit`参数添加到[`~PreTrainedModel.from_pretrained`]中，并将 `device_map="auto"` 设置为有效地将模型分配到你的硬件上：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", load_in_8bit=True)
```

## 添加新的adapters

只要新adapters的类型与当前adapters相同，你就可以使用[`~peft.PeftModel.add_adapter`]将新adapters添加到带有现有adapters的模型中。例如，如果你在模型上已添加了现有的LoRAadapters：

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

model.add_adapter(lora_config, adapter_name="adapter_1")
```

要添加新adapters：

```py
# 使用相同的配置附加新的adapters
model.add_adapter(lora_config, adapter_name="adapter_2")
```

现在，你可以使用[`~peft.PeftModel.set_adapter`]来设置要使用的adapters：

```py
# 使用adapters_1
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))

# 使用adapters_2
model.set_adapter("adapter_2")
output_enabled = model.generate(**inputs)
print(tokenizer.decode(output_enabled[0], skip_special_tokens=True))
```

## 启用和禁用adapters

向模型中添加adapters后，你可以启用或禁用adapters模块。要启用adapters模块：

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
adapter_model_id = "ybelkada/opt-350m-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)
peft_config = PeftConfig.from_pretrained(adapter_model_id)

# 用随机权重初始化
peft_config.init_lora_weights = False

model.add_adapter(peft_config)
model.enable_adapters()
output = model.generate(**inputs)
```

要禁用adapters模块：

```py
model.disable_adapters()
output = model.generate(**inputs)
```

## 训练PEFT adapters

PEFTadapters由[`Trainer`]类支持，因此你可以针对特定用例训练adapters。只需要添加几行代码即可。例如，要训练一个LoRAadapters：



>如果你不熟悉使用[`Trainer`]进行微调模型，请查看[微调预训练模型](training.md) 教程。



1. 使用任务类型和超参数定义adapters配置（有关超参数的更多详细信息，请参阅[`~peft.LoraConfig`]）。

```py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
```

2. 将adapters添加到模型中。

```py
model.add_adapter(peft_config)
```

3. 现在你可以将模型传递给[`Trainer`]！

```py
trainer = Trainer(model=model, ...)
trainer.train()
```

要保存已训练的adapters并重新加载：

```py
model.save_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)
```

<!--
TODO: (@younesbelkada @stevhliu)
-   Link to PEFT docs for further details
-   Trainer  
-   8-bit / 4-bit examples ?
-->