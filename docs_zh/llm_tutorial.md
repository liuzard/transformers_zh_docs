<!-- 版权 2023 年 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 第 2.0 版（“许可证”）获得许可；你不得使用此文件，除非符合许可证的要求。
你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证发布的软件是基于"按原样"的方式分发的，不附带任何形式的担保或条件。请查阅许可证以获取许可证下的特定语言权限和限制。

⚠️ 请注意，该文件采用 Markdown 格式，但包含我们的文档生成器的特定语法（类似于 MDX），可能在你的 Markdown 查看器中不能正确呈现。

-->

# 使用 LLMs 生成文本

[[open-in-colab]]

LLMs，即大型语言模型，是文本生成的关键组件。简而言之，它们由大型预训练的 Transformer 模型组成，用于在给定一些输入文本的情况下预测下一个单词（或更准确地说，token）。由于它们一次预测一个token，因此如果要生成新的句子，你需要进行更复杂的操作，而不仅仅是调用模型 - 你需要进行自回归生成。

自回归生成是在推断时使用模型的生成输出迭代调用模型的过程，给定一些初始输入。在 🤗Transformers 中，这由 [`~generation.GenerationMixin.generate`] 方法处理，该方法适用于具有生成能力的所有模型。

本教程将向你展示如何：

- 使用 LLM 生成文本
- 避免常见陷阱
- 下一步，帮助你充分利用 LLM

开始之前，请确保你已安装所有必要的库：

```bash
pip install transformers bitsandbytes>=0.39.0 -q
```

## 生成文本

训练用于[因果语言建模](tasks/language_modeling)的语言模型以文本token序列作为输入，并返回下一个token的概率分布。

<!-- [GIF 1 -- FWD PASS] -->

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov"
    ></video>
    <figcaption>"LLM的前向传播"</figcaption>
</figure>

使用 LLMs 进行自回归生成的关键是如何从该概率分布中选择下一个token。在这一步中，可以采用任何方法，只要能够得到下一次迭代的token。这意味着可以简单地选择概率分布中最有可能的token，也可以在从所得分布中进行采样之前应用数十个转换来实现更复杂的选择过程。

<!-- [GIF 2 -- TEXT GENERATION] -->

<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov"
    ></video>
    <figcaption>"自回归生成是通过迭代地从概率分布中选择下一个token来生成文本。"</figcaption>
</figure>

上述过程会迭代重复进行，直到满足某个停止条件。理想情况下，停止条件由模型决定，模型应该学会何时输出一个结束序列（`EOS`）token。如果不是这种情况，生成将在达到预定义的最大长度时停止。

正确设置token选择步骤和停止条件对于使你的模型在任务中表现符合预期至关重要。这就是为什么我们为每个模型都有一个与之关联的 [`~generation.GenerationConfig`] 文件，其中包含一个良好的默认生成参数设置，并与你的模型一起加载。

让我们来谈谈代码！

<Tip>

如果你对 LLMs 的基本用法感兴趣，我们提供的高级 [`Pipeline`](http://www.liuzard.com/pipeline_tutorial) 接口是一个很好的起点。然而，LLMs 通常需要高级功能，比如量化和对token选择步骤的精细控制，最好通过 [`~generation.GenerationMixin.generate`] 来实现。带有 LLMs 的自回归生成也需要大量资源，并且应该在 GPU 上执行以获得足够的吞吐量。

</Tip>

<!-- TODO: update example to llama 2 (or a newer popular baseline) when it becomes ungated -->

首先，你需要加载模型。

```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

在 `from_pretrained` 调用中，你会注意到两个标志：

- `device_map` 确保将模型移动到你的 GPU 上
- `load_in_4bit` 对模型进行 [4位动态量化](http://www.liuzard.com/main_classes/quantization)，大大减少了资源要求

还有其他初始化模型的方法，但这是开始使用 LLM 的一个不错的基准。

接下来，你需要使用 [tokenizer](http://www.liuzard.com/tokenizer_summary) 对文本输入进行预处理。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

`model_inputs` 变量保存了分词处理的文本输入以及注意力遮罩。虽然 [`~generation.GenerationMixin.generate`] 会尽最大努力在未传递时推断出注意力遮罩，但我们建议尽可能在调用中传递它以获得最佳结果。

最后，调用 [`~generation.GenerationMixin.generate`] 方法返回生成的token，应在打印之前将其转换为文本。

```py
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, black, white, and brown'
```

就是这样！只需几行代码，你就可以利用 LLM 的强大功能。

## 常见陷阱

有许多 [生成策略](http://www.liuzard.com/generation_strategies)，有时默认值可能不适合你的用例。如果生成的输出与你期望的结果不一致，我们列出了最常见的陷阱及其避免方法。

```py
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

### 生成的输出过短/过长

如果在 [`~generation.GenerationConfig`] 文件中未指定，`generate` 默认最多返回 20 个token。我们强烈建议在 `generate` 调用中手动设置 `max_new_tokens`，以控制它可以返回的最大新token数量。请记住，LLMs（更准确地说，[仅解码器模型](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt)）还会将输入提示作为输出的一部分返回。


```py
>>> model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

>>> # By default, the output will contain up to 20 tokens
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'

>>> # Setting `max_new_tokens` allows you to control the maximum length
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=50)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

### 生成模式不正确

默认情况下，除非在 [`~generation.GenerationConfig`] 文件中指定，否则 `generate` 会在每次迭代时选择最可能的token（贪婪解码）。根据你的任务，这可能是不希望的；对话型任务或写作文章等创造性任务受益于采样。另一方面，音频转录或翻译等基于输入的任务受益于贪婪解码。通过设置 `do_sample=True` 来启用采样，你可以在这篇 [博文](https://huggingface.co/blog/how-to-generate) 中了解更多关于这个主题的信息。

```py
>>> # Set seed or reproducibility -- you don't need this unless you want full reproducibility
>>> from transformers import set_seed
>>> set_seed(0)

>>> model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")

>>> # LLM + greedy decoding = repetitive, boring output
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat. I am a cat. I am a cat. I am a cat'

>>> # With sampling, the output becomes more creative!
>>> generated_ids = model.generate(**model_inputs, do_sample=True)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat.\nI just need to be. I am always.\nEvery time'
```

### 填充方向不正确

LLMs 是仅解码器架构，这意味着它们会继续迭代你的输入提示。如果你的输入长度不相同，则需要进行填充。由于 LLMs 没有训练过从填充token继续生成，因此你的输入需要进行左填充。同时确保不要忘记将注意力遮罩传递给 generate！

```py
>>> # The tokenizer initialized above has right-padding active by default: the 1st sequence,
>>> # which is shorter, has padding on the right side. Generation fails.
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
''

>>> # With left-padding, it works as expected!
>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b", padding_side="left")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

<!-- TODO: when the prompting guide is ready, mention the importance of setting the right prompt in this section -->

## 进一步资源

尽管自回归生成过程相对简单，但要充分利用你的 LLM 可能是一项具有挑战性的任务，因为其中涉及许多复杂的部分。以下是帮助你进一步了解和使用 LLM 的下一步资源：

<!-- TODO: complete with new guides -->
### 高级 generate 用法

1. [指南](http://www.liuzard.com/generation_strategies) 介绍如何控制不同的生成方法、设置生成配置文件以及流式输出；
2. [`~generation.GenerationConfig`]、[`~generation.GenerationMixin.generate`] 和与生成相关的类的[API参考](http://www.liuzard.com/internal/generation_utils)。

### LLM 排行榜

1. [Open LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)，专注于开源模型的质量；
2. [Open LLM-Perf 排行榜](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)，专注于 LLM 的吞吐量。

### 延迟和吞吐量

1. [指南](http://www.liuzard.com/main_classes/quantization) 介绍动态量化，向你展示如何大幅减少内存需求。

### 相关库

1. [`text-generation-inference`](https://github.com/huggingface/text-generation-inference)，一个面向生产环境的 LLM 服务器；
2. [`optimum`](https://github.com/huggingface/optimum)，🤗Transformers 的扩展，针对特定硬件设备进行优化。

这些资源将帮助你更深入地了解和使用 LLM，并在各种自然语言处理任务中发挥其优势。祝你进一步取得成功！