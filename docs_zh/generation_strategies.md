<!--版权所有2023年HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”），除非符合许可证的规定，否则不得使用此文件。
您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件在许可证下分发，
分发的基础上是“按原样提供的”而不带任何明示或暗示的保证或条件。
有关许可证的特定语言，请参阅许可证下的文档。

⚠️请注意，此文件是用Markdown格式撰写的，但包含特定于我们的doc-builder（类似于MDX的语法），可能无法在您的Markdown查看器中正确渲染。

-->

# 文本生成策略

文本生成在许多自然语言处理（NLP）任务中至关重要，如开放式文本生成、摘要、翻译等。
它也在许多混合模态应用程序中扮演着一种角色，其中文本作为输出的一部分，例如语音识别和图像识别。
一些可以生成文本的模型包括GPT2，XLNet，OpenAI GPT，CTRL，TransformerXL，XLM，Bart，T5，GIT和Whisper。

请查看使用[`~transformers.generation_utils.GenerationMixin.generate`]方法为不同任务生成文本输出的几个示例：

* [文本摘要](tasks/summarization#inference)
* [图像字幕](model_doc/git#transformers.GitForCausalLM.forward.example)
* [音频转录](model_doc/whisper#transformers.WhisperForConditionalGeneration.forward.example)

请注意，生成方法的输入取决于模型的模态性。它们由模型的预处理器类（例如AutoTokenizer或AutoProcessor）返回。如果模型的预处理器创建了多种类型的输入，请将所有输入传递给generate（）。您可以在相应模型的文档中了解有关单个模型的预处理器的更多信息。

选择生成文本的输出令牌的过程称为解码，并且您可以自定义generate（）方法将使用的解码策略。
修改解码策略不会更改任何可以训练的参数的值。但是，它可能会对生成的输出的质量产生明显的影响。它可以帮助减少文本中的重复，并使其更连贯。

该指南描述了：
* 默认的生成配置
* 常见的解码策略及其主要参数
* 将自定义的生成配置与您在🤗 Hub上的微调模型共享和保存

## 默认文本生成配置

模型的解码策略在其生成配置中定义。在使用针对推理的预训练模型时，模型会调用`PreTrainedModel.generate()`方法，在内部应用默认的生成配置。当没有使用自定义配置保存模型时，也会使用默认配置。

当您明确加载模型时，您可以通过`model.generation_config`查看附带的生成配置：

```python
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> model.generation_config
GenerationConfig {
    "bos_token_id": 50256,
    "eos_token_id": 50256,
}
```

打印`model.generation_config`只会显示与默认生成配置不同的值，并且不列出任何默认值。

默认的生成配置限制了将输出与输入提示合并的大小，最多为20个令牌，以避免因资源限制而出错。
默认的解码策略是贪婪搜索（greedy search），是一种最简单的解码策略，它选择一个具有最高概率的令牌作为下一个令牌。对于许多任务和较小的输出大小，这很有效。然而，当用于生成较长的输出时，贪婪搜索可能会开始产生高度重复的结果。

## 自定义文本生成

您可以通过直接将参数及其值传递给[`generate`]方法来覆盖任何`generation_config`:

```python
>>> my_model.generate(**inputs, num_beams=4, do_sample=True)  # doctest: +SKIP
```

即使默认的解码策略大多适用于您的任务，您仍然可以微调一些内容。一些常见的调节参数包括：

- `max_new_tokens`：要生成的最大令牌数。换句话说，它是输出序列的大小，不包括提示中的令牌。
- `num_beams`：通过指定大于1的束束数，您实际上是从贪婪搜索切换到束搜索（beam search）。此策略在每个时间步骤评估多个假设，最终选择整个序列的总体概率最高的假设。这样可以识别以较低概率初始令牌开头的高概率序列，这些序列可能被贪婪搜索忽略。
- `do_sample`：如果设置为 `True`，此参数将启用解码策略，例如多项式采样、束搜索多项式采样、分布式取样和有约束的离散取样。所有这些策略都使用整个词汇表上的概率分布选择下一个令牌，并具有不同的策略特定调整。
- `num_return_sequences`：要为每个输入返回的序列候选数。此选项仅适用于支持多个序列候选的解码策略，例如束搜索和取样的变体。贪婪搜索和对比搜索等解码策略返回单个输出序列。

## 使用模型保存自定义解码策略

如果您想与特定生成配置共享您微调的模型，可以：
* 创建一个[`GenerationConfig`]类实例
* 指定解码策略参数
* 使用[`GenerationConfig.save_pretrained`]将生成配置保存，确保将其`config_file_name`参数保留为空
* 将`push_to_hub`设置为`True`，将您的配置上传到模型的存储库

```python
>>> from transformers import AutoModelForCausalLM, GenerationConfig

>>> model = AutoModelForCausalLM.from_pretrained("my_account/my_model")  # doctest: +SKIP
>>> generation_config = GenerationConfig(
...     max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
... )
>>> generation_config.save_pretrained("my_account/my_model", push_to_hub=True)  # doctest: +SKIP
```

您还可以在单个目录中存储多个生成配置，利用[`GenerationConfig.save_pretrained`]中的`config_file_name`参数。您可以稍后使用[`GenerationConfig.from_pretrained`]实例化它们。如果要为单个模型存储多个生成配置（例如，一个用于采样的创造性文本生成，一个用于束搜索的摘要），则必须具备正确的Hub权限才能将配置文件添加到模型中。

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

>>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

>>> translation_generation_config = GenerationConfig(
...     num_beams=4,
...     early_stopping=True,
...     decoder_start_token_id=0,
...     eos_token_id=model.config.eos_token_id,
...     pad_token=model.config.pad_token_id,
... )

>>> # 提示：将`push_to_hub=True`添加到推送至Hub
>>> translation_generation_config.save_pretrained("/tmp", "translation_generation_config.json")

>>> # 您可以使用命名的生成配置文件来参数化生成
>>> generation_config = GenerationConfig.from_pretrained("/tmp", "translation_generation_config.json")
>>> inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")
>>> outputs = model.generate(**inputs, generation_config=generation_config)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Les fichiers de configuration sont faciles à utiliser!']
```

## 流式计算

`generate()`支持流式计算，通过其“streamer”输入实现。`streamer`输入与具有以下方法的类的任何实例兼容：“put()”和“end()”。在内部，`put()`用于推送新令牌，`end()`用于标记文本生成的结束。

<Tip warning={true}>

流式处理类的API仍在开发中，可能在将来发生变化。

</Tip>

实际上，您可以为各种目的自己创建流式处理类！我们还为您准备了一些基本的流式处理类。例如，您可以使用[`TextStreamer`]类将`generate()`的输出以每次一个字的方式流式传输到屏幕上：

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

>>> tok = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextStreamer(tok)

>>> # 尽管返回通常的输出，但streamer还将将生成的文本打印到stdout。
>>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
```

## 解码策略

一些`generate()`参数及最终的`generation_config`的组合可以用于启用特定的解码策略。
如果您对此概念不熟悉，我们建议阅读此博文以了解常见解码策略的工作原理：[https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)。

在这里，我们将显示一些控制解码策略的参数，并演示如何使用它们。

### 贪婪搜索

[`generate`]默认使用贪婪搜索解码，因此您不需要传递任何参数来启用它。这意味着参数`num_beams`设置为1，`do_sample=False`。

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "I look forward to"
>>> checkpoint = "distilgpt2"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> outputs = model.generate(**inputs)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['I look forward to seeing you all again!\n\n\n\n\n\n\n\n\n\n\n']
```

### 对比搜索

对比搜索解码策略是在2022年的论文《A Contrastive Framework for Neural Text Generation》中提出的。
它展示了在生成非重复但连贯的长输出方面的超越性能。要了解对比搜索的工作原理，请查看[此博文](https://huggingface.co/blog/introducing-csearch)。
启用和控制对比搜索行为的两个主要参数是`penalty_alpha`和`top_k`：

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> checkpoint = "gpt2-large"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> prompt = "Hugging Face Company is"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=100)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Hugging Face Company is a family owned and operated business. We pride ourselves on being the best\nin the business and our customer service is second to none.\n\nIf you have any questions about our\nproducts or services, feel free to contact us at any time. We look forward to hearing from you!']
```

### 多项式采样

与总是选择具有最高概率的令牌作为下一个令牌的贪婪搜索相反，多项式采样（也称为祖先采样）是根据模型给出的整个词汇表上的概率分布随机选择下一个令牌的方法。每个具有非零概率的令牌都有可能被选择，从而减少重复的风险。

要启用多项式采样，请将`do_sample=True`和`num_beams=1`设置。

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
>>> set_seed(0)  # 为了可重现性

>>> checkpoint = "gpt2-large"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> prompt = "Today was an amazing day because"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> outputs = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Today was an amazing day because when you go to the World Cup and you don\'t, or when you don\'t get invited,\nthat\'s a terrible feeling."']
```

### 束搜索解码

与贪婪搜索不同，束搜索解码在每个时间步骤保留多个假设，并最终选择整个序列的总体概率最高的假设。这样可以识别以较低概率初始令牌开头的高概率序列，这些序列可能被贪婪搜索忽略。

要启用此解码策略，指定大于1的`num_beams`（即要跟踪的假设数）。

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "It is astonishing how one can"
>>> checkpoint = "gpt2-medium"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, max_new_tokens=50)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['It is astonishing how one can have such a profound impact on the lives of so many people\nin such a short period of time."\n\nHe added: "I am very proud of the work I have been able to do in the last few years.\n\n"I have']
```

### 束搜索多项式采样

正如其名称所示，此解码策略将束搜索与多项式采样结合起来。您需要指定大于1的`num_beams`，并将`do_sample=True`以使用此解码策略。

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
>>> set_seed(0)  # 为了可重现性

>>> prompt = "translate English to German: The house is wonderful."
>>> checkpoint = "t5-small"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, do_sample=True)
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Das Haus ist wunderbar.'
```

### 多样化的束搜索解码

不同于贪婪搜索，束搜索解码在每个时间步骤保留多个假设，并最终选择整个序列的总体概率最高的假设。这对于识别以较低概率初始令牌开头的高概率序列很有优势，这些序列可能会被贪婪搜索忽略。

要启用此解码策略，请指定大于1的`num_beams`。

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "It is astonishing how one can"
>>> checkpoint = "gpt2-medium"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, max_new_tokens=50)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['It is astonishing how one can have such a profound impact on the lives of so many people in such a short period of time."\n\nHe added: "I am very proud of the work I have been able to do in the last few years.\n\n"I have']
```


这个多样化的beam search解码策略是beam search策略的扩展，允许生成一个更多样化的beam序列集合供选择。了解它是如何工作的，请参考[Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/pdf/1610.02424.pdf)。
这个方法有三个主要参数：`num_beams`，`num_beam_groups`和`diversity_penalty`。
多样性惩罚确保输出在组之间是不同的，并且在每个组内使用beam search。


```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> checkpoint = "google/pegasus-xsum"
>>> prompt = (
...     "The Permaculture Design Principles are a set of universal design principles "
...     "that can be applied to any location, climate and culture, and they allow us to design "
...     "the most efficient and sustainable human habitation and food production systems. "
...     "Permaculture is a design system that encompasses a wide variety of disciplines, such "
...     "as ecology, landscape design, environmental science and energy conservation, and the "
...     "Permaculture design principles are drawn from these various disciplines. Each individual "
...     "design principle itself embodies a complete conceptual framework based on sound "
...     "scientific principles. When we bring all these separate  principles together, we can "
...     "create a design system that both looks at whole systems, the parts that these systems "
...     "consist of, and how those parts interact with each other to create a complex, dynamic, "
...     "living system. Each design principle serves as a tool that allows us to integrate all "
...     "the separate parts of a design, referred to as elements, into a functional, synergistic, "
...     "whole system, where the elements harmoniously interact and work together in the most "
...     "efficient way possible."
... )

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, num_beam_groups=5, max_new_tokens=30, diversity_penalty=1.0)
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'The Design Principles are a set of universal design principles that can be applied to any location, climate and
culture, and they allow us to design the'
```

本指南说明了各种解码策略可用的主要参数。[`generate`]方法还存在更高级的参数，可进一步控制[`generate`]方法的行为。
有关可用参数的完整列表，请参阅[API文档](main_classes/text_generation.md)。

### 帮助解码

帮助解码是对上述解码策略的修改，它使用与tokenizer相同的助手模型（理想情况下是一个更小的模型）贪婪地生成几个候选词元。然后，主模型在单个前向传递中验证候选词元，从而加速解码过程。当前仅支持辅助解码的贪婪搜索和采样，并且不支持批处理输入。要了解有关辅助解码的更多信息，请查看[这篇博文](https://huggingface.co/blog/assisted-generation)。

要启用辅助解码，请使用模型设置`assistant_model`参数。

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "Alice and Bob"
>>> checkpoint = "EleutherAI/pythia-1.4b-deduped"
>>> assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
>>> outputs = model.generate(**inputs, assistant_model=assistant_model)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']
```

在使用辅助解码和采样方法时，您可以使用`temperature`参数来控制随机性，就像使用多项式采样一样。然而，在辅助解码中，降低温度将有助于提高延迟。

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
>>> set_seed(42)  # 为了可重现性

>>> prompt = "Alice and Bob"
>>> checkpoint = "EleutherAI/pythia-1.4b-deduped"
>>> assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
>>> outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob are going to the same party. It is a small party, in a small']
```
