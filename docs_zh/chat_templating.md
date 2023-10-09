<!--
版权所有2023 HuggingFace团队。保留所有权利。

根据Apache许可证，第2版（“许可证”），除非符合License中规定的条件，否则不得使用此文件。
你可以在下面的链接中获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样”的，没有任何形式的担保或条件。
在License下分发。特定语言的文件和限制。

⚠️请注意，此文件是Markdown格式，但包含了我们的文档生成器（类似MDX）的特定语法，你的Markdown查看器可能无法正确渲染该语法。

-->

# 聊天模型的模板

## 简介

LLM的一个日益普遍的应用情景是**聊天**。在聊天的情境中，模型不再以单个文本串的方式继续（标准语言模型的情况），而是继续一个由一个或多个**消息**组成的会话，其中每个消息都包括一个**角色**以及消息文本。

通常，这些角色是 "user"（用户发送的消息）和 "assistant"（模型发送的消息）。某些模型还支持 "system" 角色。系统消息通常在对话开始时发送，并包含关于模型在随后的聊天中应如何行为的指令。

所有的语言模型，包括为聊天进行了微调的模型，都在令牌的线性序列上操作，对于角色，在内在方面并没有特别处理。这意味着角色信息通常是通过在消息之间添加控制令牌来注入的，以指示消息界限和相关角色。

不幸的是，目前（还没有！）没有统一的令牌标准，因此不同的模型使用了截然不同的格式和控制令牌进行训练。这对用户来说可能是一个真正的问题-如果你使用了错误的格式，模型将会迷惑于你的输入，性能将会比预期差很多。这就是**聊天模板**的目的所在。

聊天对话通常表示为一个字典列表，其中每个字典包含 `role` 和 `content` 键，表示单个聊天消息。聊天模板是一个包含Jinja模板的字符串，指定了如何将给定模型的会话格式化为一个可标记化的序列。通过将这些信息与标记器一起存储，我们可以确保模型按照其期望的格式获得输入数据。

让我们通过使用 `BlenderBot` 模型的一个快速示例具体说明。BlenderBot 的默认模板非常简单，主要在对话轮之间添加空白：

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
" Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"
```

注意整个聊天会话被压缩成一个字符串。如果我们使用 `tokenize=True`，这是默认设置，该字符串也将被标记化。不过，为了看到一个更复杂的模板示例，让我们使用 `meta-llama/Llama-2-7b-chat-hf` 模型。请注意，此模型具有有限访问权限，因此如果你想自己运行此代码，你将需要在 [repo](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)上申请访问权限：

```python
>> from transformers import AutoTokenizer
>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>> tokenizer.use_default_system_prompt = False
>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]"
```

这次，标记器已经添加了控制令牌 [INST] 和 [/INST]，以指示用户消息（但不包括助理消息！）

## 聊天模板如何工作？

模型的聊天模板存储在 `tokenizer.chat_template` 属性上。如果没有设置聊天模板，则使用该模型类的默认模板。让我们看一下 `BlenderBot` 的模板：

```python

>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> tokenizer.default_chat_template
"{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
```

这让人有些害怕。让我们添加一些换行和缩进，使其更易读。请注意，我们默认使用 Jinja 的 `trim_blocks` 和 `lstrip_blocks` 标志，删除每个块后的第一个换行以及块之前的任何前导空格。这意味着你可以带有缩进和换行符的模板，仍然可以正确地执行！

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ ' ' }}
    {% endif %}
    {{ message['content'] }}
    {% if not loop.last %}
        {{ '  ' }}
    {% endif %}
{% endfor %}
{{ eos_token }}
```

如果你以前从未见过这些内容，这是一个 [Jinja 模板](https://jinja.palletsprojects.com/en/3.1.x/templates/)。Jinja 是一种模板语言，允许你编写简单的代码以生成文本。在很多方面，代码和语法都类似于 Python。在纯 Python 中，该模板可能看起来像这样：

```python
for idx, message in enumerate(messages):
    if message['role'] == 'user':
        print(' ')
    print(message['content'])
    if not idx == len(messages) - 1:  # 检查会话中的最后消息
        print('  ')
print(eos_token)
```

实际上，该模板执行了三项操作：

1. 对于每条消息，如果消息是用户消息，则在其之前添加一个空格，否则不打印任何内容。
2. 添加消息的内容。
3. 如果当前消息不是最后一条消息，则在消息之后添加两个空格。在最后一条消息之后，打印 EOS 令牌。

这是一个相当简单的模板-它不添加任何控制令牌，并且不支持 "system" 消息，这是一种常见的方法，用于向模型提供关于在随后的对话中应如何行为的指示。但是 Jinja 提供了很大的灵活性，可以实现这些功能！现在我们来看一个可以类似于 LLaMA 格式化输入的 Jinja 模板（请注意，真正的 LLaMA 模板包括对默认系统消息的处理以及一般情况下稍有不同的系统消息处理-请不要在实际代码中使用此模板！）

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
        {{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + message['content'] + ' ' + eos_token }}
    {% endif %}
{% endfor %}
```

希望如果你仔细查看一会儿，就可以看出这个模板在做什么-它根据每条消息的 "role" 添加特定的令牌，用于标识是谁发送的。 用户、助理和系统消息由于其包含的令牌而可以清楚地被模型区分开来。

## 我如何创建聊天模板？

很简单，只需编写 Jinja 模板并设置 `tokenizer.chat_template`。你可能会发现，从另一个模型中获取一个现有模板并为自己的需求进行简单的编辑会更容易！例如，我们可以采用上面的 LLaMA 模板，并在助理消息中添加 "[ASST]" 和 "[/ ASST]"：

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
        {{ '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {% endif %}
{% endfor %}
```

现在，只需设置 `tokenizer.chat_template` 属性即可。下一次你使用 [`~PreTrainedTokenizer.apply_chat_template`]，它将使用你的新模板！此属性将保存在 `tokenizer_config.json` 文件中，因此你可以使用 [`~utils.PushToHubMixin.push_to_hub`] 将你的新模板上传到 Hub，并确保每个人都使用正确的模板！

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # 修改系统令牌
tokenizer.chat_template = template  # 设置新模板
tokenizer.push_to_hub("model_name")  # 将新模板上传到 Hub！
```

使用你的新模板后，[`PreTrainedTokenizer.apply_chat_template`] 方法将会使用它！这意味着在使用 [`ConversationalPipeline`] 等地方，你的模型将自动与之兼容！

通过确保模型具有此属性，我们可以确保整个社区能够充分利用开源模型的强大功能。格式不匹配已经困扰这个领域并默默地损害性能太长时间了-是时候结束它们了！

## "默认"模板是什么意思？

在引入聊天模板之前，聊天处理是在模型类级别上硬编码的。为了向后兼容，我们将此类特定处理保留为默认模板，同样是在类级别设置的。如果模型没有设置聊天模板，但其模型类有一个默认模板，则 `ConversationalPipeline` 类和 `apply_chat_template` 等方法将使用类模板。你可以通过检查 `tokenizer.default_chat_template` 属性了解你的标记器的默认模板是什么。

这只是出于向后兼容的原因，以避免破坏任何现有的工作流程。即使类模板适用于你的模型，我们强烈建议通过显式设置 `chat_template` 属性来重写默认模板，以便清楚地告诉用户你的模型已经正确地配置为聊天，并为未来防止默认模板被更改或弃用。

## 我应该使用哪个模板？

当为已经为聊天进行了训练的模型设置模板时，你应该确保模板与模型在训练过程中看到的消息格式完全匹配，否则你可能会遇到性能下降。即使你正在对模型进行进一步的训练，如果保持聊天令牌不变，那么你可能会获得最佳性能。这在许多方面与分词类似-当你在推理或微调时，通常最好能够精确匹配训练时的分词。

另一方面，如果你正在从头开始训练模型，或者对基础语言模型进行聊天微调，你就有很大的自由选择合适的模板！LLMs足够聪明，可以学会处理很多不同的输入格式。我们为那些没有类特定模板的模型提供的默认模板遵循了[ChatML 格式](https://github.com/openai/openai-python/blob/main/chatml.md)，对于许多用例来说，这是一个很好的、灵活的选择。大致如下所示：

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '