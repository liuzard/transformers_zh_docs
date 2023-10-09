<!--版权所有2023 HuggingFace团队。版权所有。
根据Apache许可证2.0版（“许可证”）获得许可；你除非符合许可证的规定，否则不得使用此文件。
你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何担保或条件，无论是明示的还是暗示的。有关的特定语言许可证和限制的信息，请参阅许可证。

⚠️请注意，此文件以Markdown格式编写，但包含我们文档构建工具的特定语法（类似于MDX），这可能在你的Markdown查看器中无法正确渲染。

-->

# Transformers Agents

<Tip warning={true}>

Transformers Agents是一个实验性API，随时可能发生变化。由于API或底层模型容易改变，代理返回的结果可能会有所不同。

</Tip>

Transformers版本v4.29.0，基于*工具*和*代理*的概念进行构建。你可以在
[此Colab](https://colab.research.google.com/drive/1c7MHD-T1forUPGcC_jlwsIptOzpG3hSj）中进行测试。)

简而言之，它在transformers之上提供了一个自然语言API：我们定义了一套精心策划的工具，并设计了一个代理来解释自然语言并使用这些工具。它是可扩展的；我们策划了一些相关的工具，但我们将向你展示系统如何轻松扩展以使用由社区开发的任何工具。

我们先从几个使用这个新API可以实现的示例开始。当涉及多模态任务时，它特别强大，因此让我们试一下生成图像和朗读文本。

```py
agent.run("描述下面的图片", image=image)
```

| **输入**                                                                                                                     | **输出**                                   |
|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png" width=200> | 一只海狸正在水中游泳 |

---

```py
agent.run("大声朗读下面的文本", text=text)
```
| **输入**                                                                                                               | **输出**                                     |
|---------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| 一只海狸正在水中游泳 | <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tts_example.wav" type="audio/wav"> 你的浏览器不支持音频元素。 </audio>

---

```py
agent.run(
    "在以下`文档`中，TRRF科学顾问会议将在哪里举行？",
    document=document,
)
```
| **输入**                                                                                                                      | **输出**         |
|--------------------------------------------------------------------------------------------------------------------------------|----------------|
| <img src="https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/0/image/image.jpg" width=200> | 宴会接待区     |

## 快速入门

在能够使用`agent.run`之前，你需要实例化一个代理，这是一个大型语言模型（LLM）。
我们支持OpenAI模型以及来自BigCode和OpenAssistant的开源替代方案。OpenAI
模型表现较好（但需要你拥有OpenAI API密钥，因此不能免费使用）；Hugging Face提供了访问BigCode和OpenAssistant模型的免费端点。

首先，请安装`agents` extras以安装所有默认依赖项。
```bash
pip install transformers[agents]
```

要使用OpenAI模型，在安装完`openai`依赖项后，实例化`OpenAiAgent`：

```bash
pip install openai
```


```py
from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003", api_key="<your_api_key>")
```

要使用BigCode或OpenAssistant，请登录以访问推理API：

```py
from huggingface_hub import login

login("<YOUR_TOKEN>")
```

然后，实例化代理

```py
from transformers import HfAgent

# Starcoder
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
# StarcoderBase
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
# OpenAssistant
# agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
```

这是使用Hugging Face目前免费提供的推理API。如果你有自己的推理
端点（或其他端点），你可以用你的URL端点替换上面的URL。

<Tip>

StarCoder和OpenAssistant可免费使用，并且在处理简单任务时表现出色。但是，对于处理更复杂提示时，检查点
不合适。如果你遇到此问题，建议尝试OpenAI模型，在目前的时间内表现更好。

</Tip>

现在你已经准备就绪！让我们深入了解现在可以使用的两个API。

### 单次执行（run）

单次执行方法是使用代理的[`~Agent.run`]方法：

```py
agent.run("给出下面的图片加上标题")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200>

它会自动选择适用于你要执行的任务的工具（或工具），并适当地运行它们。它
可以在同一指令中执行一个或多个任务（尽管你的指令越复杂，代理失败的可能性越大）。

```py
agent.run("给我一张海的图片，然后转换图片以添加一个岛屿")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sea_and_island.png" width=200>

<br/>


每个[`~Agent.run`]操作都是独立的，因此你可以连续运行多次并执行不同的任务。

请注意，你的`agent`只是一个大型语言模型，因此提示中的微小变化可能会产生完全不同的结果。所以
清楚地解释你想要执行的任务非常重要。我们在此处深入讨论如何编写好的提示 [here](custom_tools.md#writing-good-user-inputs)。

如果你想在执行之间保持状态或将非文本对象传递给代理，可以通过指定
你希望代理使用的变量来实现。例如，你可以生成河流和湖泊的第一张图片，然后要求模型更新该图片以添加一个岛屿，如下所示：

```python
picture = agent.run("生成河流和湖泊的图片。")
updated_picture = agent.run("将图片在`picture`中进行转换以添加一个岛屿。", picture=picture)
```

<Tip>

当模型无法理解你的请求并混合使用工具时，这可能很有帮助。例如：

```py
agent.run("给我一张河狸在海里游泳的图片")
```

在这种情况下，模型可以有两种解释方式：
- 让`text-to-image`生成一张河狸在海里游泳的图片
- 或者，让`text-to-image`生成河狸，然后使用`image-transformation`工具让它在海中游泳

如果你想强制使用第一种情况，可以将提示作为参数传递给它：

```py
agent.run("给我一张`prompt`的图片", prompt="一只河狸在海里游泳")
```

</Tip>


### 基于对话的执行（chat）

代理还具有基于对话的方法，使用[`~Agent.chat`]方法：

```py
agent.chat("生成一张河流和湖泊的图片")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200> 

```py
agent.chat("将图片转换为其中添加一个岩石")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_and_beaver.png" width=200>

<br/>

当你想要保持跨指令的状态时，这是一种有趣的方法。对于实验，它更好，
但是在处理单个指令而不是复杂指令时，它往往会比[`~Agent.run`]方法更好。

如果你希望传递非文本类型或特定提示，则此方法也可以接受参数。

### ⚠️远程执行

出于演示目的，并使其可在所有设置中使用，我们为发布的默认工具创建了远程执行程序。这些是使用
推理端点创建的
我们现在关闭了它们，但是为了查看如何自己设置远程执行程序工具，
我们建议阅读[custom tool guide]（./custom_tools）。

### 这里发生了什么？什么是工具和代理？

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/diagram.png">

#### 代理

这里的"代理"是一个大型语言模型，我们通过提示它，使其可以访问特定的一组工具。

LLMs非常擅长生成一小部分代码，因此此API利用LLM的这一优势，通过提示LLM给出一小段执行特定任务的代码与一组工具。然后，通过
你给代理的任务和工具描述来完成你的任务，并获取工具的预期输入和输出的相关文档，以及生成相关代码。

#### 工具

工具非常简单：它们是一个具有名称和描述的单个函数。然后我们使用这些工具的描述
提示代理。通过这个提示，我们向代理展示了它如何利用工具来执行查询中要求的任务。

这是使用全新的工具而不是工作流程，因为代理使用非常原子的工具编写更好的代码。工作流程更复杂而且通常将多个任务合并为一个。工具旨在
专注于一个非常简单的任务。

#### 代码执行？

然后，我们使用我们的小型Python解释器在参数传递给工具的一组输入上执行此代码。所以
我们听到你在后面尖叫"任意代码执行！"，但是让我们解释为什么不是这种情况。

唯一可以调用的函数是你提供的工具和打印函数，因此已经
你可以执行的操作受到限制。如果限于Hugging Face工具，你应该是安全的。 

然后，我们不允许任何属性查找或导入（这些在传递
输入/输出给一小组函数时通常是不需要的），因此所有最明显的攻击（并且你需要提示LLM
输出它们）不应该是一个问题。如果你想确保非常安全，可以使用附加参数return_code=True执行run()方法，这样代理将只返回代码供你执行，你可以决定是否执行。

如果执行尝试执行非法操作的任何行或者出现正常的Python错误并且代码是由代理生成的，则执行将停止。

### 经过策划的工具集

我们确定了一组工具，可以增强这些代理。以下是我们已整合到`transformers`中的工具的更新列表：

- **文档问答**：给定图像格式的文档（例如PDF），回答有关该文档的问题（[Donut]（./model_doc/donut））
- **文本问答**：给定一段长文本和一个问题，在文本中回答问题（[Flan-T5]（./model_doc/flan-t5））
- **无条件图像标题**：为图像加标题！（[BLIP]（./model_doc/blip））
- **图像问答**：给定一张图像，回答有关该图像的问题（[VILT]（./model_doc/vilt））
- **图像分割**：给定一张图像和一个提示，输出该提示的分割掩码（[CLIPSeg]（./model_doc/clipseg））
- **语音转文本**：给定人们说话的音频录音，将该语音转录为文本（[Whisper]（./model_doc/whisper））
- **文本转语音**：将文本转换为语音（[SpeechT5]（./model_doc/speecht5））
- **零样本文本分类**：给定一段文本和一组标签，识别该文本最符合的标签（[BART]（./model_doc/bart））
- **文本摘要**：用一到几个句子概括一段长文本（[BART]（./model_doc/bart））
- **翻译**：将文本翻译为给定的语言（[NLLB]（./model_doc/nllb））

这些工具已经在transformers中进行了集成，并且也可以手动使用，例如：

```py
from transformers import load_tool

tool = load_tool("text-to-speech")
audio = tool("这是一个文本转语音工具")
```

### 自定义工具

虽然我们确定了一套经过策划的工具，但我们坚信，这个实现提供的主要价值是能够快速创建和共享自定义工具。

通过将工具的代码推送到Hugging Face Space或模型存储库，你可以直接使用代理来利用工具。我们在
[`huggingface-tools`组织](https://huggingface.co/huggingface-tools)中添加了一些
与transformers无关的工具：

- **文本下载器**：从Web URL下载文本
- **文本到图像**：根据提示生成图像，利用稳定的扩散
- **图像转换**：根据初始图像和提示修改图像，利用稳定的instruct pix2pix扩散
- **文本到视频**：根据提示生成一个小视频，利用damo-vilab

我们从一开始就一直使用的text-to-image工具是位于
[*huggingface-tools/text-to-image*](https://huggingface.co/spaces/huggingface-tools/text-to-image)的远程工具！我们将
继续在此组织和其他组织中发布此类工具，以进一步加强此实现。

这些代理默认访问存储在[`huggingface-tools`](https://huggingface.co/huggingface-tools)上的工具。
我们将解释如何编写和共享自己的工具，以及如何利用Hub上驻留的任何自定义工具，[following guide](custom_tools.md)中提供了更多详细信息。

### 代码生成

到目前为止，我们已经展示了如何使用代理程序来执行操作。然而，该代理程序只生成我们使用非常受限制的 Python 解释器执行的代码。如果你希望在不同的环境中使用生成的代码，可以通过提示代理程序来返回代码，同时返回工具定义和准确的导入信息。

例如，以下指令：

```python
agent.run("为我画一幅河流和湖泊的图片", return_code=True)
```

返回以下代码：

```python
from transformers import load_tool

image_generator = load_tool("huggingface-tools/text-to-image")

image = image_generator(prompt="rivers and lakes")
```

然后你可以修改并自行执行该代码。