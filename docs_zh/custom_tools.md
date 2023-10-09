版权所有 © 2023 HuggingFace 团队。版权所有。

根据 Apache 授权证书，版本 2.0 进行许可（“许可证”）；
除非符合许可证，否则不得使用该文件。您可以在以下位置获取许可证的副本:

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”分发的，不附带任何明示或暗示的担保或条件。
有关特定语言管理权限和限制的详细信息，请参阅许可证。

⚠️ 注意，此文件是使用 Markdown 编写的，但包含特定语法，用于我们的文档生成器（类似于 MDX），可能无法正确渲染在您的 Markdown 查看器中。

# 自定义工具和提示

<Tip>

如果您不知道 transformers 中的工具和 agents 是什么，请先阅读[Transformers Agents](transformers_agents.md)页面。

</Tip>

<Tip warning={true}>

Transformers Agents 是一个实验性的 API，随时可能发生更改。由 agents 返回的结果可能会因 API 或基础模型的更改而有所不同。

</Tip>

创建和使用自定义工具和提示对于增强 agent 的功能并使其执行新任务至关重要。在本指南中，我们将介绍以下内容：

- 如何自定义提示
- 如何使用自定义工具
- 如何创建自定义工具

## 自定义提示

如[Transformers Agents](transformers_agents.md)中所述，agents 可以在 `~Agent.run` 和 `~Agent.chat` 模式下运行。
`run` 和 `chat` 模式都遵循相同的逻辑。驱动 agent 的语言模型会以一个长提示为条件，并生成下一个标记，直到遇到停止标记。
两种模式之间唯一的区别在于，在 `chat` 模式下，提示会与前一个用户输入和模型生成的内容一起扩展。这使得 agent 可以访问以前的交互，并似乎具有某种记忆。

### 提示的结构

让我们仔细看一下提示的结构，以了解如何对其进行最佳定制。提示大致分为四个部分。

1. 介绍：说明 agent 的行为方式、工具概念的解释。
2. 所有工具的描述。这由一个 `<<all_tools>>` 标记定义，该标记在运行时动态替换为用户定义/选择的工具。
3. 一组任务示例及其解决方法
4. 当前示例以及对解决方法的请求。

为了更好地理解每个部分，让我们查看 `run` 提示的简化版本：

````text
询问您执行任务，您的工作是编写一系列在 Python 中执行任务的简单命令。
[...]
如果有必要，您可以打印中间结果。

工具：
- 文档问答：这是一个关于文档（PDF）的问题答案工具。它接受一个名为“document”的输入，该输入应该是包含信息的文档，
以及一个“question”，即关于文档的问题。它返回一个包含问题答案的文本。
- 图像描述生成器：这是一个生成图像描述的工具。它接受一个名为“image”的输入，该输入应该是要生成描述的图像，并返回一个包含英文描述的文本。
[...]

任务：“回答关于变量‘question’中存储的图像的问题。问题是用法语写的。”

我将使用以下工具：使用 ‘translator’ 将问题翻译为英文，然后使用 ‘图像问答’ 回答输入图像上的问题。

答案：
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
print(f"The answer is {answer}")
```

任务：“识别‘document’中年龄最大的人，并创建一个以这个结果为横幅的图片。”

我将使用以下工具：使用 ‘document_qa’ 找到文档中年龄最大的人，然后使用 ‘image_generator’ 根据答案生成图片。

答案：
```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
image = image_generator("A banner showing " + answer)
```

[...]

任务：“给我画一幅河流和湖泊的图片”

我将使用以下
````

介绍（“Tools:”之前的文本）精确地解释了模型应该如何行动和它应该做什么。
这部分很可能不需要定制，因为 agent 应该始终以相同的方式行动。

第二部分（在“Tools”下面的项目符号）在调用 `run` 或 `chat` 时动态添加。在 `agent.toolbox` 中的工具越多，
代理选择正确工具的难度就越大，选择正确的工具序列就更加困难。

这里有与工具名和描述相等数量的项目符号。每个项目符号由工具的名称和描述组成：

```text
- <tool.name>: <tool.description>
```

我们可以通过加载 document_qa 工具并打印其名称和描述来快速验证这一点。

```py
from transformers import load_tool

document_qa = load_tool("document-question-answering")
print(f"- {document_qa.name}: {document_qa.description}")
```

结果是：
```text
- document_qa: This is a tool that answers a question about a document (pdf). It takes an input named `document` which should be the document containing the information, as well as a `question` that is the question about the document. It returns a text that contains the answer to the question.
```

我们可以看到工具的名称简短而明确。描述由两部分组成，第一部分解释了工具的功能，第二部分说明了预期的输入参数和返回值。

良好的工具名称和工具描述对于 agent 正确使用工具来说非常重要。请注意，agent 对工具的唯一了解是其名称和描述，
因此必须确保二者都能准确地书写，并与工具箱中现有工具的风格匹配。特别要确保描述以代码样式明确提及所有预期的参数名称，
以及预期的类型和对其的描述。

<Tip>

检查精选的 Transformers 工具的命名和描述，以更好地了解工具的名称和描述应该是什么样子。您可以使用 [`Agent.toolbox`] 属性查看所有工具。

</Tip>

第三部分包括一系列精选示例，展示了 agent 应该为什么样的用户请求生成什么样的代码。驱动 agent 的大型语言模型
非常擅长识别提示中的模式，并使用新数据重复该模式。因此，非常重要的是，示例的编写方式最大化 agent 生成正确可执行代码的可能性。

让我们看一个示例：

````text
任务：“识别‘document’中年龄最大的人，并创建一个以这个结果为横幅的图片。”

我将使用以下工具：使用 ‘document_qa’ 找到文档中年龄最大的人，然后使用 ‘image_generator’ 根据答案生成图片。

答案：
```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
image = image_generator("A banner showing " + answer)
```

````

驱动模型进行重复的模式有三个部分：任务的说明、agent 对自己将要做的任务的说明，以及生成的代码。
提示中的每个示例都具有这个确切的模式，以确保 agent 生成新标记时会重复完全相同的模式。

提示示例由 Transformers 团队精选并经过严格评估，其会在一组[问题陈述](https://github.com/huggingface/transformers/blob/main/src/transformers/tools/evaluate_agent.py)上进行测试，
以确保 agent 的提示尽可能好地解决 agent 的真实使用案例。

提示的最后部分对应于以下内容：
```text
任务：“给我画一幅河流和湖泊的图片”

我将使用以下
```

是一个最终的、未完成的示例，要求 agent 进行完成。未完成的示例根据实际用户输入动态创建。对于上述示例，用户运行了：

```py
agent.run("给我画一幅河流和湖泊的图片")
```

用户输入 - 也称为任务：““给我画一幅河流和湖泊的图片”被转换为 prompt 模板：“任务： <task> \n\n 我将使用以下”。
该句子构成了 agent 所有的条件，从而强烈影响 agent 完成示例的方式，使其与之前的示例完全相同。

不过，聊天模板与 run 模板具有相同的提示结构，但示例的风格略有不同，例如：

````text
[...]

=====

Human: 回答关于变量“question”中存储的图像的问题。

Assistant: 我将使用 `图像问答` 工具回答关于输入图像的问题。

```py
answer = image_qa(text=question, image=image)
print(f"The answer is {answer}")
```

Human: 我尝试了这段代码，它运行得没问题，但没有给出好的结果。问题是用法语写的。

Assistant: 在这种情况下，需要先将问题翻译成英文。我将使用 `翻译器` 工具进行翻译。

```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(text=question, image=image)
print(f"The answer is {answer}")
```

=====

[...]
````

与示例的 run 模板相反，每个 chat 模板示例之间有一个或多个 Human 与 Assistant 的交互。每个交互的结构与 run 模板示例类似。
首先提示 agent 生成需要完成的任务，然后生成代码。一个交互可能基于以前的交互，因此允许用户引用以前的交互，就像上面的用户输入“我尝试了这段代码”引用了先前生成的代理的代码一样。

运行 `.chat` 后，用户的输入或任务被转化为未完成的示例形式:
```text
Human: <user-input>\n\nAssistant:
```
agent 完成该任务。与 `run` 命令相反，`chat` 命令将完成的示例附加到 prompt，从而为下一个 `chat` 轮提供更多上下文。

现在我们知道了 prompt 的结构，让我们看看如何对其进行定制！

### 编写良好的用户输入

尽管大型语言模型在理解用户意图方面越来越好，但尽可能精确地表达自己的意图有助于 agent 选择正确的任务。
什么是尽可能精确地表达自己的意图呢？

agent 在其提示中看到一系列工具名称和它们的描述。工具越多，代理选择正确工具的难度就越大，
选择正确的工具序列就更加困难。让我们看一个常见的失败案例，这里我们只返回代码以进行分析。

```py
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")

agent.run("Show me a tree", return_code=True)
```

返回结果是：

```text
==Explanation from the agent==
I will use the following tool: `image_segmenter` to create a segmentation mask for the image.


==Code generated by the agent==
mask = image_segmenter(image, prompt="tree")
```

这可能不是我们想要的结果。相反，我们更有可能希望生成一幅树的图像。为了更加指导 agent 使用特定的工具，使用重要的关键词可以帮助 agent 更好地选择正确的任务。
让我们来看看。
```py
agent.toolbox["image_generator"].description
```

```text
'This is a tool that creates an image according to a prompt, which is a text description. It takes an input named `prompt` which contains the image description and outputs an image.
```

名称和描述使用到了 "image"、"prompt"、"create" 和 "generate" 这些关键词。在这里使用这些词很有可能效果更好。
让我们稍微修改一下我们的提示。

```py
agent.run("Create an image of a tree", return_code=True)
```

返回结果是：
```text
==Explanation from the agent==
I will use the following tool `image_generator` to generate an image of a tree.


==Code generated by the agent==
image = image_generator(prompt="tree")
```

好多了！看起来更接近我们想要的结果。简而言之，当您注意到 agent 难以将您的任务正确映射到正确的工具时，请尝试查找工具名称的最相关关键字，并尝试使用这些关键字得到更精确的任务请求。

### 自定义工具描述

正如我们之前所见，agent 能够访问每个工具的名称和描述。基本工具应该具有非常精确的名称和描述，
但您可能会发现在特定用例中更改工具的描述或名称会有所帮助。当添加了多个非常相似的工具或者您只想将代理用于特定领域（如图像生成和转换）时，这可能变得尤为重要。

一个常见的问题是当大量用于图像生成任务时，agent 会混淆图像生成与图像转换/修改问题，例如：

```py
agent.run("Make an image of a house and a car", return_code=True)
```
返回结果是：
```text
==Explanation from the agent== 
I will use the following tools `image_generator` to generate an image of a house and `image_transformer` to transform the image of a car into the image of a house.

==Code generated by the agent==
house_image = image_generator(prompt="A house")
car_image = image_generator(prompt="A car")
house_car_image = image_transformer(image=car_image, prompt="A house")
```

这很可能不是我们想要的结果。似乎 agent 难以理解 `image_generator` 与 `image_transformer` 之间的区别，并且经常使用这两者。

我们可以通过更改 `image_transformer` 的工具名称和描述来帮助 agent，使其更像 `下图修改工具`。

```py
agent.toolbox["modifier"] = agent.toolbox.pop("image_transformer")
agent.toolbox["modifier"].description = agent.toolbox["modifier"].description.replace(
    "transforms an image according to a prompt", "modifies an image"
)
```

现在，"modify" 是使用新图像修改器的强烈线索，这应该有助于上述提示。让我们再次运行它。

```py
agent.run("Make an image of a house and a car", return_code=True)
```

现在，我们得到了：
```text
== 代理的解释 ==
我将使用以下工具：`image_generator` 生成一幅房子的图片，然后使用 `image_generator` 生成一幅汽车的图片。


== 代理生成的代码 ==
house_image = image_generator(prompt="一座房子")
car_image = image_generator(prompt="一辆汽车")
```

这确实更接近我们所想要的！但是，我们希望在同一幅图片中有房子和汽车。将任务更加倾向于生成单幅图片可以帮助：

```py
agent.run("创建图片：'一座房子和一辆汽车'", return_code=True)
```

```text
== 代理的解释 ==
我将使用以下工具：`image_generator` 来生成一幅图片。


== 代理生成的代码 ==
image = image_generator(prompt="一座房子和一辆汽车")
```

<Tip warning={true}>

对于许多用例来说，代理仍然很脆弱，特别是对于稍微复杂一些的用例，比如生成多个对象的图像。代理本身和底层提示将在未来几个月中进一步改进，以确保代理对各种用户输入更加稳健。

</Tip>

### 自定义完整提示

为了给用户最大的灵活性，完整的提示模板，如[上文](#prompt的结构)所述，可以被用户覆盖。在这种情况下，请确保您的自定义提示包括一个介绍部分、一个工具部分、一个示例部分和一个未完成示例部分。如果要覆盖 `run` 提示模板，您可以按照以下方法操作：

```py
template = """ [...] """

agent = HfAgent(your_endpoint, run_prompt_template=template)
```

<Tip warning={true}>

请确保在 `template` 的某个位置定义了 `<<all_tools>>` 字符串和 `<<prompt>>`，这样代理就可以知道它拥有哪些工具，并正确插入用户的提示。

</Tip>

类似地，您可以在实例化时通过更改 `chat_prompt_template` 来覆盖 `chat` 提示模板。请注意，`chat` 模式总是使用以下格式进行交互：
```text
Human: <<task>>

Assistant:
```
所以重要的是确保自定义 `chat` 提示模板的示例也使用这种格式。
您可以按以下方式在实例化时覆盖 `chat` 模板。

```
template = """ [...] """

agent = HfAgent(url_endpoint=your_endpoint, chat_prompt_template=template)
```

<Tip warning={true}>

请确保在 `template` 的某个位置定义了 `<<all_tools>>` 字符串，这样代理就可以知道它拥有哪些工具。

</Tip>

在以上两种情况下，如果您希望使用社区中某个人托管的模板，可以传递一个存储库 ID，而不是提示模板。默认的提示短语[存储在此存储库中](https://huggingface.co/datasets/huggingface-tools/default-prompts)作为示例。

要上传您的自定义提示到 Hub 上的一个存储库，并与社区共享，请确保：
- 使用数据集存储库
- 将 `run` 命令的提示模板放入名为 `run_prompt_template.txt` 的文件中
- 将 `chat` 命令的提示模板放入名为 `chat_prompt_template.txt` 的文件中

## 使用自定义工具

在本部分，我们将利用两个现有的专用于图像生成的自定义工具：

- 我们使用 [huggingface-tools/image-transformation](https://huggingface.co/spaces/huggingface-tools/image-transformation) 替换为 [diffusers/controlnet-canny-tool](https://huggingface.co/spaces/diffusers/controlnet-canny-tool)，以允许进行更多的图像修改。
- 我们为图像上采样添加了一个新的工具到默认工具箱中：[diffusers/latent-upscaler-tool](https://huggingface.co/spaces/diffusers/latent-upscaler-tool) 替换了现有的图像转换工具 image-transformation。

我们将首先使用便捷的 `load_tool` 函数加载自定义工具：

```py
from transformers import load_tool

controlnet_transformer = load_tool("diffusers/controlnet-canny-tool")
upscaler = load_tool("diffusers/latent-upscaler-tool")
```

在将自定义工具添加到代理之后，工具的描述和名称会自动包含在代理的提示中。因此，自定义工具需要有一个良好的描述和名称，以便代理能够理解如何使用它们。让我们看一下 `controlnet_transformer` 的描述和名称：

```py
print(f"Description: '{controlnet_transformer.description}'")
print(f"Name: '{controlnet_transformer.name}'")
```

结果为：
```text
Description: 'This is a tool that transforms an image with ControlNet according to a prompt. 
It takes two inputs: `image`, which should be the image to transform, and `prompt`, which should be the prompt to use to change it. It returns the modified image.'
Name: 'image_transformer'
```

名称和描述是准确的，并符合 [精心策划的工具集](transformers_agents.md#a-curated-set-of-tools) 的风格。接下来，让我们使用 `controlnet_transformer` 和 `upscaler` 实例化一个代理：

```py
tools = [controlnet_transformer, upscaler]
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=tools)
```

此命令应该会给出以下信息：

```text
image_transformer has been replaced by <transformers_modules.diffusers.controlnet-canny-tool.bd76182c7777eba9612fc03c0
8718a60c0aa6312.image_transformation.ControlNetTransformationTool object at 0x7f1d3bfa3a00> as provided in `additional_tools`
```

精心策划的工具已经有了一个 `image_transformer` 工具，现在被我们自定义的工具所替换。

<Tip>

如果您想为与现有工具完全相同的任务使用自定义工具，替换现有工具可能会很有好处，因为代理在使用特定任务时非常熟练。请注意，此时自定义工具应遵循与被替换工具完全相同的 API，或者您应该调整提示模板，以确保使用该工具的所有示例都得到更新。

</Tip>

上采样工具的名称为 `image_upscaler`，还不存在于默认的工具箱中，因此只需将其添加到工具列表中即可。
您可以通过查看代理的 `agent.toolbox` 属性来随时查看代理当前可用的工具箱：

```py
print("\n".join([f"- {a}" for a in agent.toolbox.keys()]))
```

```text
- document_qa
- image_captioner
- image_qa
- image_segmenter
- transcriber
- summarizer
- text_classifier
- text_qa
- text_reader
- translator
- image_transformer
- text_downloader
- image_generator
- video_generator
- image_upscaler
```

请注意，`image_upscaler` 现在是代理的工具箱的一部分。

让我们现在尝试一下新增的工具！我们将重新使用在[Transformers Agents Quickstart](transformers_agents.md#single-execution-run)中生成的图像。

```py
from diffusers.utils import load_image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png"
)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200> 

让我们将图像转换为美丽的冬季风景图：

```py
image = agent.run("转换图像：'一片冰冻的湖泊和多雪的森林'", image=image)
```

```text
== 代理的解释 ==
我将使用以下工具：`image_transformer` 来转换图像。


== 代理生成的代码 ==
image = image_transformer(image, prompt="一片冰冻的湖泊和多雪的森林")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_winter.png" width=200> 

这个新的图像处理工具是基于 ControlNet 的，它可以对图像进行非常强大的修改。默认情况下，图像处理工具返回大小为 512x512 像素的图像。让我们看看是否可以将其放大。

```py
image = agent.run("放大图像", image)
```

```text
== 代理的解释 ==
我将使用以下工具：`image_upscaler` 来放大图像。


== 代理生成的代码==
upscaled_image = image_upscaler(image)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_winter_upscale.png" width=400> 

代理根据描述和名称选择了我们的提示 "放大图像" 和我们刚刚添加的 upscaler 工具，并成功运行了它。

接下来，让我们看看如何创建一个新的自定义工具。

### 添加新工具

在本部分，我们将展示如何创建一个新的工具，该工具可以添加到代理中。

#### 创建一个新的工具

我们首先创建一个工具。我们将添加一个不太有用但有趣的任务，该任务是获取在 Hugging Face Hub 上为给定任务下载量最多的模型。

我们可以通过以下代码完成：

```python
from huggingface_hub import list_models

task = "text-classification"

model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(model.id)
```

对于任务 `text-classification`，这将返回 `'facebook/bart-large-mnli'`，对于 `translation`，它将返回 `'t5-base`。

我们如何将其转换为代理可以利用的工具？所有工具都依赖于持有主要属性的超类 `Tool`。我们将创建一个继承自它的类：

```python
from transformers import Tool


class HFModelDownloadsTool(Tool):
    pass
```

这个类有一些需要：
- 一个 `name` 属性，它对应于工具本身的名称。为了与具有可执行名称的其他工具保持一致，我们将其命名为 `model_download_counter`。
- 一个 `description` 属性，它将用于填充代理的提示。
- `inputs` 和 `outputs` 属性。定义这两个属性将帮助 Python 解释器作出明智的类型选择，并允许生成一个 gradio-demo，当我们将我们的工具推送到 Hub 时。它们都是预期值的列表，可以是 `text`、`image` 或 `audio`。
- 一个 `__call__` 方法，其中包含推断代码。这是我们刚刚尝试过的代码！

这是我们的类现在的样子：

```python
from transformers import Tool
from huggingface_hub import list_models


class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = (
        "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. "
        "It takes the name of the category (such as text-classification, depth-estimation, etc), and "
        "returns the name of the checkpoint."
    )

    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
```

现在我们有了方便的自定义工具。将它保存在一个文件中并从主脚本中导入它。我们将这个文件命名为 `model_downloads.py`，这样最后的导入代码看起来像这样：

```python
from model_downloads import HFModelDownloadsTool

tool = HFModelDownloadsTool()
```

为了让其他人受益，并为初始化更简单，我们建议将它推送到属于您的命名空间的 Hub 上。为此，只需在 `tool` 变量上调用 `push_to_hub`：

```python
tool.push_to_hub("hf-model-downloads")
```

现在你的代码已经在 Hub 上了！我们来看看最后一步，即让代理使用它。

#### 让代理使用工具

现在我们有了一个在 Hub 上的工具，可以像这样实例化它（将用户名替换为您的工具）：

```python
from transformers import load_tool

tool = load_tool("lysandre/hf-model-downloads")
```

为了在代理中使用它，只需在代理初始化方法中的 `additional_tools` 参数中传递它：

```python
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=[tool])

agent.run(
    "您能为我大声朗读一下 Hugging Face Hub 上“text-to-video”任务中下载量最多的模型的名称吗？"
)
```
它将输出如下：
```text
== 代理生成的代码 ==
model = model_download_counter(task="text-to-video")
print(f"The model with the most downloads is {model}.")
audio_model = text_reader(model)


==结构化结果==
The model with the most downloads is damo-vilab/text-to-video-ms-1.7b.
```

并且生成了以下音频。

| **音频**                                                                                                                                   |
|--------------------------------------------------------------------------------------------------------------------------------------------|
| <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/damo.wav" type="audio/wav"/> |

<Tip>

根据语言模型机制，某些模型非常脆弱，需要非常准确的提示才能良好运行。具有良好定义的工具名称和描述对于工具被代理利用至关重要。

</Tip>

### 替换现有工具

替换现有工具只需要将新项目分配给代理的工具箱项。以下是如何执行此操作的示例：

```python
from transformers import HfAgent, load_tool

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
agent.toolbox["image-transformation"] = load_tool("diffusers/controlnet-canny-tool")
```

<Tip>

例如，替换工具时要小心！这也会调整代理的提示。这在某些情况下可能是有益的，如果您有一个更适合任务的更好的提示，但它也可能导致您的工具被选择得比其他工具更多，或者其他工具被选择而不是您所定义的。

</Tip>

## 使用 gradio-tools

[gradio-tools](https://github.com/freddyaboulton/gradio-tools) 是一个强大的库，允许使用 Hugging
Face Spaces 作为工具。它支持许多现有的 Spaces，以及可以使用它设计的自定义 Spaces。

我们通过使用 `Tool.from_gradio` 方法支持 `gradio-tools`。例如，我们想要利用 `gradio-tools` 工具包中提供的 `StableDiffusionPromptGeneratorTool` 工具来改进我们的提示并生成更好的图像。

我们首先从 `gradio_tools` 导入工具并实例化它：

```python
from gradio_tools import StableDiffusionPromptGeneratorTool

gradio_tool = StableDiffusionPromptGeneratorTool()
```

我们将该实例传递给 `Tool.from_gradio` 方法：

```python
from transformers import Tool

tool = Tool.from_gradio(gradio_tool)
```

现在，我们可以像处理常规自定义工具一样对待它。我们利用它来改进我们的提示 `一个穿着太空服的兔子`：

```python
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=[tool])

agent.run("生成一个 `prompt` 图像，先改进它。", prompt="一个穿着太空服的兔子")
```

## 模型充分利用了工具：<br>

Agent的解释：<br>

我将使用以下工具：`StableDiffusionPromptGenerator`来改善提示，然后使用`image_generator`根据改善后的提示生成图像。

Agent生成的代码：<br>

improved_prompt = StableDiffusionPromptGenerator(prompt)<br>
print(f"改善后的提示是 {improved_prompt}.")<br>
image = image_generator(improved_prompt)

最终生成图像之前：<br>

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png">

<Tip warning={true}>

gradio-tools要求使用*文本*输入和输出，即使在处理不同的模态时也是如此。此实现适用于图像和音频对象。目前这两者不兼容，但随着我们努力改进支持，它们将迅速兼容。

</Tip>

## 与Langchain的未来兼容性

我们喜欢Langchain，并且认为它具有非常吸引人的工具套件。为了处理这些工具，Langchain要求使用*文本*输入和输出，即使在处理不同的模态时也是如此。这通常是对象的序列化版本（即，保存到磁盘上）。

这种差异意味着transformers-agents和langchain之间不能处理多模态。我们希望在未来的版本中解决这个限制，并欢迎来自热衷于langchain的用户的任何帮助，帮助我们实现这种兼容性。

我们非常希望得到更好的支持。如果您愿意提供帮助，请[提出问题](https://github.com/huggingface/transformers/issues/new)并分享您的想法。