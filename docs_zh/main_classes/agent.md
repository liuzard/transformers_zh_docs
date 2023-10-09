<!--
版权所有2023年HuggingFace团队保留。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证要求，否则不得使用此文件。您可以获取许可证副本，网址为

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于“按原样”提供，不附带任何明示或暗示的保证或条件。有关许可证的特定语言和限制，请参阅许可证。

⚠️请注意，此文件使用Markdown格式，但包含我们doc-builder（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正确渲染。
-->

# 代理和工具

<Tip warning={true}>

Huggingface团队的Transformers Agents是一个实验性的API，随时可能发生变化。由于API或底层模型容易改变，代理返回的结果可能会有所不同。

</Tip>

要了解有关代理和工具的更多信息，请确保阅读[入门指南](../transformers_agents.md)。本页面包含底层类的API文档。

## 代理

我们提供三种类型的代理：[`HfAgent`] 使用开源模型的推理端点，[`LocalAgent`] 在本地使用您选择的模型，以及[`OpenAiAgent`] 使用OpenAI的闭源模型。

### HfAgent

[[autodoc]] HfAgent

### LocalAgent

[[autodoc]] LocalAgent

### OpenAiAgent

[[autodoc]] OpenAiAgent

### AzureOpenAiAgent

[[autodoc]] AzureOpenAiAgent

### Agent

[[autodoc]] Agent
    - chat
    - run
    - prepare_for_new_chat

## 工具

### load_tool

[[autodoc]] load_tool

### Tool

[[autodoc]] Tool

### PipelineTool

[[autodoc]] PipelineTool

### RemoteTool

[[autodoc]] RemoteTool

### launch_gradio_demo

[[autodoc]] launch_gradio_demo

## 代理类型

代理可以处理工具之间的任何类型的对象；工具完全支持多模态，可以接受和返回文本、图像、音频、视频等其他类型。为了增加工具之间的兼容性，以及正确地在ipython（jupyter，colab，ipython笔记本等）中呈现这些返回结果，我们在这些类型周围实现了包装类。

包装对象应该继续按照最初的行为；文本对象仍应像字符串一样工作，图像对象应该仍然是`PIL.Image`类型。

这些类型有三个特定目的：

- 在类型上调用`to_raw`应返回底层对象
- 在类型上调用`to_string`应将对象作为字符串返回：在`AgentText`的情况下可能是字符串，但在其他实例中将是对象序列化版本的路径
- 在ipython内核中显示它应正确显示对象

### AgentText

[[autodoc]] transformers.tools.agent_types.AgentText

### AgentImage

[[autodoc]] transformers.tools.agent_types.AgentImage

### AgentAudio

[[autodoc]] transformers.tools.agent_types.AgentAudio
