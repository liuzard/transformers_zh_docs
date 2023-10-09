<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache License，Version 2.0（“许可证”）获得许可；除非符合License的规定，
否则你不得使用该文件。你可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，
不附带任何形式的担保或条件。请参阅许可证中有关特定语言的规定，
以及许可证下给予的限制。

⚠️请注意，此文件采用Markdown格式，但其中包含特定语法，我们的doc-builder中包含特定的语法（类似于MDX），可能不能在Markdown查看器中正确呈现。-->

# Blenderbot Small

请注意，[`BlenderbotSmallModel`]和 [`BlenderbotSmallForConditionalGeneration`]与检查点 [facebook/blenderbot-90M](https://huggingface.co/facebook/blenderbot-90M) 仅组合使用。较大的Blenderbot检查点应使用 [`BlenderbotModel`] 和 [`BlenderbotForConditionalGeneration`]。

## 概述

Blender chatbot模型在2020年4月30日由[Recipes for building an open-domain chatbot](https://arxiv.org/pdf/2004.13637.pdf)中提出，作者是Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston。

该论文的摘要如下:

建立开放域聊天机器人是机器学习研究中的一个具有挑战性的领域。以往的研究表明，将神经模型的参数数量和训练数据的规模扩大可以得到改进的结果，但是我们展示了要构建一个性能良好的聊天机器人还需要其他的要素。优秀的对话需要一些技能，一个专业的会话者可以巧妙地将这些技能无缝地结合起来：提供引人入胜的话题并倾听对方，恰如其分地展示知识、移情和个性，同时保持连贯的个性。我们展示了大规模模型在给定适当的训练数据和生成策略的情况下可以学会这些技能。我们构建了这些食谱的90M、2.7B和9.4B参数模型的变种，并公开提供我们的模型和代码。人类评价表明我们的最佳模型在多轮对话中的吸引力和人性化的度量方面优于现有方法。然后我们通过分析模型的失败案例讨论了这项工作的局限性。

提示:

- Blenderbot Small是一个带有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧填充输入。

这个模型的贡献者是[patrickvonplaten](https://huggingface.co/patrickvonplaten)。作者的代码可以在[https://github.com/facebookresearch/ParlAI](https://github.com/facebookresearch/ParlAI)找到。

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## BlenderbotSmallConfig

[[autodoc]] BlenderbotSmallConfig

## BlenderbotSmallTokenizer

[[autodoc]] BlenderbotSmallTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## BlenderbotSmallTokenizerFast

[[autodoc]] BlenderbotSmallTokenizerFast

## BlenderbotSmallModel

[[autodoc]] BlenderbotSmallModel
    - forward

## BlenderbotSmallForConditionalGeneration

[[autodoc]] BlenderbotSmallForConditionalGeneration
    - forward

## BlenderbotSmallForCausalLM

[[autodoc]] BlenderbotSmallForCausalLM
    - forward

## TFBlenderbotSmallModel

[[autodoc]] TFBlenderbotSmallModel
    - call

## TFBlenderbotSmallForConditionalGeneration

[[autodoc]] TFBlenderbotSmallForConditionalGeneration
    - call

## FlaxBlenderbotSmallModel

[[autodoc]] FlaxBlenderbotSmallModel
    - __call__
    - encode
    - decode

## FlaxBlenderbotForConditionalGeneration

[[autodoc]] FlaxBlenderbotSmallForConditionalGeneration
    - __call__
    - encode
    - decode