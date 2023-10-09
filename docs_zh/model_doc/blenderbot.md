<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）进行许可；除非符合许可证，否则不能使用此文件
。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”基础分发的
，没有任何明示或暗示的担保或条件。有关许可证的详细信息，请参阅许可证
特定语言在许可证下所规定的限制。

⚠️请注意，此文件是使用Markdown格式的，但包含了我们文档生成器（类似于MDX）的特定语法，可能不会在你的Markdown查看器中正确显示。

-->

# Blenderbot

**免责声明：**如果你看到一些奇怪的问题，请提交一个[GitHub问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

## 概述

Blender聊天机器人模型的提议来源于Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston于2020年4月30日提出的[Recipes for building an open-domain chatbot](https://arxiv.org/pdf/2004.13637.pdf)。

论文的摘要如下：

*为机器学习研究而构建开放域聊天机器人是一个具有挑战性的领域。虽然先前的研究表明，通过增加神经模型的参数数量和训练数据的规模可以得到改善的结果，但我们表明，其他因素对于高性能的聊天机器人来说也很重要。良好的对话需要许多技能，一个专业的对话者能够将它们无缝地融合在一起：提供有吸引力的话题并倾听他们的伙伴，适当地展示知识、同情心和个性，同时保持一致的个人形象。我们表明，当给予适当的训练数据和生成策略选择时，大规模的模型可以学习这些技能。我们构建了这些配方的90M、2.7B和9.4B参数模型的变体，并公开提供我们的模型和代码。人类评估表明，我们最好的模型在多轮对话的吸引力和人性度度量方面优于现有方法。然后，我们通过分析我们模型的失败案例来讨论这项工作的局限性。*

提示：

- Blenderbot是一个具有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧填充输入。

此模型由[sshleifer](https://huggingface.co/sshleifer)贡献。作者的代码可以在[这里](https://github.com/facebookresearch/ParlAI)找到。


## 实现注意事项

- Blenderbot使用基于[序列到序列模型的Transformer](https://arxiv.org/pdf/1706.03762.pdf)架构。
- 可在[模型中心](https://huggingface.co/models?search=blenderbot)找到可用的检查点。
- 这是*默认*的Blenderbot模型类。然而，一些较小的检查点（例如`facebook/blenderbot_small_90M`）具有不同的架构，因此应与[BlenderbotSmall](blenderbot-small)一起使用。


## 用法

这是一个模型使用的例子：

```python
>>> from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

>>> mname = "facebook/blenderbot-400M-distill"
>>> model = BlenderbotForConditionalGeneration.from_pretrained(mname)
>>> tokenizer = BlenderbotTokenizer.from_pretrained(mname)
>>> UTTERANCE = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer([UTTERANCE], return_tensors="pt")
>>> reply_ids = model.generate(**inputs)
>>> print(tokenizer.batch_decode(reply_ids))
["<s> That's unfortunate. Are they trying to lose weight or are they just trying to be healthier?</s>"]
```

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## BlenderbotConfig

[[autodoc]] BlenderbotConfig

## BlenderbotTokenizer

[[autodoc]] BlenderbotTokenizer
    - build_inputs_with_special_tokens

## BlenderbotTokenizerFast

[[autodoc]] BlenderbotTokenizerFast
    - build_inputs_with_special_tokens

## BlenderbotModel

有关*forward*和*generate*的参数，请参阅`transformers.BartModel`。

[[autodoc]] BlenderbotModel
    - forward

## BlenderbotForConditionalGeneration

有关*forward*和*generate*的参数，请参阅[`~transformers.BartForConditionalGeneration`]。

[[autodoc]] BlenderbotForConditionalGeneration
    - forward

## BlenderbotForCausalLM

[[autodoc]] BlenderbotForCausalLM
    - forward

## TFBlenderbotModel

[[autodoc]] TFBlenderbotModel
    - call

## TFBlenderbotForConditionalGeneration

[[autodoc]] TFBlenderbotForConditionalGeneration
    - call

## FlaxBlenderbotModel

[[autodoc]] FlaxBlenderbotModel
    - __call__
    - encode
    - decode

## FlaxBlenderbotForConditionalGeneration

[[autodoc]] FlaxBlenderbotForConditionalGeneration
    - __call__
    - encode
    - decode