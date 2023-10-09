<!--版权2022年The HuggingFace团队。版权所有。

根据Apache许可证Version 2.0（“许可证”）进行许可;除非符合许可证的规定，否则您不得使用本文件。
您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“原样”基础分发的，
没有任何明示或暗示的保证或条件。有关许可证下的特定语言的详细信息，请参阅许可证。
下的特定语。

⚠️请注意，本文件采用Markdown格式，但包含我们文档构建工具（类似于MDX）的特定语法，
在您的Markdown查看器中可能无法正确渲染。-->

# 创建

每个框架都有一个用于文本生成的`GenerationMixin`类实现的生成方法：

- PyTorch [`~generation.GenerationMixin.generate`] 在 [`~generation.GenerationMixin`] 中实现。
- TensorFlow [`~generation.TFGenerationMixin.generate`] 在 [`~generation.TFGenerationMixin`] 中实现。
- Flax/JAX [`~generation.FlaxGenerationMixin.generate`] 在 [`~generation.FlaxGenerationMixin`] 中实现。

无论您选择的框架如何，都可以使用[`~generation.GenerationConfig`]类实例来对生成方法进行参数化。
有关生成方法的行为，请参考该类以获取完整的生成参数列表。

要了解如何检查模型的生成配置，了解默认值，如何临时更改参数以及如何创建和保存自定义的生成配置，请参阅
[text generation strategies guide](../generation_strategies.md)。
该指南还解释了如何使用相关功能，如令牌流式传输。

## GenerationConfig

[[autodoc]] generation.GenerationConfig
	- from_pretrained
	- from_model_config
	- save_pretrained

## GenerationMixin

[[autodoc]] generation.GenerationMixin
	- generate
	- compute_transition_scores
	- greedy_search
	- sample
	- beam_search
	- beam_sample
	- contrastive_search
	- group_beam_search
	- constrained_beam_search

## TFGenerationMixin

[[autodoc]] generation.TFGenerationMixin
	- generate
	- compute_transition_scores

## FlaxGenerationMixin

[[autodoc]] generation.FlaxGenerationMixin
	- generate