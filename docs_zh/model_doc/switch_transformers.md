<!--版权所有2022 HuggingFace团队。保留所有权利。

根据Apache许可证，版本2.0（“许可证”）许可；除非符合许可证，否则你不得使用此文件。你可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”​​的基础分发的，不附带任何形式的保证或条件。有关许可的特定语言，请参阅许可证。

⚠️请注意，此文件采用Markdown格式，但包含针对我们的构建者（类似于MDX）的特定语法，可能无法在Markdown查看器中正确显示。-->

# SwitchTransformers

## 概述

SwitchTransformers模型是由William Fedus、Barret Zoph和Noam Shazeer在[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)中提出的。

Switch Transformer模型使用了一种稀疏的T5编码器-解码器架构，其中MLP被一个专家混合体（MoE）所取代。路由机制（在本例中是top 1）将每个标记关联到一个专家，其中每个专家都是一个密集的MLP。虽然开关变压器的权重比其等效的密集模型多得多，但由于稀疏性，可以实现更好的扩展性和更好的规模微调性能。
在前向传递过程中，只使用权重的一小部分。路由机制允许模型根据需要选择相关的权重，从而提高了模型的容量，而不会增加操作的数量。


该论文的摘要如下：

*在深度学习中，模型通常为所有输入重用相同的参数。专家混合模型（MoE）违背了这一点，而是为每个传入的示例选择不同的参数。结果是一个稀疏激活的模型 - 具有大量参数 - 但是计算成本不变。然而，尽管MoE取得了一些显着的成功，但由于复杂性、通信成本和训练不稳定性等原因，它的广泛采用受到了阻碍 - 我们用Switch Transformer来解决这些问题。我们简化了MoE路由算法，并设计了具有降低通信和计算成本的直观改进模型。我们提出的训练技术有助于解决不稳定性问题，并且我们首次证明可以使用更低精度（bfloat16）格式训练大型稀疏模型。我们基于T5-Base和T5-Large设计了模型，以在相同的计算资源下提高高达7倍的预训练速度。这些改进在多语言环境中延伸，我们在所有101种语言中均获得了对mT5-Base版本的增益。最后，我们使用“庞大的干净爬取语料库”对语言模型的当前规模进行了升级，并在T5-XXL模型上实现了4倍的加速。*

提示：

- SwitchTransformers使用[`T5Tokenizer`]，可以直接从每个模型的存储库加载。
- 发布的权重是在英语[掩码语言建模](https://moon-ci-docs.huggingface.co/docs/transformers/pr_19323/en/glossary#general-terms)任务上进行预训练的，并且应进行微调。

该模型由[Younes Belkada](https://huggingface.co/ybelkada)和[Arthur Zucker](https://huggingface.co/ArtZucker)贡献。
原始代码可以在[这里](https://github.com/google/flaxformer/tree/main/flaxformer/architectures/moe)找到。

## 资源

- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## SwitchTransformersConfig

[[autodoc]] SwitchTransformersConfig

## SwitchTransformersTop1Router

[[autodoc]] SwitchTransformersTop1Router
    - _compute_router_probabilities
    - forward

## SwitchTransformersSparseMLP

[[autodoc]] SwitchTransformersSparseMLP
    - forward

## SwitchTransformersModel

[[autodoc]] SwitchTransformersModel
    - forward

## SwitchTransformersForConditionalGeneration

[[autodoc]] SwitchTransformersForConditionalGeneration
    - forward

## SwitchTransformersEncoderModel

[[autodoc]] SwitchTransformersEncoderModel
    - forward