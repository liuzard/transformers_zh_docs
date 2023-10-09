版权所有 © 2020 HuggingFace 团队。

根据 Apache 许可证第 2.0 版（"许可证"），除非符合许可证的要求，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或事先书面同意，根据许可证分发的软件以“按原样”方式分发，不附带任何明示或暗示的担保或条件。有关特定语言的限制或条件，请参阅许可证。

⚠️请注意，此文件采用 Markdown 格式，但包含特定于我们 doc-builder（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确显示。

# RAG

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=rag">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-rag-blueviolet">
</a>
</div>

## 概述

检索增强生成（"RAG"）模型结合了预训练的密集检索（DPR）和序列到序列模型的能力。RAG 模型检索文档，将其传递给序列到序列模型，然后进行边缘化以生成输出。检索模块和序列到序列模块是从预训练模型初始化的，并且联合进行微调，允许检索和生成两者都适应下游任务。

它基于 Patrick Lewis、Ethan Perez、Aleksandara Piktus、Fabio Petroni、Vladimir
Karpukhin、Naman Goyal、Heinrich Küttler、Mike Lewis、Wen-tau Yih、Tim Rocktäschel、Sebastian Riedel 和 Douwe Kiela 的论文 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)。

论文中的摘要如下：

*大型预训练语言模型已被证明在其参数中存储了事实知识，并且在下游 NLP 任务上进行微调时取得了最先进的结果。然而，它们对访问和精确操纵知识的能力仍然有限，因此在知识密集型任务上，它们的表现落后于任务特定的架构。此外，为其决策提供来源和更新其世界知识仍然是开放的研究问题。具有可微分访问机制的预训练模型可以解决这个问题，但迄今为止只针对抽取型下游任务进行了研究。我们探索了一种用于检索增强生成（RAG）的通用微调方法 - 模型将预训练的参数型和非参数型内存结合起来用于语言生成。我们引入了 RAG 模型，其中参数型内存是一个预训练的序列到序列模型，非参数型内存是通过预训练的神经检索器访问维基百科的密集向量索引。我们比较了两种 RAG 记录，一种是在整个生成的序列中对应于相同的检索段落，另一种可以对每个标记使用不同的段落。我们在广泛的知识密集型 NLP 任务上进行了微调和评估，并在三个开放领域 QA 任务中取得了最先进的结果，优于参数型序列到序列模型和任务特定的检索和抽取架构。对于语言生成任务，我们发现 RAG 模型生成的语言比最先进的仅参数型序列到序列基线更具体、多样和实际明确。*

该模型由 [ola13](https://huggingface.co/ola13) 贡献。

提示：
- 检索增强生成（“RAG”）模型结合了预训练的密集检索（DPR）和 Seq2Seq 模型的能力。RAG 模型检索文档，将其传递给 Seq2Seq 模型，然后进行边缘化以生成输出。检索器和 Seq2Seq 模块是从预训练模型初始化的，并且联合进行微调，使得检索和生成都适应于下游任务。

## RagConfig

[[autodoc]] RagConfig

## RagTokenizer

[[autodoc]] RagTokenizer

## Rag 特定的输出

[[autodoc]] models.rag.modeling_rag.RetrievAugLMMarginOutput

[[autodoc]] models.rag.modeling_rag.RetrievAugLMOutput

## RagRetriever

[[autodoc]] RagRetriever

## RagModel

[[autodoc]] RagModel
    - forward

## RagSequenceForGeneration

[[autodoc]] RagSequenceForGeneration
    - forward
    - generate

## RagTokenForGeneration

[[autodoc]] RagTokenForGeneration
    - forward
    - generate

## TFRagModel

[[autodoc]] TFRagModel
    - call

## TFRagSequenceForGeneration

[[autodoc]] TFRagSequenceForGeneration
    - call
    - generate

## TFRagTokenForGeneration

[[autodoc]] TFRagTokenForGeneration
    - call
    - generate