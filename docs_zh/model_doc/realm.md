<!--版权 2022 The HuggingFace 团队版权所有。

根据 Apache 许可证2.0版（“许可证”）授权；除非符合许可证，否则您不得使用此文件。您可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据许可证分发的软件是基于“原样”的基础上，没有任何明示或暗示的保证或条件。请看许可证中的特定语言，了解许可证下的特定语言的权限和限制。

⚠️请注意，此文件是以 Markdown 格式编写的，但包含了我们 doc-builder（类似于 MDX）的特定语法，可能不会在您的 Markdown 查看器中正确呈现。-->

# REALM

## 概览

REALM 模型是由 Kelvin Guu、Kenton Lee、Zora Tung、Panupong Pasupat 和 Ming-Wei Chang 在[REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) 中提出的。这是一个检索增强的语言模型，首先从文字知识语料库中检索文档，然后利用检索到的文档来处理问答任务。

论文中的摘要如下所示：

*已经证明，语言模型预训练可以捕捉到大量的世界知识，对于问答等自然语言处理任务十分重要。然而，这些知识是以神经网络参数的方式隐式存储的，需要越来越大的网络来覆盖更多的事实。为了以更加模块化和可解释的方式捕捉知识，我们在语言模型预训练中增加了一个潜在的知识检索器，允许模型从诸如维基百科等大型语料库中检索和关注文档，该语料库在预训练、微调和推理期间都会被使用。我们首次展示了如何使用无监督的方式来预训练这样一个知识检索器，使用掩码语言模型作为学习信号，并通过考虑数百万个文档进行检索步骤的反向传播来训练。我们通过在 Open-domain Question Answering (Open-QA) 这一具有挑战性的任务上进行微调来证明了 Retrieval-Augmented Language Model pre-training (REALM) 的有效性。我们在三个流行的 Open-QA 基准上与最先进的显式和隐式知识存储模型进行了比较，并发现我们在准确性方面的表现显著优于所有以前的方法（4-16% 的绝对准确率），同时还提供了解释性和模块性等质量的优势。*

该模型由 [qqaatw](https://huggingface.co/qqaatw) 贡献。原始代码可在[此处](https://github.com/google-research/language/tree/master/language/realm)找到。

## RealmConfig

[[autodoc]] RealmConfig

## RealmTokenizer

[[autodoc]] RealmTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary
    - batch_encode_candidates

## RealmTokenizerFast

[[autodoc]] RealmTokenizerFast
    - batch_encode_candidates

## RealmRetriever

[[autodoc]] RealmRetriever

## RealmEmbedder

[[autodoc]] RealmEmbedder
    - forward

## RealmScorer

[[autodoc]] RealmScorer
    - forward

## RealmKnowledgeAugEncoder

[[autodoc]] RealmKnowledgeAugEncoder
    - forward

## RealmReader

[[autodoc]] RealmReader
    - forward

## RealmForOpenQA

[[autodoc]] RealmForOpenQA
    - block_embedding_to
    - forward