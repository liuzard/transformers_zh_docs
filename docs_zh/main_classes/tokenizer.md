<!--
版权所有2020 The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证规定的使用，否则不得使用此文件。你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则按“原样”分发的软件在没有明示或暗示的情况下分发，
请参阅许可证以获得特定语言的权限和限制。

⚠️需要注意的是，这个文件是Markdown格式的，但包含特定于我们文档构建器（类似于MDX）的语法，这可能不会在你的Markdown查看器中正确显示。

-->

# Tokenizer

一个Tokenizer负责为模型准备输入。该库包含所有模型的Tokenizer。大多数Tokenizer都有两种类型：一种是完整的Python实现，另一种是基于Rust库[🤗 Tokenizers](https://github.com/huggingface/tokenizers)的"快速"实现。"快速"实现提供了以下功能：

1. 在进行批处理分词时显著加快速度；
2. 附加方法用于在原始字符串（字符和单词）和标记空间之间进行映射（例如，获取包含给定字符的标记的索引或对应于给定标记的字符范围）。

基类[`PreTrainedTokenizer`]和[`PreTrainedTokenizerFast`]实现了模型输入中对字符串输入进行编码的常用方法（参见下文），以及从本地文件或目录或从提供的预训练Tokenizer（从HuggingFace的AWS S3存储库下载）实例化/保存Python和"快速" tokenizer。它们都依赖于[`~tokenization_utils_base.PreTrainedTokenizerBase`]，其中包含了常用的方法，以及[`~tokenization_utils_base.SpecialTokensMixin`]。

因此，[`PreTrainedTokenizer`]和[`PreTrainedTokenizerFast`] 实现了所有Tokenizer的主要方法：

- 分词（将字符串拆分为子词标记字符串），将标记字符串转换为ID以及进行编码/解码（即，分词和转换为整数）。
- 以一种与底层结构（BPE、SentencePiece等）无关的方式添加新标记到词汇表中。
- 管理特殊标记（如掩码、句子开头等）：添加它们，将它们分配给Tokenizer中的属性以供轻松访问，并确保它们在分词过程中不会被拆分。

[`BatchEncoding`]保存了[`~tokenization_utils_base.PreTrainedTokenizerBase`]的编码方法(`__call__`、`encode_plus`和`batch_encode_plus`)的输出，是从Python字典派生出来的。当Tokenizer是纯Python分词器时，此类的行为就像标准的Python字典一样，保存这些方法计算的各种模型输入（`input_ids`、`attention_mask`等）。当Tokenizer是“快速”分词器时（即由HuggingFace的[tokenizers库](https://github.com/huggingface/tokenizers)支持），此类还提供几种高级对齐方法，可用于在原始字符串（字符和单词）与标记空间之间进行映射（例如，获取包含给定字符的标记的索引或对应于给定标记的字符范围）。

## PreTrainedTokenizer

[[autodoc]] PreTrainedTokenizer
    - __call__
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## PreTrainedTokenizerFast

[`PreTrainedTokenizerFast`]依赖于[tokenizers](https://huggingface.co/docs/tokenizers)库。从🤗 Tokenizers库获取的Tokenizer可以很简单地加载到🤗 Transformers中。请查看[Using tokenizers from 🤗 tokenizers](../fast_tokenizers.md)页面了解如何使用。

[[autodoc]] PreTrainedTokenizerFast
    - __call__
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## BatchEncoding

[[autodoc]] BatchEncoding
