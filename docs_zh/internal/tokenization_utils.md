<!--
版权所有 2020 年 HuggingFace 团队保留。

根据 Apache 许可证，版本 2.0（“许可证”），您不能在不符合许可证的情况下使用此文件。您可以在以下网址获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非依法要求或书面同意，根据许可证分发的软件是按“原样”基础分发的，不附带任何明示或暗示的担保或条件。有关许可证的特定语言，请参阅许可证下列的限制条款。

⚠️ 请注意，此文件是 Markdown 格式的，但包含特定于我们的文档构建器（类似于 MDX）的语法，可能无法正确显示在您的 Markdown 查看器中。

-->

# Tokenizers 的实用程序

本页面列出了 Tokenizers 使用的所有实用函数，主要为 [`~tokenization_utils_base.PreTrainedTokenizerBase`] 类实现公共方法，该类同时实现了 [`PreTrainedTokenizer`] 和 [`PreTrainedTokenizerFast`]，以及混入类 [`~tokenization_utils_base.SpecialTokensMixin`]。

如果您正在研究库中的 tokenizers 代码，这些函数大多数只有在这种情况下才有用。

## PreTrainedTokenizerBase

[[autodoc]] tokenization_utils_base.PreTrainedTokenizerBase
- __call__
- all

## SpecialTokensMixin

[[autodoc]] tokenization_utils_base.SpecialTokensMixin

## 枚举和命名元组

[[autodoc]] tokenization_utils_base.TruncationStrategy

[[autodoc]] tokenization_utils_base.CharSpan

[[autodoc]] tokenization_utils_base.TokenSpan