<!--版权所有2021年HuggingFace团队。版权所有。

根据Apache许可证，第2.0版（“许可证”）许可；除非符合许可证的规定，
否则您不得使用此文件。您可以在下面的链接中获得许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

除非有适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
没有任何明示或暗示的保证或条件。
请参阅许可证以获取特定权限和限制的语言。

⚠️请注意，此文件是Markdown格式的，但包含我们doc-builder（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确呈现。

-->

# 通用实用函数

本页列出了Transformers库中在`utils.py`文件中找到的所有通用实用函数。

大部分实用函数仅在您研究库中的通用代码时才有用。

## 枚举和命名元组

[[autodoc]] utils.ExplicitEnum

[[autodoc]] utils.PaddingStrategy

[[autodoc]] utils.TensorType

## 特殊装饰器

[[autodoc]] utils.add_start_docstrings

[[autodoc]] utils.add_start_docstrings_to_model_forward

[[autodoc]] utils.add_end_docstrings

[[autodoc]] utils.add_code_sample_docstrings

[[autodoc]] utils.replace_return_docstrings

## 特殊属性

[[autodoc]] utils.cached_property

## 其他实用函数

[[autodoc]] utils._LazyModule