<!--版权所有2020 HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（"许可证"），您不得使用此文件，除非符合许可证的要求。您可以从以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于"按原样"的基础上提供的，不附带任何明示或暗示的担保或条件。请参阅许可证中特定的权限和限制条件。

⚠️注意，此文件为Markdown格式，但包含特定于我们的doc-builder（类似于MDX）的语法，可能无法在您的Markdown查看器中正确渲染。-->

# 数据整理器

数据整理器是使用数据集元素列表作为输入来构建批次的对象。这些元素与`train_dataset`或`eval_dataset`中的元素类型相同。

为了能够构建批次，数据整理器可能会应用一些处理（如填充）。其中一些（如`DataCollatorForLanguageModeling`）还会在已形成的批次上应用一些随机数据增强（如随机屏蔽）。

可以在[示例脚本](../examples)或[示例笔记本](../notebooks.md)中找到使用示例。


## 默认数据整理器

[[autodoc]] data.data_collator.default_data_collator

## DefaultDataCollator

[[autodoc]] data.data_collator.DefaultDataCollator

## DataCollatorWithPadding

[[autodoc]] data.data_collator.DataCollatorWithPadding

## DataCollatorForTokenClassification

[[autodoc]] data.data_collator.DataCollatorForTokenClassification

## DataCollatorForSeq2Seq

[[autodoc]] data.data_collator.DataCollatorForSeq2Seq

## DataCollatorForLanguageModeling

[[autodoc]] data.data_collator.DataCollatorForLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## DataCollatorForWholeWordMask

[[autodoc]] data.data_collator.DataCollatorForWholeWordMask
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## DataCollatorForPermutationLanguageModeling

[[autodoc]] data.data_collator.DataCollatorForPermutationLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens