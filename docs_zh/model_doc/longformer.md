<!--版权所有2020 HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），除非符合许可证的规定，
否则不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据本许可证分发的软件按原样提供，
没有任何明示或暗示的保证或条件。请参阅许可证了解许可证下的特定语言和限制。

⚠️请注意，此文件是Markdown格式的，但包含有关我们文档构建器的特定语法（类似于MDX），
这可能在你的Markdown查看器中无法正确呈现。-->

# Longformer

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=longformer">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-longformer-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/longformer-base-4096-finetuned-squadv1">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

Longformer模型是由Iz Beltagy、Matthew E. Peters和Arman Cohan在[Longformer：The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf)中提出的。

来自本文的摘要如下：

*由于自我注意力操作的复杂度与序列长度呈二次方关系，基于Transformer的模型无法处理长序列。为了解决这个问题，
我们引入了Longformer，它的注意力机制与序列长度呈线性关系，从而可以轻松处理成千上万个标记或更长文档。
Longformer的注意力机制可以直接替换标准的自我注意力，并将本地滑动窗口注意力与任务驱动的全局注意力相结合。
在之前关于长序列Transformer的研究中，我们在字符级语言建模上评估Longformer，并在text8和enwik8上取得了最先进的结果。
与大多数之前的工作不同的是，我们还对Longformer进行了预训练，并在多个下游任务上进行了微调。
我们预训练的Longformer在处理长文档任务上始终优于RoBERTa，并在WikiHop和TriviaQA上取得了最新的最好成绩。*

提示：

- 由于Longformer基于RoBERTa，因此它没有`token_type_ids`。你不需要指示哪个标记属于哪个段落。
  只需使用分隔符`tokenizer.sep_token`（或`</s>`）分隔不同的段落即可。
- 转换器模型通过稀疏矩阵替代注意力矩阵以提高速度。通常，本地上下文（例如，左侧和右侧两个标记是什么？）足以针对给定的标记采取行动。
  仍然为一些预先选择的输入标记提供了全局注意力，但是注意力矩阵的参数要少得多，从而加快了速度。有关更多信息，请参见本地注意力部分。

此模型由[beltagy](https://huggingface.co/beltagy)提供。作者的代码可以在[此处](https://github.com/allenai/longformer)找到。

## Longformer自注意力

Longformer自注意力在“本地”上下文和“全局”上下文中进行自注意力。大多数标记只与彼此“本地”关注，这意味着每个标记都与其前\\(\frac{1}{2} w\\)个标记和后\\(\frac{1}{2} w\\)个标记进行关注，其中\\(w\\)是在`config.attention_window`中定义的窗口长度。
注意，`config.attention_window`可以是`List`类型，以定义每个层的不同\\(w\\)。仅有少数选择的标记与所有其他标记“全局”关注，就像`BertSelfAttention`中所做的那样。

请注意，“本地”和“全局”关注的标记是由不同的查询、键和值矩阵进行投影。还要注意，每个“本地”关注的标记不仅要关注其窗口\\(w\\)内的标记，还要关注所有“全局”关注的标记，从而使全局注意力*对称*。

用户可以通过适当设置张量`global_attention_mask`来定义哪些标记“本地”关注，哪些标记“全局”关注。所有Longformer模型都使用以下逻辑来处理`global_attention_mask`：

- 0：标记“本地”关注
- 1：标记“全局”关注

有关更多信息，请参阅[`～LongformerModel.forward`]方法。

使用Longformer自注意力，通常表示内存和时间的查询-键乘法运算的内存和时间复杂度可以从 \\(\mathcal{O}(n_s \times n_s)\\)降低到 \\(\mathcal{O}(n_s \times w)\\)，其中\\(n_s\\)是序列长度，\\(w\\)是平均窗口大小。假设“全局”关注的标记数量相对于“本地”关注的标记数量可以忽略不计。

有关更多信息，请参见官方[论文](https://arxiv.org/pdf/2004.05150.pdf)。

## 训练

[`LongformerForMaskedLM`]的训练方式与[`RobertaForMaskedLM`]完全相同
应如下使用：

```python
input_ids = tokenizer.encode("这是来自[MASK]训练数据的句子", return_tensors="pt")
mlm_labels = tokenizer.encode("这是来自训练数据的句子", return_tensors="pt")

loss = model(input_ids, labels=input_ids, masked_lm_labels=mlm_labels)[0]
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## LongformerConfig

[[autodoc]] LongformerConfig

## LongformerTokenizer

[[autodoc]] LongformerTokenizer

## LongformerTokenizerFast

[[autodoc]] LongformerTokenizerFast

## Longformer特定输出

[[autodoc]] models.longformer.modeling_longformer.LongformerBaseModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling

[[autodoc]] models.longformer.modeling_longformer.LongformerMaskedLMOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerSequenceClassifierOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerTokenClassifierOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutputWithPooling

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerMaskedLMOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerQuestionAnsweringModelOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerSequenceClassifierOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerMultipleChoiceModelOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerTokenClassifierOutput

## LongformerModel

[[autodoc]] LongformerModel
    - forward

## LongformerForMaskedLM

[[autodoc]] LongformerForMaskedLM
    - forward

## LongformerForSequenceClassification

[[autodoc]] LongformerForSequenceClassification
    - forward

## LongformerForMultipleChoice

[[autodoc]] LongformerForMultipleChoice
    - forward

## LongformerForTokenClassification

[[autodoc]] LongformerForTokenClassification
    - forward

## LongformerForQuestionAnswering

[[autodoc]] LongformerForQuestionAnswering
    - forward

## TFLongformerModel

[[autodoc]] TFLongformerModel
    - call

## TFLongformerForMaskedLM

[[autodoc]] TFLongformerForMaskedLM
    - call

## TFLongformerForQuestionAnswering

[[autodoc]] TFLongformerForQuestionAnswering
    - call

## TFLongformerForSequenceClassification

[[autodoc]] TFLongformerForSequenceClassification
    - call

## TFLongformerForTokenClassification

[[autodoc]] TFLongformerForTokenClassification
    - call

## TFLongformerForMultipleChoice

[[autodoc]] TFLongformerForMultipleChoice
    - call