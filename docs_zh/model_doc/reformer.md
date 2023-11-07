<!--
版权所有2020年HuggingFace团队保留。

根据Apache License版本2.0（“许可证”）的规定，你不得使用此文件，除非符合许可证的规定。你可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本许可证下的软件在“原样”基础上分发
没有任何形式的明示或暗示担保，包括但不限于对于特定用途和适销性的担保而不是
有关许可的特定语言和限制的明示或暗示的担保。

⚠️请注意，此文件为Markdown格式，但包含特定于我们的文档生成器的语法（类似于MDX），可能无法
在Markdown查看器中正确渲染。

-->

# Reformer

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=reformer">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-reformer-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/reformer-crime-and-punishment">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**DISCLAIMER：**这个模型目前仍处于开发中，如果你发现奇怪的现象，请[提交Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

## 概述

Reformer模型是由Nikita Kitaev，Łukasz Kaiser和Anselm Levskaya在论文中提出的[Reformer: 高效Transformer](https://arxiv.org/abs/2001.04451.pdf)。

论文中的摘要如下：

*大型的Transformer模型通常在许多任务上都可以实现最先进的结果，但是训练这些模型可能代价高昂，特别是在处理长序列时。我们引入了两种技术来提高Transformer的效率。首先，我们将点积注意力替换为使用局部敏感哈希的注意力，将其复杂度从O(L^2)降低为O(Llog(L))，其中L是序列的长度。此外，我们使用可逆残差层代替标准的残差层，这使得在训练过程中只需存储一次激活，而不是N次，其中N是层数。结果模型Reformer在性能上与Transformer模型相当，同时在处理长序列时更加内存高效且速度更快。*

该模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。作者的代码可以在[这里](https://github.com/google/trax/tree/master/trax/models/reformer)找到。

提示：

- Reformer不支持使用torch.nn.DataParallel，因为PyTorch中存在一个错误，参见[问题＃36035](https://github.com/pytorch/pytorch/issues/36035)。
- 使用轴向位置编码（详见下文）.这是一种机制，可以避免在序列长度非常大时拥有一个巨大的位置编码矩阵，通过将其分解为较小的矩阵。
- 使用LSH（局部敏感哈希）注意力（详见下文）替代传统的注意力。这是一种避免在注意力层中计算完整的查询-键积的技术。
- 通过使用可逆Transformer层来计算每个层的中间结果，并在向后传播过程中从下一层的输入中减去残差的方式来避免存储每层的中间结果（该过程使它们返回）或者在给定层内对结果重新计算（比存储它们低效，但节省内存）。
- 对块而不是整个批次进行前馈操作。

## 轴向位置编码

轴向位置编码最初在Google的[trax库](https://github.com/google/trax/blob/4d99ad4965bab1deba227539758d59f0df0fef48/trax/layers/research/position_encodings.py#L29)中实现，由本模型论文的作者进一步开发。在处理非常长的输入序列的模型中，传统的位置ID编码对于每个位置i、...、n_s存储大小为d的嵌入向量（即config.hidden_size）。这意味着如果序列长度为n_s = 2^19≈0.5M，而config.hidden_size = d = 2^10≈1000，则将会得到一个位置编码矩阵：

$$X_{i,j}, 其中 i \in \left[1,\ldots, d\right] 且 j \in \left[1,\ldots, n_s\right]$$

该矩阵本身需要存储超过5亿个参数。轴向位置编码将X_{i,j}分解为两个矩阵：

$$X^{1}_{i,j}, 其中 i \in \left[1,\ldots, d^1\right] 且 j \in \left[1,\ldots, n_s^1\right]$$

和

$$X^{2}_{i,j}, 其中 i \in \left[1,\ldots, d^2\right] 且 j \in \left[1,\ldots, n_s^2\right]$$

并满足以下条件：

$$d = d^1 + d^2 且 n_s = n_s^1 \times n_s^2 .$$

因此，我们有：

$$X_{i,j} = \begin{cases}
X^{1}_{i, k}, 如果i < d^1，其中 k = j \mod n_s^1 \\
X^{2}_{i - d^1, l}, 如果 i \ge d^1，其中 l = \lfloor\frac{j}{n_s^1}\rfloor
\end{cases}$$

直观地讲，位置嵌入向量$x_j \in \mathbb{R}^{d}$现在由两个因式分解的嵌入向量组成：$x^1_{k, l} + x^2_{l, k}$，其中参数`config.axial_pos_embds_dim`设置为元组$(d^1, d^2)$，其和必须等于`config.hidden_size`，参数`config.axial_pos_shape`设置为元组$(n_s^1, n_s^2)$，其乘积必须等于`config.max_embedding_size`，在训练期间，这个值必须等于`input_ids`的*序列长度*。

## LSH自注意力

在LSH（局部敏感哈希）自注意力中，键和查询投影权重是绑定在一起的，因此，键查询嵌入向量也是相应绑定的。LSH自注意力使用局部敏感哈希机制，该机制在[Practical and Optimal LSH for Angular Distance](https://arxiv.org/abs/1509.02897)中提出，将这些绑定的键查询嵌入向量分配给可能的`config.num_buckets`个桶中的一个。基本思想是，*余弦相似度*越接近的键查询嵌入向量，它们被分配到同一个桶的可能性就越大。

可以通过增加`config.num_hashes`或直接增加前向函数的`num_hashes`参数来改善LSH机制的准确性，以便LSH自注意力的输出更好地近似于“标准”完全自注意力的输出。然后对桶进行排序，并将其分组成查询键嵌入向量每个分组长度为`config.lsh_chunk_length`。对于每个分组，查询嵌入向量会关注到其本身的键向量（它们与自身绑定）以及`config.lsh_num_chunks_before`个前面相邻分组的键嵌入向量和`config.lsh_num_chunks_after`个后面相邻分组的键嵌入向量。

有关更多信息，请参见[原始论文](https://arxiv.org/abs/2001.04451)或这篇很棒的[博客文章](https://www.pragmatic.ml/reformer-deep-dive/)。

注意，`config.num_buckets`也可以分解为列表$(n_{\text{buckets}}^1, n_{\text{buckets}}^2)$。这样，将查询键嵌入向量分配给$(1, \ldots, n_{\text{buckets}})$之间的某个值，而是分配给$(1-1, \ldots, n_{\text{buckets}}^1-1, \ldots, 1-n_{\text{buckets}}^2, \ldots, n_{\text{buckets}}^1-n_{\text{buckets}}^2)$之间的某个值。这对于非常长的序列来说非常重要，可以节省内存。

在从头开始训练模型时，建议将`config.num_buckets=None`，以便根据序列长度实时计算`num_buckets`的良好值。然后将该值自动保存在配置中，并应在推断中重用。

使用LSH自注意力，查询键矩阵乘法操作的内存和时间复杂度可以从$\mathcal{O}(n_s \times n_s)$降低到$\mathcal{O}(n_s \times \log(n_s))$，通常在Transformer模型中代表内存和时间瓶颈，其中$n_s$是序列的长度。

## 局部自注意力

局部自注意力本质上是一个“普通”的自注意力层，具有键、查询和值的投影，但是分块处理，以便在每个长度为`config.local_chunk_length`的分块中，查询嵌入向量仅与其块中的键嵌入向量以及上述邻近块（config.local_num_chunks_before个前邻块和config.local_num_chunks_after个后邻块）的键嵌入向量进行关注。

使用局部自注意力，查询键矩阵乘法操作的内存和时间复杂度可以从$\mathcal{O}(n_s \times n_s)$降低到$\mathcal{O}(n_s \times \log(n_s))$，通常在Transformer模型中代表内存和时间的瓶颈，其中$n_s$是序列的长度。

## 训练

在训练过程中，我们必须确保将序列长度设置为能够被`config.lsh_chunk_length`和`config.local_chunk_length`的最小公倍数整除的值，并且轴向位置编码的参数要正确设置如上所述。Reformer非常内存高效，因此模型可以轻松地在长度为64000个标记的序列上进行训练。

对于训练，应该按照以下方式使用[`ReformerModelWithLMHead`]：

```python
input_ids = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")
loss = model(input_ids, labels=input_ids)[0]
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言模型任务指南](../tasks/language_modeling)
- [掩码语言模型任务指南](../tasks/masked_language_modeling)

## ReformerConfig

[[autodoc]] ReformerConfig

## ReformerTokenizer

[[autodoc]] ReformerTokenizer
    - save_vocabulary

## ReformerTokenizerFast

[[autodoc]] ReformerTokenizerFast

## ReformerModel

[[autodoc]] ReformerModel
    - forward

## ReformerModelWithLMHead

[[autodoc]] ReformerModelWithLMHead
    - forward

## ReformerForMaskedLM

[[autodoc]] ReformerForMaskedLM
    - forward

## ReformerForSequenceClassification

[[autodoc]] ReformerForSequenceClassification
    - forward

## ReformerForQuestionAnswering

[[autodoc]] ReformerForQuestionAnswering
    - forward
