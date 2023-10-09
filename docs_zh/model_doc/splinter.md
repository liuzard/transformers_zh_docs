版权所有 © 2021 HuggingFace团队。

根据Apache License，Version 2.0（“许可证”）许可；除非符合许可证的规定，否则不得使用此文件。您可以从以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件按“原样”分发，不附带任何形式的明示或暗示担保。请注意，此文件是Markdown格式的，但包含对于我们的文档构建器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确显示。

# Splinter

## 概述

Splinter模型是由Ori Ram、Yuval Kirstain、Jonathan Berant、Amir Globerson和Omer Levy在《Few-Shot Question Answering by Pretraining Span Selection》一文中提出的。Splinter是一个仅编码器的transformer（与BERT类似），在包含维基百科和多伦多图书语料库的大型语料库上使用重复跨度选择任务进行预训练。

文章中的摘要如下：

在几个问题回答基准测试中，经过微调的预训练模型已经达到了与人类相当的水平，这是通过对大约100,000个已注释的问题和答案进行微调实现的。我们探索了更现实的少样本设置，即只有几百个训练样例可用，并观察到标准模型的表现不佳，突出了当前预训练目标与问题回答之间的差异。我们提出了一种适用于问题回答的新的预训练方案：重复跨度选择。给定一个包含多组重复跨度的段落，我们对每组重复跨度中的所有跨度进行掩码，然后要求模型为每个被掩码的跨度在段落中选择正确的跨度。被掩码的跨度将被替换为一个特殊的标记，被视为一个问题表示，稍后在微调过程中用于选择答案跨度。结果模型在多个基准测试中获得了令人惊讶的好结果（例如，在只有128个训练样例的情况下，SQuAD上的F1值达到了72.7），同时在高资源设置中保持着竞争性的性能。

提示：

- Splinter是通过特殊的[QUESTION]标记来预测答案跨度的。这些标记将上下文化为用于预测答案的问题表示。这一层被称为QASS层，并且是SplinterForQuestionAnswering类的默认行为。因此：
- 使用SplinterTokenizer（而不是BertTokenizer），因为它已经包含了这个特殊的标记。此外，默认情况下，当给出两个序列（例如在run_qa.py脚本中）时，它会使用这个标记。
- 如果您计划在run_qa.py之外使用Splinter，请记住问题标记-在少样本设置中，这可能对于您的模型的成功至关重要。
- 请注意，对于每个Splinter大小，有两个不同的检查点。两者基本上是相同的，只是一个还有QASS层的预训练权重（*tau/splinter-base-qass*和*tau/splinter-large-qass*），而另一个没有（*tau/splinter-base*和*tau/splinter-large*）。这样做是为了支持在微调时随机初始化这一层，因为据论文中的一些案例表明，这样做可以获得更好的结果。

此模型由yuvalkirstain和oriram贡献。原始代码可以在此处找到：https://github.com/oriram/splinter。

## 文档资源

- 问题回答任务指南（../tasks/question-answering）

## SplinterConfig

[[autodoc]] SplinterConfig

## SplinterTokenizer

[[autodoc]] SplinterTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SplinterTokenizerFast

[[autodoc]] SplinterTokenizerFast

## SplinterModel

[[autodoc]] SplinterModel
    - forward

## SplinterForQuestionAnswering

[[autodoc]] SplinterForQuestionAnswering
    - forward

## SplinterForPreTraining

[[autodoc]] SplinterForPreTraining
    - forward