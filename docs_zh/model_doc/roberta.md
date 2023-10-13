<!-- 版权 2020 年 HuggingFace 团队保留所有权利。

根据 Apache 许可证第 2.0 版 （“许可证”）的规定，你不得使用此文件，除非符合许可证的要求，你可以在以下网址获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”基础分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以了解许可的特定语言和限制。

⚠️ 请注意，此文件采用 Markdown 编写，但包含我们文档生成器（类似于 MDX）的特定语法，这可能无法在你的 Markdown 查看器中正确渲染。
-->

# RoBERTa

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=roberta">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-roberta-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/roberta-base">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
<a href="https://huggingface.co/papers/1907.11692">
<img alt="Paper page" src="https://img.shields.io/badge/Paper%20page-1907.11692-green">
</a>
</div>

## 概述

RoBERTa 模型是由 Yinhan Liu、Myle Ott、Naman Goyal、Jingfei Du、Mandar Joshi、Danqi Chen、Omer Levy、Mike Lewis、Luke Zettlemoyer、Veselin Stoyanov 在文章 [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) 中提出的。它基于 Google 在2018年发布的 BERT 模型。

RoBERTa 在 BERT 的基础上进行了改进，修改了关键超参数，去除了下一个句子预训练目标，并采用了更大的小批量和学习率进行训练。

下面是来自论文的摘要：

*语言模型的预训练带来了显著的性能提升，但是不同方法之间的细致比较是具有挑战性的。训练是计算密集型的，通常在不同大小的私有数据集上进行，并且正如我们将要展示的，超参数选择对最终结果有重要影响。我们对 BERT 预训练（Devlin et al., 2019）进行了一项复制性研究，认真测量了许多关键超参数和训练数据大小的影响。我们发现 BERT 训练不足，可以与之后发布的每个模型的性能相匹配甚至超过。我们的最佳模型在 GLUE、RACE 和 SQuAD 上取得了最新的最佳结果。这些结果突显了以前被忽视的设计选择的重要性，并对最近报道的改进来源提出了质疑。我们发布我们的模型和代码。*

提示：

- 此实现与 [`BertModel`] 相同，只是进行了微小的嵌入调整，并为 RoBERTa 预训练模型设置。
- RoBERTa 与 BERT 具有相同的架构，但使用字节级的 BPE 作为分词器（与 GPT-2 相同），并采用了不同的预训练方案。
- RoBERTa 不使用 `token_type_ids`，你无需指示哪个标记属于哪个片段。只需使用分隔标记 `tokenizer.sep_token`（或 `</s>`）将片段分隔开即可。
- 与 BERT 相同，但具有更好的预训练技巧：

    - 动态遮蔽：在每个时期对标记进行不同方式的遮蔽，而 BERT 仅一次性进行遮蔽
      - 一起达到 512 个标记的目标（因此句子的顺序可能涵盖多个文档）
      - 使用更大的批次进行训练
      - 使用字节作为子单元的 BPE，而不是字符（因为涉及到 Unicode 字符）
- [CamemBERT](camembert) 是 RoBERTa 的一个封装。请参考此页面了解使用示例。

此模型由 [julien-c](https://huggingface.co/julien-c) 贡献。原始代码可以在 [这里](https://github.com/pytorch/fairseq/tree/master/examples/roberta) 找到。

## 资源

以下是官方 Hugging Face 和社区（由 🌎 标识）资源列表，可帮助你开始使用 RoBERTa。如果你有兴趣提交资源以包含在此处，请随时发起拉取请求，我们将进行审查！此资源应该展示出某种新的东西，而不是重复现有的资源。

<PipelineTag pipeline="text-classification"/>

- 一篇关于使用 RoBERTa 和 [Inference API](https://huggingface.co/inference-api) 进行 Twitter 情感分析的博客：[入门：使用 RoBERTa 对 Twitter 进行情感分析](https://huggingface.co/blog/sentiment-analysis-twitter)。
- 一篇 [使用 Kili 和 Hugging Face AutoTrain 进行意见分类](https://huggingface.co/blog/opinion-classification-with-kili) 的博客，使用了 RoBERTa。
- 一篇关于如何 [对 RoBERTa 进行情感分析微调](https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb) 的笔记本。🌎
- [`RobertaForSequenceClassification`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb) 的支持。
- [`TFRobertaForSequenceClassification`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb) 的支持。
- [`FlaxRobertaForSequenceClassification`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb) 的支持。
- [文本分类任务指南](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [`RobertaForTokenClassification`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb) 的支持。
- [`TFRobertaForTokenClassification`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb) 的支持。
- [`FlaxRobertaForTokenClassification`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification) 的支持。
- 🤗Hugging Face 课程中的 [token分类](https://huggingface.co/course/chapter7/2?fw=pt) 章节。
- [token分类任务指南](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- 一篇关于如何使用 Transformers 和 Tokenizers 从头开始训练新语言模型（使用 RoBERTa）的博客：[如何训练新的语言模型](https://huggingface.co/blog/how-to-train)。
- [`RobertaForMaskedLM`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) 的支持。
- [`TFRobertaForMaskedLM`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb) 的支持。
- [`FlaxRobertaForMaskedLM`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb) 的支持。
- 🤗Hugging Face 课程中的 [掩码语言建模](https://huggingface.co/course/chapter7/3?fw=pt) 章节。
- [掩码语言建模任务指南](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- 一篇关于使用 RoBERTa 进行问答的博客：[利用Optimum和Transformers Pipelines实现加速推理](https://huggingface.co/blog/optimum-inference)。
- [`RobertaForQuestionAnswering`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) 的支持。
- [`TFRobertaForQuestionAnswering`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb) 的支持。
- [`FlaxRobertaForQuestionAnswering`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering) 的支持。
- 🤗Hugging Face 课程中的 [问答](https://huggingface.co/course/chapter7/7?fw=pt) 章节。
- [问答任务指南](../tasks/question_answering)

**多选**
- [`RobertaForMultipleChoice`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb) 的支持。
- [`TFRobertaForMultipleChoice`] 受此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb) 的支持。
- [多选任务指南](../tasks/multiple_choice)

## RobertaConfig

[[autodoc]] RobertaConfig

## RobertaTokenizer

[[autodoc]] RobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RobertaTokenizerFast

[[autodoc]] RobertaTokenizerFast
    - build_inputs_with_special_tokens

## RobertaModel

[[autodoc]] RobertaModel
    - forward

## RobertaForCausalLM

[[autodoc]] RobertaForCausalLM
    - forward

## RobertaForMaskedLM

[[autodoc]] RobertaForMaskedLM
    - forward

## RobertaForSequenceClassification

[[autodoc]] RobertaForSequenceClassification
    - forward

## RobertaForMultipleChoice

[[autodoc]] RobertaForMultipleChoice
    - forward

## RobertaForTokenClassification

[[autodoc]] RobertaForTokenClassification
    - forward

## RobertaForQuestionAnswering

[[autodoc]] RobertaForQuestionAnswering
    - forward

## TFRobertaModel

[[autodoc]] TFRobertaModel
    - call

## TFRobertaForCausalLM

[[autodoc]] TFRobertaForCausalLM
    - call

## TFRobertaForMaskedLM

[[autodoc]] TFRobertaForMaskedLM
    - call

## TFRobertaForSequenceClassification

[[autodoc]] TFRobertaForSequenceClassification
    - call

## TFRobertaForMultipleChoice

[[autodoc]] TFRobertaForMultipleChoice
    - call

## TFRobertaForTokenClassification

[[autodoc]] TFRobertaForTokenClassification
    - call

## TFRobertaForQuestionAnswering

[[autodoc]] TFRobertaForQuestionAnswering
    - call

## FlaxRobertaModel

[[autodoc]] FlaxRobertaModel
    - __call__

## FlaxRobertaForCausalLM

[[autodoc]] FlaxRobertaForCausalLM
    - __call__

## FlaxRobertaForMaskedLM

[[autodoc]] FlaxRobertaForMaskedLM
    - __call__

## FlaxRobertaForSequenceClassification

[[autodoc]] FlaxRobertaForSequenceClassification
    - __call__

## FlaxRobertaForMultipleChoice

[[autodoc]] FlaxRobertaForMultipleChoice
    - __call__

## FlaxRobertaForTokenClassification

[[autodoc]] FlaxRobertaForTokenClassification
    - __call__

## FlaxRobertaForQuestionAnswering

[[autodoc]] FlaxRobertaForQuestionAnswering
    - __call__