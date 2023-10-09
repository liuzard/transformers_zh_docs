<!--版权 2020 年 HuggingFace 团队。版权所有。

根据 Apache 许可证第 2.0 版（“许可证”），除非符合许可证的规定，否则无法使用此文件。
你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据该许可证分发的软件是按
“原样”基础分发的，不附带任何明示或暗示的担保或条件。
有关明示或暗示的更多信息，请参阅许可证中的条款。

注意，该文件是以 Markdown 的格式编写的，但包含了特定语法，供我们的 doc-builder（类似于 MDX）使用，这可能不能
在你的 Markdown 查看器中正确渲染。-->

# DistilBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=distilbert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-distilbert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/distilbert-base-uncased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
<a href="https://huggingface.co/papers/1910.01108">
<img alt="Paper page" src="https://img.shields.io/badge/Paper%20page-1910.01108-green">
</a>
</div>

## 概述

DistilBERT 模型是在博客文章 [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a
distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5)和论文 [DistilBERT, a
distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/papers/1910.01108) 中提出的。DistilBERT 是
通过精简 BERT base 模型进行训练的一个小型、快速、便宜和轻量级的 Transformer 模型。相比于 *bert-base-uncased*，它的参数量减少了 40%，运行速度提高了 60%，同时在 GLUE 语言理解基准测试中保持了超过 95% 的 BERT 性能。

论文中的摘要如下：

*随着大规模预训练模型的迁移学习在自然语言处理（NLP）中变得越来越普遍，将这些大模型操作在边缘设备上或在计算训练或推断资源受限的条件下仍然具有挑战性。在本研究中，我们提出了一种方法来预训练一个更小的通用语言表示模型 DistilBERT，然后，该模型可以在各种任务上进行良好的性能微调，就像它的较大模型一样。虽然大多数之前的工作都研究了使用蒸馏方法构建特定任务模型的用途，但我们利用了蒸馏知识在预训练阶段，并显示一个 BERT 模型的大小可以减少 40%，而其语言理解能力保持在 97%，速度提高了 60%。为了利用预训练期间大型模型学习到的归纳偏差，我们引入了三重损失，结合语言建模、蒸馏和余弦相似度损失。我们的较小、更快和更轻的模型更便宜进行预训练，我们在概念验证实验和边缘设备的比较研究中证明了它的能力。*

提示：

- DistilBERT 模型没有 `token_type_ids`，你不需要指示哪个标记属于哪个片段。只需使用分隔标记 `tokenizer.sep_token`（或 `[SEP]`）将片段分开即可。
- DistilBERT 模型没有选择输入位置的选项（`position_ids` 输入）。如果需要，可以添加该选项，只需让我们知道你是否需要。
- 与 BERT 相同但更小。通过对预训练的 BERT 模型进行蒸馏训练，这意味着它被训练为预测与较大模型相同的概率。实际目标是：

    * 提供与教师模型相同的概率
    * 正确预测被掩码的标记（但没有下一个句子的任务）
    * 学生模型的隐藏状态与教师模型的隐藏状态之间的余弦相似度

此模型是由 [victorsanh](https://huggingface.co/victorsanh) 贡献的。此模型 jax 版本由 [kamalkraj](https://huggingface.co/kamalkraj) 贡献的。原始代码可以在 [这里](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) 找到。

## 资源

以下是官方 Hugging Face 和社区（由 🌎 标记）资源列表，可帮助你开始使用 DistilBERT。如果你有兴趣提交资源以包含在此处，请随时提出拉取请求，我们将进行审核！资源最好应该展示一些新的东西，而不是重复现有的资源。

<PipelineTag pipeline="text-classification"/>

- 一篇关于使用 Python 进行情感分析的博客文章 [Getting Started with Sentiment Analysis using Python](https://huggingface.co/blog/sentiment-analysis-python)。
- 一篇关于如何使用 Blurr 进行 DistilBERT 序列分类训练的博客文章 [train DistilBERT with Blurr for sequence classification](https://huggingface.co/blog/fastai)。
- 一篇关于如何使用 Ray 来调整 DistilBERT 超参数的博客文章 [train DistilBERT with Ray for hyperparameter tuning](https://huggingface.co/blog/ray-tune)。
- 一篇关于如何使用 Hugging Face 和 Amazon SageMaker 来训练 DistilBERT 的博客文章 [train DistilBERT with Hugging Face and Amazon SageMaker](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face)。
- 一篇关于如何对多标签分类进行 DistilBERT 微调的笔记本 [finetune DistilBERT for multi-label classification](https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb)。🌎
- 一篇关于如何使用 PyTorch 对多类别分类进行 DistilBERT 微调的笔记本 [finetune DistilBERT for multiclass classification with PyTorch](https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multiclass_classification.ipynb)。🌎
- 一篇关于如何使用 TensorFlow 对文本分类进行 DistilBERT 微调的笔记本 [finetune DistilBERT for text classification in TensorFlow](https://colab.research.google.com/github/peterbayerle/huggingface_notebook/blob/main/distilbert_tf.ipynb)。🌎
- [`DistilBertForSequenceClassification`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb) 中找到。
- [`TFDistilBertForSequenceClassification`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb) 中找到。
- [`FlaxDistilBertForSequenceClassification`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb) 中找到。
- [序列分类任务指南](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [`DistilBertForTokenClassification`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb) 中找到。
- [`TFDistilBertForTokenClassification`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb) 中找到。
- [`FlaxDistilBertForTokenClassification`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification) 中找到。
- [🤗 Hugging Face 课程中的标记分类](https://huggingface.co/course/chapter7/2?fw=pt) 章节。
- [标记分类任务指南](../tasks/token_classification)

<PipelineTag pipeline="fill-mask"/>

- [`DistilBertForMaskedLM`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) 中找到。
- [`TFDistilBertForMaskedLM`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb) 中找到。
- [`FlaxDistilBertForMaskedLM`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb) 中找到。
- [🤗 Hugging Face 课程中的掩码语言建模](https://huggingface.co/course/chapter7/3?fw=pt) 章节。
- [掩码语言建模任务指南](../tasks/masked_language_modeling)

<PipelineTag pipeline="question-answering"/>

- [`DistilBertForQuestionAnswering`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) 中找到。
- [`TFDistilBertForQuestionAnswering`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb) 中找到。
- [`FlaxDistilBertForQuestionAnswering`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering) 中找到。
- [🤗 Hugging Face 课程中的问答](https://huggingface.co/course/chapter7/7?fw=pt) 章节。
- [问答任务指南](../tasks/question_answering)

**多选题**
- [`DistilBertForMultipleChoice`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb) 中找到。
- [`TFDistilBertForMultipleChoice`] 的示例脚本可在此 [链接](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb) 中找到。
- [多选题任务指南](../tasks/multiple_choice)

⚗️ 优化

- 一篇关于如何利用 🤗 Optimum 和 Intel 对 DistilBERT 进行量化的博客文章 [quantize DistilBERT with 🤗 Optimum and Intel](https://huggingface.co/blog/intel)。
- 一篇关于如何使用 🤗 Optimum 优化 GPU 上的 Transformers 的博客文章 [Optimizing Transformers for GPUs with 🤗 Optimum](https://www.philschmid.de/optimizing-transformers-with-optimum-gpu)。
- 一篇关于使用 Hugging Face Optimum 优化 Transformers 的博客文章 [Optimizing Transformers with Hugging Face Optimum](https://www.philschmid.de/optimizing-transformers-with-optimum)。

⚡️ 推理

- 一篇关于如何使用 Hugging Face Transformers 和 AWS Inferentia 加速 BERT 推理的博客文章 [Accelerate BERT inference with Hugging Face Transformers and AWS Inferentia](https://huggingface.co/blog/bert-inferentia-sagemaker)，使用 DistilBERT。
- 一篇关于 [使用 Hugging Face 的 Transformers、DistilBERT 和 Amazon SageMaker 进行无服务器推理](https://www.philschmid.de/sagemaker-serverless-huggingface-distilbert) 的博客文章。

🚀 部署

- 一篇关于如何在 Google Cloud 上部署 DistilBERT 的博客文章 [deploy DistilBERT on Google Cloud](https://huggingface.co/blog/how-to-deploy-a-pipeline-to-google-clouds)。
- 一篇关于如何使用 Amazon SageMaker 部署 DistilBERT 的博客文章 [deploy DistilBERT with Amazon SageMaker](https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker)。
- 一篇关于如何使用 Hugging Face Transformers、Amazon SageMaker 和 Terraform 模块部署 BERT 的博客文章 [Deploy BERT with Hugging Face Transformers, Amazon SageMaker and Terraform module](https://www.philschmid.de/terraform-huggingface-amazon-sagemaker)。

## DistilBertConfig

[[autodoc]] DistilBertConfig

## DistilBertTokenizer

[[autodoc]] DistilBertTokenizer

## DistilBertTokenizerFast

[[autodoc]] DistilBertTokenizerFast

## DistilBertModel

[[autodoc]] DistilBertModel
    - forward

## DistilBertForMaskedLM

[[autodoc]] DistilBertForMaskedLM
    - forward

## DistilBertForSequenceClassification

[[autodoc]] DistilBertForSequenceClassification
    - forward

## DistilBertForMultipleChoice

[[autodoc]] DistilBertForMultipleChoice
    - forward

## DistilBertForTokenClassification

[[autodoc]] DistilBertForTokenClassification
    - forward

## DistilBertForQuestionAnswering

[[autodoc]] DistilBertForQuestionAnswering
    - forward

## TFDistilBertModel

[[autodoc]] TFDistilBertModel
    - call

## TFDistilBertForMaskedLM

[[autodoc]] TFDistilBertForMaskedLM
    - call

## TFDistilBertForSequenceClassification

[[autodoc]] TFDistilBertForSequenceClassification
    - call

## TFDistilBertForMultipleChoice

[[autodoc]] TFDistilBertForMultipleChoice
    - call

## TFDistilBertForTokenClassification

[[autodoc]] TFDistilBertForTokenClassification
    - call

## TFDistilBertForQuestionAnswering

[[autodoc]] TFDistilBertForQuestionAnswering
    - call

## FlaxDistilBertModel

[[autodoc]] FlaxDistilBertModel
    - __call__

## FlaxDistilBertForMaskedLM

[[autodoc]] FlaxDistilBertForMaskedLM
    - __call__

## FlaxDistilBertForSequenceClassification

[[autodoc]] FlaxDistilBertForSequenceClassification
    - __call__

## FlaxDistilBertForMultipleChoice

[[autodoc]] FlaxDistilBertForMultipleChoice
    - __call__

## FlaxDistilBertForTokenClassification

[[autodoc]] FlaxDistilBertForTokenClassification
    - __call__

## FlaxDistilBertForQuestionAnswering

[[autodoc]] FlaxDistilBertForQuestionAnswering
    - __call__