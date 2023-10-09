<!--版权所有2020年The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（"许可证"）授权；除非你遵守许可证的规定，否则你不得使用此文件。

你可以在以下位置获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按"原样"分发的，不附带任何明示或暗示的条件或担保。请参阅许可证以了解在许可证下的特定语言和限制。

⚠️请注意，此文件是Markdown格式，但包含特定于我们的文档生成器（类似MDX）的语法，可能无法在你的Markdown查看器中正确显示。

-->

# BART

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=bart">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-bart-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/bart-large-mnli">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**免责声明：**如果你发现了任何奇怪的地方，请提一个[Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)，并分配给@patrickvonplaten

## 概述

Bart模型是由Mike Lewis、Yinhan Liu、Naman Goyal、Marjan Ghazvininejad、Abdelrahman Mohamed、Omer Levy、Ves Stoyanov和Luke Zettlemoyer于2019年10月29日提出的[BART：用于自然语言生成、翻译和理解的去噪序列到序列预训练](https://arxiv.org/abs/1910.13461)。

根据摘要，

- Bart使用标准的seq2seq/机器翻译架构，具有双向编码器（类似BERT）和从左到右的解码器（类似GPT）。
- 预训练任务包括随机打乱原始句子的顺序和一种新颖的填充方案，其中文本的一段被替换为单个掩码标记。
- Bart在进行文本生成的精调时特别有效，但也适用于理解任务。它与GLUE和SQuAD上使用相似的训练资源的RoBERTa的性能相匹配，在一系列摘要对话、问答和总结任务中取得了最新的最优结果，最高ROUGE提高了6个百分点。

提示：

- Bart是一个具有绝对位置嵌入的模型，因此通常建议将输入在右侧而不是左侧进行填充。
- Seq2seq模型，由编码器和解码器组成。编码器接收到一个被损坏的令牌版本，解码器接收原始令牌（但有一个遮罩来隐藏未来的单词，就像常规的transformers解码器一样）。对于编码器的预训练任务，应用以下转换的组合：

  * 随机掩码令牌（类似于BERT）
  * 删除随机令牌
  * 使用单个掩码令牌掩盖k个令牌的一段（0个令牌的一段是插入一个掩码令牌）
  * 排列句子
  * 旋转文档，使其从特定令牌开始

此模型由[sshleifer](https://huggingface.co/sshleifer)贡献。作者的代码在[这里](https://github.com/pytorch/fairseq/tree/master/examples/bart)。

### 示例

- 可以在[examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md)中找到有关微调BART和其他序列到序列任务的示例和脚本。
- 可以在这个[论坛讨论](https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904)中找到如何使用Hugging Face的`datasets`对象训练[`BartForConditionalGeneration`]的示例。
- 这个[paper](https://arxiv.org/abs/2010.13002)中描述了[蒸馏检查点](https://huggingface.co/models?search=distilbart)。

## 实现注意事项

- Bart不使用`token_type_ids`进行序列分类。请使用[`BartTokenizer`]或[`~BartTokenizer.encode`]完成正确的拆分。
- 如果未传递`decoder_input_ids`，[`BartModel`]的前向传递将创建它们。这与某些其他建模API不同。此功能的一个典型用例是填充遮罩。
- 当`forced_bos_token_id=0`时，模型预测意图与原始实现相同。然而，这仅在你传递给[`fairseq.encode`]的字符串以空格开头时有效。
- 应该使用[`~generation.GenerationMixin.generate`]进行条件生成任务，例如总结，有关示例，请参阅该文档字符串。
- 加载*facebook/bart-large-cnn*权重的模型将不具有`mask_token_id`，也无法执行填充掩码的任务。

## 填充掩码

`facebook/bart-base` 和 `facebook/bart-large` 检查点可用于填充包含多个令牌的掩码。

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-large")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"])
assert tok.batch_decode(generated_ids, skip_special_tokens=True) == [
    "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
]
```

## 资源

下面是Hugging Face官方和社区（🌎）的一些资源，可以帮助你入门BART。如果你有兴趣提交一个资源并包含在这里，请随时打开一个Pull Request，我们将检查它！该资源应该理想地展示一些新的东西，而不是重复现有的资源。

<PipelineTag pipeline="summarization"/>

- 有关[分布式训练：使用🤗 Transformers和Amazon SageMaker训练BART/T5进行摘要](https://huggingface.co/blog/sagemaker-distributed-training-seq2seq)的博客文章。
- 有关如何使用fastai和blurr对BART进行[总结微调的笔记本](https://colab.research.google.com/github/ohmeow/ohmeow_website/blob/master/posts/2021-05-25-mbart-sequence-classification-with-blurr.ipynb)。🌎
- 有关如何使用Trainer类在两种语言中对BART进行[总结微调的笔记本](https://colab.research.google.com/github/elsanns/xai-nlp-notebooks/blob/master/fine_tune_bart_summarization_two_langs.ipynb)。🌎
- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)支持[`BartForConditionalGeneration`]。
- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb)支持[`TFBartForConditionalGeneration`]。
- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/summarization)支持[`FlaxBartForConditionalGeneration`]。
- 🤗 Hugging Face课程的[摘要](https://huggingface.co/course/chapter7/5?fw=pt#summarization)章节。
- [摘要任务指南](../tasks/summarization)

<PipelineTag pipeline="fill-mask"/>

- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)支持[`BartForConditionalGeneration`]。
- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)支持[`TFBartForConditionalGeneration`]。
- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb)支持[`FlaxBartForConditionalGeneration`]。
- 🤗 Hugging Face课程的[掩码语言模型](https://huggingface.co/course/chapter7/3?fw=pt)章节。
- [掩码语言模型任务指南](../tasks/masked_language_modeling)

<PipelineTag pipeline="translation"/>

- 如何使用Seq2SeqTrainer对mBART进行[Hindi到English翻译的摘要微调的笔记本](https://colab.research.google.com/github/vasudevgupta7/huggingface-tutorials/blob/main/translation_training.ipynb)。🌎
- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)支持[`BartForConditionalGeneration`]。
- 通过这个[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/translation)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb)支持[`TFBartForConditionalGeneration`]。
- [翻译任务指南](../tasks/translation)

另请参阅：
- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)

## BartConfig

[[autodoc]] BartConfig
    - all

## BartTokenizer

[[autodoc]] BartTokenizer
    - all

## BartTokenizerFast

[[autodoc]] BartTokenizerFast
    - all

## BartModel

[[autodoc]] BartModel
    - forward

## BartForConditionalGeneration

[[autodoc]] BartForConditionalGeneration
    - forward

## BartForSequenceClassification

[[autodoc]] BartForSequenceClassification
    - forward

## BartForQuestionAnswering

[[autodoc]] BartForQuestionAnswering
    - forward

## BartForCausalLM

[[autodoc]] BartForCausalLM
    - forward

## TFBartModel

[[autodoc]] TFBartModel
    - call

## TFBartForConditionalGeneration

[[autodoc]] TFBartForConditionalGeneration
    - call

## TFBartForSequenceClassification

[[autodoc]] TFBartForSequenceClassification
    - call

## FlaxBartModel

[[autodoc]] FlaxBartModel
    - __call__
    - encode
    - decode

## FlaxBartForConditionalGeneration

[[autodoc]] FlaxBartForConditionalGeneration
    - __call__
    - encode
    - decode

## FlaxBartForSequenceClassification

[[autodoc]] FlaxBartForSequenceClassification
    - __call__
    - encode
    - decode

## FlaxBartForQuestionAnswering

[[autodoc]] FlaxBartForQuestionAnswering
    - __call__
    - encode
    - decode

## FlaxBartForCausalLM

[[autodoc]] FlaxBartForCausalLM
    - __call__