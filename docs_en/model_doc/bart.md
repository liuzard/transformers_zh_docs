<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

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

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign
@patrickvonplaten

## Overview

The Bart model was proposed in [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
Translation, and Comprehension](https://arxiv.org/abs/1910.13461) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan
Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.

According to the abstract,

- Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a
  left-to-right decoder (like GPT).
- The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme,
  where spans of text are replaced with a single mask token.
- BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It
  matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new
  state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains
  of up to 6 ROUGE.

Tips:

- BART is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- Sequence-to-sequence model with an encoder and a decoder. Encoder is fed a corrupted version of the tokens, decoder is fed the original tokens (but has a mask to hide the future words like a regular transformers decoder). A composition of the following transformations are applied on the pretraining tasks for the encoder:

  * mask random tokens (like in BERT)
  * delete random tokens
  * mask a span of k tokens with a single mask token (a span of 0 tokens is an insertion of a mask token)
  * permute sentences
  * rotate the document to make it start at a specific token

This model was contributed by [sshleifer](https://huggingface.co/sshleifer). The Authors' code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/bart).


### Examples

- Examples and scripts for fine-tuning BART and other models for sequence to sequence tasks can be found in
  [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md).
- An example of how to train [`BartForConditionalGeneration`] with a Hugging Face `datasets`
  object can be found in this [forum discussion](https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904).
- [Distilled checkpoints](https://huggingface.co/models?search=distilbart) are described in this [paper](https://arxiv.org/abs/2010.13002).


## Implementation Notes

- Bart doesn't use `token_type_ids` for sequence classification. Use [`BartTokenizer`] or
  [`~BartTokenizer.encode`] to get the proper splitting.
- The forward pass of [`BartModel`] will create the `decoder_input_ids` if they are not passed.
  This is different than some other modeling APIs. A typical use case of this feature is mask filling.
- Model predictions are intended to be identical to the original implementation when
  `forced_bos_token_id=0`. This only works, however, if the string you pass to
  [`fairseq.encode`] starts with a space.
- [`~generation.GenerationMixin.generate`] should be used for conditional generation tasks like
  summarization, see the example in that docstrings.
- Models that load the *facebook/bart-large-cnn* weights will not have a `mask_token_id`, or be able to perform
  mask-filling tasks.

## Mask Filling

The `facebook/bart-base` and `facebook/bart-large` checkpoints can be used to fill multi-token masks.

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

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BART. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="summarization"/>

- A blog post on [Distributed Training: Train BART/T5 for Summarization using 🤗 Transformers and Amazon SageMaker](https://huggingface.co/blog/sagemaker-distributed-training-seq2seq).
- A notebook on how to [finetune BART for summarization with fastai using blurr](https://colab.research.google.com/github/ohmeow/ohmeow_website/blob/master/posts/2021-05-25-mbart-sequence-classification-with-blurr.ipynb). 🌎
- A notebook on how to [finetune BART for summarization in two languages with Trainer class](https://colab.research.google.com/github/elsanns/xai-nlp-notebooks/blob/master/fine_tune_bart_summarization_two_langs.ipynb). 🌎
- [`BartForConditionalGeneration`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb).
- [`TFBartForConditionalGeneration`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb).
- [`FlaxBartForConditionalGeneration`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/summarization).
- [Summarization](https://huggingface.co/course/chapter7/5?fw=pt#summarization) chapter of the 🤗 Hugging Face course.
- [Summarization task guide](../tasks/summarization)

<PipelineTag pipeline="fill-mask"/>

- [`BartForConditionalGeneration`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFBartForConditionalGeneration`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxBartForConditionalGeneration`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).
- [Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt) chapter of the 🤗 Hugging Face Course.
- [Masked language modeling task guide](../tasks/masked_language_modeling)

<PipelineTag pipeline="translation"/>

- A notebook on how to [finetune mBART using Seq2SeqTrainer for Hindi to English translation](https://colab.research.google.com/github/vasudevgupta7/huggingface-tutorials/blob/main/translation_training.ipynb). 🌎
- [`BartForConditionalGeneration`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb).
- [`TFBartForConditionalGeneration`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/translation) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb).
- [Translation task guide](../tasks/translation)

See also:
- [Text classification task guide](../tasks/sequence_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)

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
