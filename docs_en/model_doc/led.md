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

# LED

## Overview

The LED model was proposed in [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz
Beltagy, Matthew E. Peters, Arman Cohan.

The abstract from the paper is the following:

*Transformer-based models are unable to process long sequences due to their self-attention operation, which scales
quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention
mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or
longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local
windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we
evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In
contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our
pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on
WikiHop and TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting
long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization
dataset.*

Tips:

- [`LEDForConditionalGeneration`] is an extension of
  [`BartForConditionalGeneration`] exchanging the traditional *self-attention* layer with
  *Longformer*'s *chunked self-attention* layer. [`LEDTokenizer`] is an alias of
  [`BartTokenizer`].
- LED works very well on long-range *sequence-to-sequence* tasks where the `input_ids` largely exceed a length of
  1024 tokens.
- LED pads the `input_ids` to be a multiple of `config.attention_window` if required. Therefore a small speed-up is
  gained, when [`LEDTokenizer`] is used with the `pad_to_multiple_of` argument.
- LED makes use of *global attention* by means of the `global_attention_mask` (see
  [`LongformerModel`]). For summarization, it is advised to put *global attention* only on the first
  `<s>` token. For question answering, it is advised to put *global attention* on all tokens of the question.
- To fine-tune LED on all 16384, *gradient checkpointing* can be enabled in case training leads to out-of-memory (OOM)
  errors. This can be done by executing `model.gradient_checkpointing_enable()`. 
 Moreover, the `use_cache=False`
  flag can be used to disable the caching mechanism to save memory.
- A notebook showing how to evaluate LED, can be accessed [here](https://colab.research.google.com/drive/12INTTR6n64TzS4RrXZxMSXfrOd9Xzamo?usp=sharing).
- A notebook showing how to fine-tune LED, can be accessed [here](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing).
- LED is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Question answering task guide](../tasks/question_answering)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## LEDConfig

[[autodoc]] LEDConfig

## LEDTokenizer

[[autodoc]] LEDTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LEDTokenizerFast

[[autodoc]] LEDTokenizerFast

## LED specific outputs

[[autodoc]] models.led.modeling_led.LEDEncoderBaseModelOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqModelOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqLMOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput

[[autodoc]] models.led.modeling_tf_led.TFLEDEncoderBaseModelOutput

[[autodoc]] models.led.modeling_tf_led.TFLEDSeq2SeqModelOutput

[[autodoc]] models.led.modeling_tf_led.TFLEDSeq2SeqLMOutput

## LEDModel

[[autodoc]] LEDModel
    - forward

## LEDForConditionalGeneration

[[autodoc]] LEDForConditionalGeneration
    - forward

## LEDForSequenceClassification

[[autodoc]] LEDForSequenceClassification
    - forward

## LEDForQuestionAnswering

[[autodoc]] LEDForQuestionAnswering
    - forward

## TFLEDModel

[[autodoc]] TFLEDModel
    - call

## TFLEDForConditionalGeneration

[[autodoc]] TFLEDForConditionalGeneration
    - call
