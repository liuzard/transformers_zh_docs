<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Whisper

## Overview

The Whisper model was proposed in [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf) by Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever.

The abstract from the paper is the following:

*We study the capabilities of speech processing systems trained simply to predict large amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual and multitask supervision, the resulting models generalize well to standard benchmarks and are often competitive with prior fully supervised results but in a zeroshot transfer setting without the need for any finetuning. When compared to humans, the models approach their accuracy and robustness. We are releasing models and inference code to serve as a foundation for further work on robust speech processing.*


Tips:

- The model usually performs well without requiring any finetuning.
- The architecture follows a classic encoder-decoder architecture, which means that it relies on the [`~generation.GenerationMixin.generate`] function for inference.
- Inference is currently only implemented for short-form i.e. audio is pre-segmented into <=30s segments. Long-form (including timestamps) will be implemented in a future release.
- One can use [`WhisperProcessor`] to prepare audio for the model, and decode the predicted ID's back into text.

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ). The Tensorflow version of this model was contributed by [amyeroberts](https://huggingface.co/amyeroberts).
The original code can be found [here](https://github.com/openai/whisper).


## WhisperConfig

[[autodoc]] WhisperConfig

## WhisperTokenizer

[[autodoc]] WhisperTokenizer
    - set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## WhisperTokenizerFast

[[autodoc]] WhisperTokenizerFast
    - set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## WhisperFeatureExtractor

[[autodoc]] WhisperFeatureExtractor
    - __call__

## WhisperProcessor

[[autodoc]] WhisperProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## WhisperModel

[[autodoc]] WhisperModel
    - forward
    - _mask_input_features

## WhisperForConditionalGeneration

[[autodoc]] WhisperForConditionalGeneration
    - forward

## WhisperForAudioClassification

[[autodoc]] WhisperForAudioClassification
    - forward


## TFWhisperModel

[[autodoc]] TFWhisperModel
    - call

## TFWhisperForConditionalGeneration

[[autodoc]] TFWhisperForConditionalGeneration
    - call


## FlaxWhisperModel

[[autodoc]] FlaxWhisperModel
    - __call__

## FlaxWhisperForConditionalGeneration

[[autodoc]] FlaxWhisperForConditionalGeneration
    - __call__

## FlaxWhisperForAudioClassification

[[autodoc]] FlaxWhisperForAudioClassification
    - __call__

