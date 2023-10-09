<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GPT Neo

## Overview

The GPTNeo model was released in the [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) repository by Sid
Black, Stella Biderman, Leo Gao, Phil Wang and Connor Leahy. It is a GPT2 like causal language model trained on the
[Pile](https://pile.eleuther.ai/) dataset.

The architecture is similar to GPT2 except that GPT Neo uses local attention in every other layer with a window size of
256 tokens.

This model was contributed by [valhalla](https://huggingface.co/valhalla).

### Generation

The `generate()` method can be used to generate text using GPT Neo model.

```python
>>> from transformers import GPTNeoForCausalLM, GPT2Tokenizer

>>> model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
>>> tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

>>> prompt = (
...     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
...     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
...     "researchers was the fact that the unicorns spoke perfect English."
... )

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Causal language modeling task guide](../tasks/language_modeling)

## GPTNeoConfig

[[autodoc]] GPTNeoConfig

## GPTNeoModel

[[autodoc]] GPTNeoModel
    - forward

## GPTNeoForCausalLM

[[autodoc]] GPTNeoForCausalLM
    - forward

## GPTNeoForQuestionAnswering

[[autodoc]] GPTNeoForQuestionAnswering
    - forward

## GPTNeoForSequenceClassification

[[autodoc]] GPTNeoForSequenceClassification
    - forward

## GPTNeoForTokenClassification

[[autodoc]] GPTNeoForTokenClassification
    - forward

## FlaxGPTNeoModel

[[autodoc]] FlaxGPTNeoModel
    - __call__

## FlaxGPTNeoForCausalLM

[[autodoc]] FlaxGPTNeoForCausalLM
    - __call__
