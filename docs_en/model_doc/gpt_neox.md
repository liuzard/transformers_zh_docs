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

# GPT-NeoX

## Overview

We introduce GPT-NeoX-20B, a 20 billion parameter autoregressive language model trained on the Pile, whose weights will
be made freely and openly available to the public through a permissive license. It is, to the best of our knowledge,
the largest dense autoregressive model that has publicly available weights at the time of submission. In this work,
we describe GPT-NeoX-20B's architecture and training and evaluate its performance on a range of language-understanding,
mathematics, and knowledge-based tasks. We find that GPT-NeoX-20B is a particularly powerful few-shot reasoner and
gains far more in performance when evaluated five-shot than similarly sized GPT-3 and FairSeq models. We open-source
the training and evaluation code, as well as the model weights, at [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox).

Development of the model was led by Sid Black, Stella Biderman and Eric Hallahan, and the model was trained with
generous the support of [CoreWeave](https://www.coreweave.com/).

GPT-NeoX-20B was trained with fp16, thus it is recommended to initialize the model as follows:

```python
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half().cuda()
```

GPT-NeoX-20B also has a different tokenizer from the one used in GPT-J-6B and GPT-Neo. The new tokenizer allocates
additional tokens to whitespace characters, making the model more suitable for certain tasks like code generation.

### Generation

The `generate()` method can be used to generate text using GPT Neo model.

```python
>>> from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

>>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
>>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

>>> prompt = "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI."

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

- [Causal language modeling task guide](../tasks/language_modeling)

## GPTNeoXConfig

[[autodoc]] GPTNeoXConfig

## GPTNeoXTokenizerFast

[[autodoc]] GPTNeoXTokenizerFast

## GPTNeoXModel

[[autodoc]] GPTNeoXModel
    - forward

## GPTNeoXForCausalLM

[[autodoc]] GPTNeoXForCausalLM
    - forward

## GPTNeoXForQuestionAnswering

[[autodoc]] GPTNeoXForQuestionAnswering
    - forward

## GPTNeoXForSequenceClassification

[[autodoc]] GPTNeoXForSequenceClassification
    - forward

## GPTNeoXForTokenClassification

[[autodoc]] GPTNeoXForTokenClassification
    - forward
