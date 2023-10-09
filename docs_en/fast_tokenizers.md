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

# Use tokenizers from 🤗 Tokenizers

The [`PreTrainedTokenizerFast`] depends on the [🤗 Tokenizers](https://huggingface.co/docs/tokenizers) library. The tokenizers obtained from the 🤗 Tokenizers library can be
loaded very simply into 🤗 Transformers.

Before getting in the specifics, let's first start by creating a dummy tokenizer in a few lines:

```python
>>> from tokenizers import Tokenizer
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import BpeTrainer
>>> from tokenizers.pre_tokenizers import Whitespace

>>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
>>> trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

>>> tokenizer.pre_tokenizer = Whitespace()
>>> files = [...]
>>> tokenizer.train(files, trainer)
```

We now have a tokenizer trained on the files we defined. We can either continue using it in that runtime, or save it to
a JSON file for future re-use.

## Loading directly from the tokenizer object

Let's see how to leverage this tokenizer object in the 🤗 Transformers library. The
[`PreTrainedTokenizerFast`] class allows for easy instantiation, by accepting the instantiated
*tokenizer* object as an argument:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

This object can now be used with all the methods shared by the 🤗 Transformers tokenizers! Head to [the tokenizer
page](main_classes/tokenizer) for more information.

## Loading from a JSON file

In order to load a tokenizer from a JSON file, let's first start by saving our tokenizer:

```python
>>> tokenizer.save("tokenizer.json")
```

The path to which we saved this file can be passed to the [`PreTrainedTokenizerFast`] initialization
method using the `tokenizer_file` parameter:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

This object can now be used with all the methods shared by the 🤗 Transformers tokenizers! Head to [the tokenizer
page](main_classes/tokenizer) for more information.
