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

# Models

The base classes [`PreTrainedModel`], [`TFPreTrainedModel`], and
[`FlaxPreTrainedModel`] implement the common methods for loading/saving a model either from a local
file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace's AWS
S3 repository).

[`PreTrainedModel`] and [`TFPreTrainedModel`] also implement a few methods which
are common among all the models to:

- resize the input token embeddings when new tokens are added to the vocabulary
- prune the attention heads of the model.

The other methods that are common to each model are defined in [`~modeling_utils.ModuleUtilsMixin`]
(for the PyTorch models) and [`~modeling_tf_utils.TFModuleUtilsMixin`] (for the TensorFlow models) or
for text generation, [`~generation.GenerationMixin`] (for the PyTorch models),
[`~generation.TFGenerationMixin`] (for the TensorFlow models) and
[`~generation.FlaxGenerationMixin`] (for the Flax/JAX models).


## PreTrainedModel

[[autodoc]] PreTrainedModel
    - push_to_hub
    - all

<a id='from_pretrained-torch-dtype'></a>

### Large model loading

In Transformers 4.20.0, the [`~PreTrainedModel.from_pretrained`] method has been reworked to accommodate large models using [Accelerate](https://huggingface.co/docs/accelerate/big_modeling). This requires Accelerate >= 0.9.0 and PyTorch >= 1.9.0. Instead of creating the full model, then loading the pretrained weights inside it (which takes twice the size of the model in RAM, one for the randomly initialized model, one for the weights), there is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded.

This option can be activated with `low_cpu_mem_usage=True`. The model is first created on the Meta device (with empty weights) and the state dict is then loaded inside it (shard by shard in the case of a sharded checkpoint). This way the maximum RAM used is the full size of the model only.

```py
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", low_cpu_mem_usage=True)
```

Moreover, you can directly place the model on different devices if it doesn't fully fit in RAM (only works for inference for now). With `device_map="auto"`, Accelerate will determine where to put each layer to maximize the use of your fastest devices (GPUs) and offload the rest on the CPU, or even the hard drive if you don't have enough GPU RAM (or CPU RAM). Even if the model is split across several devices, it will run as you would normally expect.

When passing a `device_map`, `low_cpu_mem_usage` is automatically set to `True`, so you don't need to specify it:

```py
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

You can inspect how the model was split across devices by looking at its `hf_device_map` attribute:

```py
t0pp.hf_device_map
```

```python out
{'shared': 0,
 'decoder.embed_tokens': 0,
 'encoder': 0,
 'decoder.block.0': 0,
 'decoder.block.1': 1,
 'decoder.block.2': 1,
 'decoder.block.3': 1,
 'decoder.block.4': 1,
 'decoder.block.5': 1,
 'decoder.block.6': 1,
 'decoder.block.7': 1,
 'decoder.block.8': 1,
 'decoder.block.9': 1,
 'decoder.block.10': 1,
 'decoder.block.11': 1,
 'decoder.block.12': 1,
 'decoder.block.13': 1,
 'decoder.block.14': 1,
 'decoder.block.15': 1,
 'decoder.block.16': 1,
 'decoder.block.17': 1,
 'decoder.block.18': 1,
 'decoder.block.19': 1,
 'decoder.block.20': 1,
 'decoder.block.21': 1,
 'decoder.block.22': 'cpu',
 'decoder.block.23': 'cpu',
 'decoder.final_layer_norm': 'cpu',
 'decoder.dropout': 'cpu',
 'lm_head': 'cpu'}
```

You can also write your own device map following the same format (a dictionary layer name to device). It should map all parameters of the model to a given device, but you don't have to detail where all the submodules of one layer go if that layer is entirely on the same device. For instance, the following device map would work properly for T0pp (as long as you have the GPU memory):

```python
device_map = {"shared": 0, "encoder": 0, "decoder": 1, "lm_head": 1}
```

Another way to minimize the memory impact of your model is to instantiate it at a lower precision dtype (like `torch.float16`) or use direct quantization techniques as described below.

### Model Instantiation dtype

Under Pytorch a model normally gets instantiated with `torch.float32` format. This can be an issue if one tries to
load a model whose weights are in fp16, since it'd require twice as much memory. To overcome this limitation, you can
either explicitly pass the desired `dtype` using `torch_dtype` argument:

```python
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype=torch.float16)
```

or, if you want the model to always load in the most optimal memory pattern, you can use the special value `"auto"`,
and then `dtype` will be automatically derived from the model's weights:

```python
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype="auto")
```

Models instantiated from scratch can also be told which `dtype` to use with:

```python
config = T5Config.from_pretrained("t5")
model = AutoModel.from_config(config)
```

Due to Pytorch design, this functionality is only available for floating dtypes.


## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

[[autodoc]] TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

[[autodoc]] FlaxPreTrainedModel
    - push_to_hub
    - all

## Pushing to the Hub

[[autodoc]] utils.PushToHubMixin

## Sharded checkpoints

[[autodoc]] modeling_utils.load_sharded_checkpoint
