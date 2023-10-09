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

# Sharing custom models

The 🤗 Transformers library is designed to be easily extensible. Every model is fully coded in a given subfolder
of the repository with no abstraction, so you can easily copy a modeling file and tweak it to your needs.

If you are writing a brand new model, it might be easier to start from scratch. In this tutorial, we will show you
how to write a custom model and its configuration so it can be used inside Transformers, and how you can share it
with the community (with the code it relies on) so that anyone can use it, even if it's not present in the 🤗
Transformers library.

We will illustrate all of this on a ResNet model, by wrapping the ResNet class of the
[timm library](https://github.com/rwightman/pytorch-image-models) into a [`PreTrainedModel`].

## Writing a custom configuration

Before we dive into the model, let's first write its configuration. The configuration of a model is an object that
will contain all the necessary information to build the model. As we will see in the next section, the model can only
take a `config` to be initialized, so we really need that object to be as complete as possible.

In our example, we will take a couple of arguments of the ResNet class that we might want to tweak. Different
configurations will then give us the different types of ResNets that are possible. We then just store those arguments,
after checking the validity of a few of them.

```python
from transformers import PretrainedConfig
from typing import List


class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

The three important things to remember when writing you own configuration are the following:
- you have to inherit from `PretrainedConfig`,
- the `__init__` of your `PretrainedConfig` must accept any kwargs,
- those `kwargs` need to be passed to the superclass `__init__`.

The inheritance is to make sure you get all the functionality from the 🤗 Transformers library, while the two other
constraints come from the fact a `PretrainedConfig` has more fields than the ones you are setting. When reloading a
config with the `from_pretrained` method, those fields need to be accepted by your config and then sent to the
superclass.

Defining a `model_type` for your configuration (here `model_type="resnet"`) is not mandatory, unless you want to
register your model with the auto classes (see last section).

With this done, you can easily create and save your configuration like you would do with any other model config of the
library. Here is how we can create a resnet50d config and save it:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

This will save a file named `config.json` inside the folder `custom-resnet`. You can then reload your config with the
`from_pretrained` method:

```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

You can also use any other method of the [`PretrainedConfig`] class, like [`~PretrainedConfig.push_to_hub`] to
directly upload your config to the Hub.

## Writing a custom model

Now that we have our ResNet configuration, we can go on writing the model. We will actually write two: one that
extracts the hidden features from a batch of images (like [`BertModel`]) and one that is suitable for image
classification (like [`BertForSequenceClassification`]).

As we mentioned before, we'll only write a loose wrapper of the model to keep it simple for this example. The only
thing we need to do before writing this class is a map between the block types and actual block classes. Then the
model is defined from the configuration by passing everything to the `ResNet` class:

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig


BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}


class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

For the model that will classify images, we just change the forward method:

```py
import torch


class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

In both cases, notice how we inherit from `PreTrainedModel` and call the superclass initialization with the `config`
(a bit like when you write a regular `torch.nn.Module`). The line that sets the `config_class` is not mandatory, unless
you want to register your model with the auto classes (see last section).

<Tip>

If your model is very similar to a model inside the library, you can re-use the same configuration as this model.

</Tip>

You can have your model return anything you want, but returning a dictionary like we did for
`ResnetModelForImageClassification`, with the loss included when labels are passed, will make your model directly
usable inside the [`Trainer`] class. Using another output format is fine as long as you are planning on using your own
training loop or another library for training.

Now that we have our model class, let's create one:

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

Again, you can use any of the methods of [`PreTrainedModel`], like [`~PreTrainedModel.save_pretrained`] or
[`~PreTrainedModel.push_to_hub`]. We will use the second in the next section, and see how to push the model weights
with the code of our model. But first, let's load some pretrained weights inside our model.

In your own use case, you will probably be training your custom model on your own data. To go fast for this tutorial,
we will use the pretrained version of the resnet50d. Since our model is just a wrapper around it, it's going to be
easy to transfer those weights:

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

Now let's see how to make sure that when we do [`~PreTrainedModel.save_pretrained`] or [`~PreTrainedModel.push_to_hub`], the
code of the model is saved.

## Sending the code to the Hub

<Tip warning={true}>

This API is experimental and may have some slight breaking changes in the next releases.

</Tip>

First, make sure your model is fully defined in a `.py` file. It can rely on relative imports to some other files as
long as all the files are in the same directory (we don't support submodules for this feature yet). For our example,
we'll define a `modeling_resnet.py` file and a `configuration_resnet.py` file in a folder of the current working
directory named `resnet_model`. The configuration file contains the code for `ResnetConfig` and the modeling file
contains the code of `ResnetModel` and `ResnetModelForImageClassification`.

```
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

The `__init__.py` can be empty, it's just there so that Python detects `resnet_model` can be use as a module.

<Tip warning={true}>

If copying a modeling files from the library, you will need to replace all the relative imports at the top of the file
to import from the `transformers` package.

</Tip>

Note that you can re-use (or subclass) an existing configuration/model.

To share your model with the community, follow those steps: first import the ResNet model and config from the newly
created files:

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

Then you have to tell the library you want to copy the code files of those objects when using the `save_pretrained`
method and properly register them with a given Auto class (especially for models), just run:

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

Note that there is no need to specify an auto class for the configuration (there is only one auto class for them,
[`AutoConfig`]) but it's different for models. Your custom model could be suitable for many different tasks, so you
have to specify which one of the auto classes is the correct one for your model.

Next, let's create the config and models as we did before:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

Now to send the model to the Hub, make sure you are logged in. Either run in your terminal:

```bash
huggingface-cli login
```

or from a notebook:

```py
from huggingface_hub import notebook_login

notebook_login()
```

You can then push to your own namespace (or an organization you are a member of) like this:

```py
resnet50d.push_to_hub("custom-resnet50d")
```

On top of the modeling weights and the configuration in json format, this also copied the modeling and
configuration `.py` files in the folder `custom-resnet50d` and uploaded the result to the Hub. You can check the result
in this [model repo](https://huggingface.co/sgugger/custom-resnet50d).

See the [sharing tutorial](model_sharing) for more information on the push to Hub method.

## Using a model with custom code

You can use any configuration, model or tokenizer with custom code files in its repository with the auto-classes and
the `from_pretrained` method. All files and code uploaded to the Hub are scanned for malware (refer to the [Hub security](https://huggingface.co/docs/hub/security#malware-scanning) documentation for more information), but you should still 
review the model code and author to avoid executing malicious code on your machine. Set `trust_remote_code=True` to use
a model with custom code:

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

It is also strongly encouraged to pass a commit hash as a `revision` to make sure the author of the models did not
update the code with some malicious new lines (unless you fully trust the authors of the models).

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

Note that when browsing the commit history of the model repo on the Hub, there is a button to easily copy the commit
hash of any commit.

## Registering a model with custom code to the auto classes

If you are writing a library that extends 🤗 Transformers, you may want to extend the auto classes to include your own
model. This is different from pushing the code to the Hub in the sense that users will need to import your library to
get the custom models (contrarily to automatically downloading the model code from the Hub).

As long as your config has a `model_type` attribute that is different from existing model types, and that your model
classes have the right `config_class` attributes, you can just add them to the auto classes like this:

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

Note that the first argument used when registering your custom config to [`AutoConfig`] needs to match the `model_type`
of your custom config, and the first argument used when registering your custom models to any auto model class needs
to match the `config_class` of those models.
