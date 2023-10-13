<!--版权所有2020 The HuggingFace团队。保留所有权利。

根据Apache License，Version 2.0的许可证（“许可证”），你不得使用此文件，除非符合许可证的规定。
你可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，否则根据许可证分发的软件是按“原样”方式分发的，不附带任何形式的保证或条件，无论是明示的还是暗示的。请参阅许可证中的特定语言，以了解许可的特定语言
对于限制和限制的情况。

⚠ 注意，这个文件是用Markdown格式编写的，但包含我们的doc-builder的特定语法（类似于MDX），这可能不能在你的Markdown查看器中正确显示。

-->

# 共享自定义模型

🤗Transformers库旨在易于扩展。每个模型都在仓库的给定子文件夹中完全编码，没有任何抽象，因此你可以轻松复制建模文件并针对你的需求进行调整。

如果你要编写全新的模型，最好从头开始。在本教程中，我们将向你展示如何编写自定义模型及其配置，以便在Transformers中使用，并展示如何与社区共享（使用它所依赖的代码），以便任何人都可以使用它，即使它不在🤗Transformers库中。

我们将通过将ResNet类包装到 [`PreTrainedModel`] 中来说明所有这些内容。

## 编写自定义配置

在我们深入模型之前，让我们首先编写其配置。模型的配置是一个对象，它将包含构建模型所需的所有必要信息。正如我们将在下一节中看到的，模型只能采用 `config` 来初始化，因此我们确实需要该对象尽可能完整。

在我们的示例中，我们将使用可能需要调整的ResNet类的几个参数。然后，不同的配置将给我们带来不同类型的ResNet。然后，我们只需要存储这些参数，并检查其中一些参数的有效性。

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

编写自己的配置时要记住的三个重要事项如下：
- 必须继承自`PretrainedConfig`,
- `PretrainedConfig` 的 `__init__` 必须接受任何 kwargs,
- 这些 `kwargs` 需要传递给超类中的`__init__`.

继承是为了确保你获得来自🤗Transformers库的所有功能，而另外两个限定条件是因为`PretrainedConfig`的字段比你设置的字段多。在使用`from_pretrained` 方法重新加载配置时，这些字段需要被你的配置接受，然后发送给超类。

为你的配置定义 `model_type`（这里是 `model_type="resnet"`）是可选的，除非你希望使用自动类注册你的模型（见下一节）。

完成后，你可以像使用库中任何其他模型配置一样轻松创建和保存你的配置。下面是如何创建一个 resnet50d 配置并保存它的示例：

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

这将在文件夹 `custom-resnet` 中保存一个名为 `config.json` 的文件。然后，你可以使用 `from_pretrained` 方法重新加载你的配置：

```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

你还可以使用 [`PretrainedConfig`] 类的任何其他方法，例如 [`~PretrainedConfig.push_to_hub`] 将你的配置直接上传到 Hub 中。

## 编写自定义模型

现在我们有了 ResNet 配置，可以继续编写模型。事实上，我们将编写两个：一个从图像批量提取隐藏特征的模型（如 [`BertModel`]），以及一个适用于图像分类的模型（如 [`BertForSequenceClassification`]）。

如前所述，我们只编写一个简单的模型包装器，以使示例保持简单。在编写此类之前，我们只需要一个块类型和实际块类之间的映射。然后，通过将所有东西传递到 `ResNet` 类的配置，定义模型：

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

对于将对图像进行分类的模型，我们只需更改 forward 方法：

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

在这两种情况下，请注意我们从 `PreTrainedModel` 继承并使用 `config` 调用超类初始化（有点像你编写常规的 `torch.nn.Module` 时）。设置 `config_class` 的行不是必需的，除非你想要使用自动类注册模型（见下一节）。

<Tip>

如果你的模型与库中的模型非常相似，则可以重用与该模型相同的配置。

</Tip>

你可以使你的模型返回任何你想要的内容，但是对于如`ResnetModelForImageClassification`，我们返回一个字典，包含在传递标签时包含损失的logits。只要你打算使用自己的训练循环或其他训练库，使用其他输出格式都是可以的。

现在我们有了模型类，我们来创建一个：

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

同样，你可以使用[`PreTrainedModel`] 的任何方法，如 [`~PreTrainedModel.save_pretrained`] 或 [`~PreTrainedModel.push_to_hub`]。我们将在下一节中使用后者，看一下如何将模型权重与模型的代码一起上传。但是首先，让我们加载一些预训练权重到我们的模型中。

在你自己的用例中，你可能会使用自己的数据对自定义模型进行训练。为了加快本教程的进展，我们将使用 resnet50d 的预训练版本。由于我们的模型只是它的封装器，所以很容易转移这些权重：

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

现在让我们看一下如何确保我们执行 [`~PreTrainedModel.save_pretrained`] 或 [`~PreTrainedModel.push_to_hub`] 时，模型的代码得到保存。

## 将代码上传到Hub

<提示 警告={true}>

此 API 是实验性的，在今后的发布中可能会有一些细微的破坏性更改。

</Tip>

首先，请确保你的模型在一个 `.py` 文件中完全定义。它可以依赖相对引入到其他文件，只要所有文件都在同一个目录中（我们暂时不支持子模块）。对于我们的示例，我们将在当前工作目录命名为 `resnet_model` 的文件夹中定义一个 `modeling_resnet.py` 文件和一个 `configuration_resnet.py` 文件。配置文件包含 `ResnetConfig` 的代码，建模文件包含 `ResnetModel` 和 `ResnetModelForImageClassification` 的代码。

```
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

`__init__.py`可以是空的，只是为了让Python检测到`resnet_model`可以作为一个模块。

<提示警告={true}>

如果从库中复制建模文件，则需要将文件顶部的所有相对引入替换为从 `transforms` 包中引入。

</Tip>

请注意，你可以重新使用（或者为之类别化）现有的配置/模型。

要与社区共享你的模型，请遵循以下步骤：首先从新创建的文件中导入 ResNet 模型和配置：

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

然后，你要告诉库在使用 `save_pretrained ` 方法时你要复制这些对象的代码文件，并使用给定的自动类来适当地注册它们（特别是对于模型），请运行：

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

请注意，对于配置，不需要指定自动类（对于它们只有一个自动类，[`AutoConfig`]）但是对于模型是不同的。你的自定义模型可能适用于许多不同的任务，因此必须指定自动类中的哪个是你的模型的正确类。

接下来，让我们像之前一样创建配置和模型：

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

现在，要将模型推送到 Hub，请确保已登录。可以在终端中运行：

```bash
huggingface-cli login
```

或者在笔记本中运行：

```py
from huggingface_hub import notebook_login

notebook_login()
```

然后，你可以将其推送到自己的命名空间（或你是其成员的组织）中，如下所示：

```py
resnet50d.push_to_hub("custom-resnet50d")
```

除了以 `.json` 格式保存的建模权重和配置外，这还会将建模和配置的 `.py` 文件复制到文件夹 `custom-resnet50d` 中，并将结果上传到Hub。你可以在这个[model repo](https://huggingface.co/sgugger/custom-resnet50d)中查看结果。

有关Hub的更多信息，请参见[sharing tutorial](model_sharing.md)。

## 使用具有自定义代码的模型

你可以使用任何带有存储在其存储库中的自定义代码文件的配置、模型或标记器与自动类和 `from_pretrained` 方法一起使用。将所有文件和代码上传到Hub后，将扫描以及进行恶意软件检查（有关更多信息，请参阅[Hub安全](https://huggingface.co/docs/hub/security#malware-scanning)文档），但是你仍然应该
review模型代码和作者以避免在计算机上执行恶意代码。将 `trust_remote_code=True` 设置为使用具有自定义代码的模型：

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

同时，强烈建议传递一个提交哈希作为 `revision`，以确保模型的作者没有使用一些恶意的新行更新代码（除非你完全信任模型的作者）。

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

请注意，在Hub上浏览模型仓库的提交历史记录时，有一个按钮可以轻松复制任何提交的提交哈希。

## 向自动类中注册具有自定义代码的模型

如果要编写一个扩展🤗Transformers的库，你可能希望扩展自动类以包括自己的模型。这在某种程度上与将代码推送到Hub不同，在这种情况下，用户需要导入你的库以获取自定义模型（与自动从Hub中下载模型代码相反）。

只要你的配置具有与现有的模型类型不同的 `model_type` 属性，且 你的模型类具有正确的 `config_class` 属性，你就可以像这样将它们添加到自动类中：

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

请注意，你在对 [`AutoConfig`] 注册自定义配置时使用的第一个参数需要与自定义配置的 `model_type` 匹配，你在注册任何自动模型类时使用的第一个参数需要与这些模型的`config_class` 匹配。

