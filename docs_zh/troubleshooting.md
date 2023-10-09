<!---
版权所有2022年的HuggingFace团队。

根据Apache许可证，第2.0版本（“许可证”）进行许可；
除非符合许可证的规定，否则你不得使用此文件。
你可以获得许可证的副本，该副本可以在以下位置获取：

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件是按“原样”无任何形式的“嘉奖”方式分发的，
没有任何形式的明示或暗示的保证及条件，包括但不限于适销性保证、适用于特定用途的保证及对非侵权的保证。
请参阅许可证了解有关许可证的详细信息。

⚠️ 请注意，此文件是Markdown格式的，但包含我们的doc-builder的特定语法（类似于MDX），
可能在你的Markdown查看器中无法正确呈现。

-->

# 疑难解答

有时会发生错误，但我们在这里提供帮助！本指南涵盖了一些我们遇到的最常见问题以及如何解决它们。但是，本指南不意味着是一个全面收集每个🤗Transformers问题的集合。如果需要更多帮助来解决问题，可以尝试以下方法：

<Youtube id="S2EEG3JIt2A"/>

1. 在[论坛](https://discuss.huggingface.co/)上寻求帮助。有特定的类别可以发布问题，例如[新手](https://discuss.huggingface.co/c/beginners/5)或[🤗 Transformes](https://discuss.huggingface.co/c/transformers/9)等。请确保你在论坛帖子中写入一个良好的描述论坛帖子，并提供一些可重现的代码，以最大程度地提高解决问题的可能性！

<Youtube id="_PAli-V4wj0"/>

2. 如果这是与库相关的错误，则可以在🤗Transformers存储库上创建一个[Issue](https://github.com/huggingface/transformers/issues/new/choose)。请尽量提供尽可能多的描述错误的信息，以帮助我们更好地找出问题出在哪里以及如何修复它。

3. 如果你使用的是旧版🤗Transformers，请查看[Migration](migration)指南，因为在版本之间引入了一些重要的更改。

有关疑难解答和获取帮助的更多详细信息，请参阅Hugging Face课程的[第8章](https://huggingface.co/course/chapter8/1?fw=pt)。


## 防火墙环境

在云和内部网络设置的某些GPU实例上有防火墙，无法连接到外部连接，导致连接错误。当你的脚本尝试下载模型权重或数据集时，下载过程将会中断并在超时后显示以下消息：

```
ValueError: Connection error, and we cannot find the requested files in the cached path.
Please try again or make sure your Internet connection is on.
```

在这种情况下，你可以尝试在[离线模式](installation.md#offline-mode)下运行🤗Transformers，以避免连接错误。

## CUDA内存不足

在没有适当硬件的情况下训练拥有数百万参数的大型模型可能是具有挑战性的。当GPU的内存用完时，你可能遇到如下错误：

```
CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 11.17 GiB total capacity; 9.70 GiB already allocated; 179.81 MiB free; 9.85 GiB reserved in total by PyTorch)
```

以下是你可以尝试的一些潜在解决方案来减少内存使用：

- 减少[`TrainingArguments`]中的[`per_device_train_batch_size`]值。
- 尝试在[`TrainingArguments`]中使用[`gradient_accumulation_steps`]来有效增加总的批量大小。

<Tip>

有关节省内存的技巧的详细信息，请参阅性能[指南](performance.md)。

</Tip>


## 无法加载已保存的TensorFlow模型

TensorFlow的[model.save](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model)方法将整个模型（架构，权重，训练配置）保存在一个文件中。然而，当你再次加载模型文件时，可能会遇到错误，因为🤗Transformers可能无法加载模型文件中的所有与TensorFlow相关的对象。为避免保存和加载TensorFlow模型时出现问题，我们建议你：

- 使用[`model.save_weights`]将模型权重保存为`h5`文件扩展名，然后使用[`~TFPreTrainedModel.from_pretrained`]重新加载模型：

```py
>>> from transformers import TFPreTrainedModel
>>> from tensorflow import keras

>>> model.save_weights("some_folder/tf_model.h5")
>>> model = TFPreTrainedModel.from_pretrained("some_folder")
```

- 使用[`~TFPretrainedModel.save_pretrained`]保存模型，并使用[`~TFPreTrainedModel.from_pretrained`]重新加载它：

```py
>>> from transformers import TFPreTrainedModel

>>> model.save_pretrained("path_to/model")
>>> model = TFPreTrainedModel.from_pretrained("path_to/model")
```

## ImportError

你可能会遇到的另一个常见错误，尤其是对于新发布的模型，是`ImportError`：

```
ImportError: cannot import name 'ImageGPTImageProcessor' from 'transformers' (unknown location)
```

对于这些错误，请检查你是否已经安装了最新版本的🤗Transformers，以便访问最新的模型：

```bash
pip install transformers --upgrade
```

## CUDA错误：设备侧断言被触发

有时你可能会遇到关于设备代码中错误的通用CUDA错误。

```
RuntimeError: CUDA error: device-side assert triggered
```

你应该首先在CPU上运行代码以获得更详细的错误消息。在你的代码开头添加以下环境变量以切换到CPU：

```py
>>> import os

>>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

另一个选择是从GPU获取更好的回溯。在你的代码开头添加以下环境变量，以使回溯指向错误来源：

```py
>>> import os

>>> os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

## 在未对填充令牌进行掩码处理时输出不正确

在某些情况下，如果`input_ids`包含填充令牌，则输出的`hidden_state`可能是不正确的。为了演示，加载一个模型和分词器。你可以访问模型的`pad_token_id`来查看其值。对于某些模型，`pad_token_id`可能为`None`，但你始终可以手动设置它。

```py
>>> from transformers import AutoModelForSequenceClassification
>>> import torch

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
>>> model.config.pad_token_id
0
```

以下示例显示了在不掩码填充令牌的情况下的输出：

```py
>>> input_ids = torch.tensor([[7592, 2057, 2097, 2393, 9611, 2115], [7592, 0, 0, 0, 0, 0]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [ 0.1317, -0.1683]], grad_fn=<AddmmBackward0>)
```

以下是第二个序列的实际输出：

```py
>>> input_ids = torch.tensor([[7592]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

大多数情况下，应该为模型提供一个`attention_mask`以忽略填充令牌，以避免这种潜在的错误。现在第二个序列的输出与其实际输出匹配：

<Tip>

默认情况下，分词器根据特定分词器的默认设置为你创建一个`attention_mask`。

</Tip>

```py
>>> attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
>>> output = model(input_ids, attention_mask=attention_mask)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

🤗Transformers不会自动创建一个`attention_mask`以遮蔽填充令牌，如果提供了填充令牌，因为：

- 有些模型没有填充令牌。
- 对于某些用例，用户希望模型关注填充令牌。

## ValueError：无法识别的配置类XYZ，不适用于此类AutoModel

通常，我们建议使用[`AutoModel`]类来加载预训练模型的实例。该类可以根据给定的检查点中的配置自动推断和加载正确的体系结构。如果在加载检查点时出现此`ValueError`，这意味着自动类无法从给定检查点中的配置到你要加载的模型种类之间找到映射。最常见的情况是当一个检查点不支持给定的任务时，就会出现这个错误。例如，在以下示例中，你会看到此错误，因为没有适用于问题回答的GPT2：

```py
>>> from transformers import AutoProcessor, AutoModelForQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("gpt2-medium")
>>> model = AutoModelForQuestionAnswering.from_pretrained("gpt2-medium")
ValueError: Unrecognized configuration class <class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> for this kind of AutoModel: AutoModelForQuestionAnswering.
Model type should be one of AlbertConfig, BartConfig, BertConfig, BigBirdConfig, BigBirdPegasusConfig, BloomConfig, ...
```
