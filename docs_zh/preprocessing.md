# 预处理

[[在 colab 中打开]]

在使用数据集对模型进行训练之前，需要将数据预处理为模型期望的输入格式。无论数据是文本、图片还是音频，都需要转换和组装成张量批次。🤗 Transformers 提供了一组预处理类来帮助准备数据供模型使用。在本教程中，你将学到以下内容：

* 对于文本，使用[Tokenizer](main_classes/tokenizer)将文本转换为token序列，创建token的数值表示，并将其组装成张量。
* 对于语音和音频，使用[Feature extractor](main_classes/feature_extractor)从音频波形中提取时序特征，并将其转换为张量。
* 对于图像输入，使用[ImageProcessor](main_classes/image)将图像转换为张量。
* 对于多模式输入，使用[Processor](main_classes/processors)将分词处理器与特征提取器或图像处理器组合在一起。

<Tip>

`AutoProcessor`**总是**工作，并自动选择适合你使用的模型的正确类别，无论你使用的是分词处理器、图像处理器、特征提取器，还是处理器。

</Tip>

在开始之前，首先安装 🤗 Datasets，这样可以加载一些数据集进行实验：

```bash
pip install datasets
```

## 自然语言处理

<Youtube id="Yffk5aydLzg"/>

预处理文本数据的主要工具是[分词处理器](main_classes/tokenizer)。分词处理器根据一组规则将文本拆分为*token*。然后将这些token转换为数值，并将其组装成张量，作为模型的输入。分词处理器还会添加模型所需的任何其他输入。

<Tip>

如果打算使用预训练模型，使用相应的预训练分词处理器非常重要。这样可以确保文本的拆分方式与预训练语料库相同，并在预训练过程中使用相同的索引标记（通常称为*词汇表*）。

</Tip>

首先使用[`AutoTokenizer.from_pretrained`]方法加载预训练分词处理器。这将下载模型的*词汇表*：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

然后将文本传递给分词处理器：

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

分词处理器返回一个包含三个重要项的字典：

* [input_ids](glossary.md#input-ids) 是句子中每个token对应的索引。
* [attention_mask](glossary.md#attention-mask) 指示一个token是否应该被注意。
* [token_type_ids](glossary.md#token-type-ids) 是当有多个序列时，标识一个token属于哪个序列。

通过对`input_ids`进行解码，可以返回输入的内容：

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

可以看到，分词处理器在句子中添加了两个特殊token - `CLS`和`SEP`（分类器和分隔符）。并非所有模型都需要特殊token，但如果需要，分词处理器会自动为你添加。

如果有几个句子需要预处理，可以将它们作为列表传递给分词处理器：

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]]}
```

### 填充

句子的长度并不总是相同，这可能会有问题，因为张量（即模型输入）需要具有统一的形状。填充是一种策略，通过为较短的句子添加一个特殊的*填充标记*，使张量变得规则而矩形。

将 `padding` 参数设置为 `True`，以匹配最长的序列，将批次中较短的序列进行填充：

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

前面的第一句和第三句现在都填充了 `0`，以匹配最长的句子。

### 截断

另一方面，有时候一个序列可能过长，以至于模型无法处理。在这种情况下，需要将序列截断为较短的长度。

将 `truncation` 参数设置为 `True`，将序列截断为模型接受的最大长度：

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

<Tip>

查看[填充和截断](pad_truncation.md)概念指南，了解更多不同的填充和截取参数。

</Tip>

### 构建张量

最后，希望分词处理器返回实际传递给模型的张量。

将 `return_tensors` 参数设置为 `pt`（PyTorch）或 `tf`（TensorFlow）：

<frameworkcontent>
<pt>

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                      [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                      [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```
</pt>
<tf>
```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
       [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
       [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>,
 'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>,
 'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>}
```
</tf>
</frameworkcontent>

## 音频

对于音频任务，需要使用[特征提取器](main_classes/feature_extractor)来准备数据集以供模型使用。特征提取器旨在从原始音频数据中提取特征，并将其转换为张量。

加载 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 数据集（参见 🤗 [Datasets 教程](https://huggingface.co/docs/datasets/load_hub.html)了解如何加载数据集的更多细节），以查看如何将特征提取器应用于音频数据集：

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

访问 `audio` 列的第一个元素，查看输入的内容。调用 `audio` 列会自动加载和重采样音频文件：

```py
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

这返回了三个项目：

* `array` 是作为1D数组加载 - 并且可能重采样 - 的语音信号。
* `path` 指向音频文件的位置。
* `sampling_rate` 表示每秒测量语音信号的数据点数。

在本教程中，你将使用[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base)模型。查看模型卡片，你将了解到 Wav2Vec2 是在16kHz 采样率的语音音频上预训练的。确保音频数据的采样率与用于预训练模型的数据集的采样率相匹配很重要。如果数据的采样率不同，那么需要对数据进行重新采样。

1. 使用 🤗 Datasets 的 [`~datasets.Dataset.cast_column`] 方法将采样率提高到 16kHz：

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

2. 再次调用 `audio` 列，以重新采样音频文件：

```py
>>> dataset[0]["audio"]
{'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
         3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 16000}
```

接下来，加载特征提取器来对输入进行归一化和填充。当填充文本数据时，会将 `0` 添加到较短的序列。音频数据也是同样道理。特征提取器会将 `array` 添加一个 `0`，并将其解释为空白。

使用[`AutoFeatureExtractor.from_pretrained`] 加载特征提取器：

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

将音频 `array` 传递给特征提取器。我们还建议在特征提取器中添加 `sampling_rate` 参数，以便更好地调试可能发生的静默错误。

```py
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
        5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

与分词处理器类似，可以对批次中的可变长度序列应用填充或截断以进行处理。来看一下这两个音频样本的序列长度：

```py
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

创建一个函数来预处理数据集，以使音频样本有相同的长度。指定最大的样本长度，特征提取器将填充或截断序列以匹配长度：

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays,
...         sampling_rate=16000,
...         padding=True,
...         max_length=100000,
...         truncation=True,
...     )
...     return inputs
```

将 `preprocess_function` 应用到数据集中的前几个示例：

```py
>>> processed_dataset = preprocess_function(dataset[:5])
```

样本长度现在相同，并与指定的最大长度匹配。现在可以将处理后的数据集传递给模型了！

```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```

## 计算机视觉

对于计算机视觉任务，你需要一个[image processor](main_classes/image_processor)来为模型准备数据集。
图像预处理包括将图像转换为模型所需的输入的多个步骤。这些步骤包括但不限于resize、归一化、颜色通道校正和将图像转换为张量。

<Tip>

图像预处理通常在图像增强之后进行。图像预处理和图像增强都可以转换图像数据，但它们有不同的目的：

* 图像增强以一种可以帮助防止过拟合和增加模型鲁棒性的方式改变图像。你可以在数据增强中进行创造性操作 - 调整亮度和颜色、裁剪、旋转、resize、缩放等。但是，请注意，不要通过增强改变图像的含义。
* 图像预处理保证图像与模型的预期输入格式相匹配。在微调计算机视觉模型时，图像必须像训练模型时那样进行预处理。

你可以使用任何喜欢的库进行图像增强。对于图像预处理，请使用与模型相关联的`ImageProcessor`。

</Tip>

加载[food101](https://huggingface.co/datasets/food101)数据集（有关如何加载数据集的详细信息，请参见🤗 [Datasets教程](https://huggingface.co/docs/datasets/load_hub.html)），看看你如何在计算机视觉数据集中使用图像处理器：

<Tip>

使用🤗 Datasets的`split`参数只加载训练集中的一小部分样本，因为数据集非常大！

</Tip>

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

接下来，查看包含于🤗 Datasets [`Image`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=image#datasets.Image)特征的图像：

```py
>>> dataset[0]["image"]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png"/>
</div>

使用[`AutoImageProcessor.from_pretrained`]加载图像处理器：

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

首先，让我们添加一些图像增强。你可以使用任何你喜欢的库，但在本教程中，我们将使用torchvision的[`transforms`](https://pytorch.org/vision/stable/transforms.html)模块。如果你想使用另一个数据增强库，请参阅[Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb)或[Kornia notebooks](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb)了解更多信息。

1. 在这里，我们使用[`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html)将几个转换链接在一起 - [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)和[`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)。注意，对于resize，我们可以从`image_processor`获取图像大小要求。对于某些模型，期望精确的高度和宽度，对于其他模型只定义了"shortest_edge"。

```py
>>> from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )

>>> _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

2. 该模型接受[`pixel_values`](model_doc/visionencoderdecoder#transformers.VisionEncoderDecoderModel.forward.pixel_values)作为输入。`ImageProcessor`可以负责归一化图像并生成适当的张量。创建一个函数，用于将一批图像进行图像增强和图像预处理，并生成`pixel_values`：

```py
>>> def transforms(examples):
...     images = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
...     return examples
```

<Tip>

在上面的示例中，我们设置了`do_resize=False`，因为我们已经在图像增强转换中调整了图像大小，并利用了适当的`image_processor`的`size`属性。对于不在图像增强期间调整图像大小的情况，请省略此参数。默认情况下，`ImageProcessor`会处理调整大小。

如果你希望将规范化图像作为增强转换的一部分，使用`image_processor.image_mean`和`image_processor.image_std`值。
</Tip>

3. 然后使用🤗 Datasets [`set_transform`](https://huggingface.co/docs/datasets/process.html#format-transform)来实时应用转换：

```py
>>> dataset.set_transform(transforms)
```

4. 现在，当你访问图像时，你会注意到图像处理器已添加了`pixel_values`。你现在可以将处理后的数据集传递给模型！

```py
>>> dataset[0].keys()
```

在应用转换后，图像的处理结果如下所示。图像已被随机裁剪，其颜色属性也不同。

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png"/>
</div>

<Tip>

对于对象检测、语义分割、实例分割和全景分割等任务，`ImageProcessor`提供了后处理方法。这些方法将模型的原始输出转换为有意义的预测，例如边界框或分割图。

</Tip>

### 填充

在某些情况下，例如，微调[DETR](model_doc/detr)时，模型会在训练时应用比例增强。这可能导致批处理中的图像大小不同。你可以使用[`DetrImageProcessor.pad`](model_doc/detr/#transformers.DetrImageProcessor.pad)和定义一个自定义的`collate_fn`来将图像批处理在一起。

```py
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## 多模态

对于涉及多模态输入的任务，你需要一个[processor](main_classes/processors)来为模型准备数据集。处理器将两个处理对象（例如tokenizer和feature extractor）耦合在一起。

加载[LJ Speech](https://huggingface.co/datasets/lj_speech)数据集（有关如何加载数据集的详细信息，请参见🤗 [Datasets教程](https://huggingface.co/docs/datasets/load_hub.html)），以查看如何在自动语音识别（ASR）中使用处理器：

```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

对于ASR，你主要关注`audio`和`text`，因此可以删除其他列：

```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

现在查看`audio`和`text`列：

```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

请记住，你应该始终[重新采样](preprocessing.md#audio)音频数据集的采样率，以与用于预训练模型的数据集的采样率相匹配！

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

使用[`AutoProcessor.from_pretrained`]加载处理器：

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. 创建一个函数来处理包含于`array`中的音频数据到`input_values`，并将`text`进行分词后得到`labels`。这些是模型的输入：

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. 将`prepare_dataset`函数应用于一个样本：

```py
>>> prepare_dataset(lj_speech[0])
```

处理器现在已经添加了`input_values`和`labels`，并且采样率也正确地降采样为16kHz。你现在可以将处理后的数据集传递给模型了！