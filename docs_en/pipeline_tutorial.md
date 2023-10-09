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

# Pipelines用于推理

[`pipeline`]使得在任何语言、计算机视觉、语音和多模态任务上使用[Hub](https://huggingface.co/models)中的任何模型变得非常简单。即使您对特定模态没有经验或不熟悉模型背后的底层代码，您仍然可以使用[`pipeline`]进行推理！本教程将教您：

- 使用[`pipeline`]进行推理。
- 使用特定的分词器或模型。
- 在音频、视觉和多模态任务中使用[`pipeline`]。

<Tip>

查看[`pipeline`]文档，以获取支持的任务列表和可用参数的完整信息。

</Tip>

## Pipeline usage

虽然每个任务都有一个相关联的[`pipeline`]，但使用通用的[`pipeline`]抽象更为简单，该抽象包含了所有特定任务的pipelines。[`pipeline`]会自动加载默认模型和适用于您任务的预处理类，以实现推理功能。

1. 首先创建一个[`pipeline`]并指定一个推理任务：

```py
>>> from transformers import pipeline

>>> generator = pipeline(task="automatic-speech-recognition")
```

2. 将输入文本传递给[`pipeline`]：

```py
>>> generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}
```

结果不符合您的期望？请查看Hub上一些[最受欢迎的自动语音识别模型](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads)，看看是否可以获得更好的转录结果。 

让我们尝试一下[openai/whisper-large](https://huggingface.co/openai/whisper-large)：

```py
>>> generator = pipeline(model="openai/whisper-large")
>>> generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

现在的结果看起来更准确了！

我们鼓励您在Hub上寻找适用于不同语言、专门针对您领域的模型等。您可以直接在浏览器上查看和比较Hub上的模型结果，以确定是否符合您的需求或是否能够更好地处理特殊情况。如果您没有找到适合您用例的模型，您始终可以开始[训练](http://www.liuzard.com/training)自己的模型！

如果您有多个输入，您可以将输入作为列表传递给[`pipeline`]：

```py
generator(
    [
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ]
)
```

如果您想迭代整个数据集，或者想将其用于Web服务器中的推理，请查看专门的部分

[在数据集上使用pipelines](#using-pipelines-on-a-dataset)

[在Web服务器上使用pipelines](pipeline_webserver.md)

## 参数

[`pipeline`]支持许多参数；其中一些是特定于任务的，而另一些是适用于所有pipelines的通用参数。
一般来说，您可以在任何地方指定参数：

```py
generator = pipeline(model="openai/whisper-large", my_parameter=1)
out = generator(...)  # This will use `my_parameter=1`.
out = generator(..., my_parameter=2)  # This will override and use `my_parameter=2`.
out = generator(...)  # This will go back to using `my_parameter=1`.
```

让我们查看三个重要的参数：

### 设备（Device）

如果您使用`device=n`，则[`pipeline`]会自动将模型放置在指定的设备上。
无论您是使用PyTorch还是Tensorflow，都可以使用这个参数。

```py
generator = pipeline(model="openai/whisper-large", device=0)
```

如果模型对于单个GPU来说太大了，您可以将`device_map="auto"`设置为允许🤗 [Accelerate](https://huggingface.co/docs/accelerate)自动确定如何加载和存储模型权重。

```py
#!pip install accelerate
generator = pipeline(model="openai/whisper-large", device_map="auto")
```

请注意，如果传递了`device_map="auto"`参数，则在实例化[`pipeline`]时不需要添加`device=device`参数，否则可能会遇到一些意外的行为！

### 批处理大小（Batch size）

默认情况下，pipelines不会对推理进行批处理，详细原因可以在[这里](https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching)找到。原因是批处理并不一定更快，而且在某些情况下实际上可能更慢。

但是，如果在您的用例中可以使用批处理，请使用以下方式：

```py
generator = pipeline(model="openai/whisper-large", device=0, batch_size=2)
audio_filenames = [f"audio_{i}.flac" for i in range(10)]
texts = generator(audio_filenames)
```

这会对提供的10个音频文件运行pipeline，但它会将它们以2个一组的批次传递给模型（模型位于GPU上，在这种情况下批处理可能更有帮助），而无需您进一步编写任何代码。 输出应该始终与您在没有批处理的情况下接收到的结果相匹配。这只是一种帮助您提高pipeline速度的方式。

pipelines还可以减轻批处理的一些复杂性，因为对于某些pipelines来说，需要将单个项目（如长音频文件）分成多个部分以供模型处理。pipeline会为您执行这种[*chunk batching*](http://www.liuzard.com/main_classes/pipelines#pipeline-chunk-batching)。

### 特定任务的参数

所有任务都提供特定任务的参数，这些参数允许额外的灵活性和选项，以帮助您完成工作。 例如，[`transformers.AutomaticSpeechRecognitionPipeline.__call__`]方法具有一个`return_timestamps`参数，对于为视频生成字幕似乎是一个有希望的选择：

### 特定任务的参数

所有任务都提供特定任务的参数，这些参数允许额外的灵活性和选项，以帮助您完成工作。
例如，[`transformers.AutomaticSpeechRecognitionPipeline.__call__`]方法具有一个`return_timestamps`参数，对于为视频生成字幕似乎是一个有希望的选择：


```py
>>> # Not using whisper, as it cannot provide timestamps.
>>> generator = pipeline(model="facebook/wav2vec2-large-960h-lv60-self", return_timestamps="word")
>>> generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP AND LIVE OUT THE TRUE MEANING OF ITS CREED', 'chunks': [{'text': 'I', 'timestamp': (1.22, 1.24)}, {'text': 'HAVE', 'timestamp': (1.42, 1.58)}, {'text': 'A', 'timestamp': (1.66, 1.68)}, {'text': 'DREAM', 'timestamp': (1.76, 2.14)}, {'text': 'BUT', 'timestamp': (3.68, 3.8)}, {'text': 'ONE', 'timestamp': (3.94, 4.06)}, {'text': 'DAY', 'timestamp': (4.16, 4.3)}, {'text': 'THIS', 'timestamp': (6.36, 6.54)}, {'text': 'NATION', 'timestamp': (6.68, 7.1)}, {'text': 'WILL', 'timestamp': (7.32, 7.56)}, {'text': 'RISE', 'timestamp': (7.8, 8.26)}, {'text': 'UP', 'timestamp': (8.38, 8.48)}, {'text': 'AND', 'timestamp': (10.08, 10.18)}, {'text': 'LIVE', 'timestamp': (10.26, 10.48)}, {'text': 'OUT', 'timestamp': (10.58, 10.7)}, {'text': 'THE', 'timestamp': (10.82, 10.9)}, {'text': 'TRUE', 'timestamp': (10.98, 11.18)}, {'text': 'MEANING', 'timestamp': (11.26, 11.58)}, {'text': 'OF', 'timestamp': (11.66, 11.7)}, {'text': 'ITS', 'timestamp': (11.76, 11.88)}, {'text': 'CREED', 'timestamp': (12.0, 12.38)}]}
```

正如您所见，模型推断出了文本，并输出了句子中各个单词的发音时间。

每个任务都有许多可用的参数，因此请查看每个任务的API参考，了解您可以进行哪些调整！例如，[`~transformers.AutomaticSpeechRecognitionPipeline`]具有一个`chunk_length_s`参数，对于处理非常长的音频文件（例如，为整部电影或长达一小时的视频添加字幕）非常有帮助。这样的音频文件通常一个模型无法独自处理。

如果您找不到真正有用的参数，请随时[提出请求](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml)！

## 在数据集上使用pipelines

pipeline还可以在大型数据集上运行推理。我们建议的最简单方法是使用迭代器：

```py
def data():
    for i in range(1000):
        yield f"My example {i}"


pipe = pipeline(model="gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])
```

迭代器`data()`会逐个生成结果，而pipeline会自动识别输入为可迭代对象，并在继续在GPU上处理数据的同时开始获取数据（这在底层使用了[DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)）。这很重要，因为您不需要为整个数据集分配内存，可以尽可能快地将数据提供给GPU。

由于批处理可能加快速度，调整`batch_size`参数可能会有所帮助。

迭代整个数据集的最简单方法就是从🤗 [Datasets](https://github.com/huggingface/datasets/)中加载数据集：

```py
# KeyDataset is a util that will just output the item we're interested in.
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

for out in pipe(KeyDataset(dataset, "audio")):
    print(out)
```

## 使用pipelines构建Web服务器

<Tip> 创建推理引擎是一个复杂的主题，值得拥有自己的页面。 </Tip>

[链接](http://www.liuzard.com/pipeline_webserver)

## 视觉任务的pipeline

对于视觉任务，使用[`pipeline`]几乎是相同的。

指定您的任务，并将图像传递给分类器。图像可以是链接、本地路径或Base64编码的图像。例如，下面显示的是哪种猫的品种？

![pipeline-cat-chonk](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)

```py
>>> from transformers import pipeline

>>> vision_classifier = pipeline(model="google/vit-base-patch16-224")
>>> preds = vision_classifier(
...     images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4335, 'label': 'lynx, catamount'}, {'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}, {'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}, {'score': 0.0239, 'label': 'Egyptian cat'}, {'score': 0.0229, 'label': 'tiger cat'}]
```

## 文本任务的pipeline

对于自然语言处理（NLP）任务，使用[`pipeline`]几乎是相同的。

```py
>>> from transformers import pipeline

>>> # This model is a `zero-shot-classification` model.
>>> # It will classify text, except you are free to choose any label you might imagine
>>> classifier = pipeline(model="facebook/bart-large-mnli")
>>> classifier(
...     "I have a problem with my iphone that needs to be resolved asap!!",
...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
```

## 多模态任务的pipeline

[`pipeline`]支持多个模态。例如，视觉问答（VQA）任务结合了文本和图像。您可以随意使用任何您喜欢的图像链接和要提出的问题。图像可以是URL或指向图像的本地路径。

例如，如果您使用这张[发票图像](https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png)：

```py
>>> from transformers import pipeline

>>> vqa = pipeline(model="impira/layoutlm-document-qa")
>>> vqa(
...     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
...     question="What is the invoice number?",
... )
[{'score': 0.42515, 'answer': 'us-001', 'start': 16, 'end': 16}]
```

<Tip>

要运行上面的示例，除了🤗 Transformers之外，您还需要安装[`pytesseract`](https://pypi.org/project/pytesseract/)：

```bash
sudo apt install -y tesseract-ocr
pip install pytesseract
```

</Tip>

## 使用`pipeline`处理大型模型与🤗 `accelerate`：

您可以使用🤗 `accelerate`轻松地在大型模型上运行`pipeline`！首先确保已经使用`pip install accelerate`安装了`accelerate`。

首先使用`device_map="auto"`加载您的模型！我们将在示例中使用`facebook/opt-1.3b`。

```py
# pip install accelerate
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="auto")
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```

如果您安装了`bitsandbytes`并添加了参数`load_in_8bit=True`，还可以传递加载的8位模型。

```py
# pip install accelerate bitsandbytes
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": True})
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```

请注意，您可以将检查点替换为任何支持大模型加载的Hugging Face模型，例如BLOOM！
