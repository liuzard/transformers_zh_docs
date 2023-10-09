版权所有 2020年The HuggingFace团队。版权所有。

根据Apache许可证2.0版（“许可证”），除非符合许可证的规定，否则你不得使用此文件。你可以在以下网址获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证的规定分发，
创建于"AS IS"基础上，不附带任何形式的明示或默示保证。请参阅许可证的特定语言和限制条款。

⚠️请注意，此文件是Markdown格式的，但包含用于我们的doc-builder（类似于MDX）的特定语法，可能无法正确显示在你的Markdown查看器中。

# Pipelines

pipeline是使用模型进行推理的一种很好且简单的方法。这些pipeline是一个封装库中大部分复杂代码的对象，提供了一个简单的API，专用于多个任务，包括命名实体识别、掩码语言建模、情感分析、特征提取和问答。有关用法示例，请参阅任务摘要。

有两种类别的pipeline抽象需要注意：

- [`pipeline`] 是最强大的对象，封装了所有其他pipeline。
- 针对[音频](#音频)、[计算机视觉](#计算机视觉)、[自然语言处理](#自然语言处理)和[多模态](#多模态)任务提供了特定任务的pipeline。

## pipeline抽象

*pipeline*抽象是对所有其他可用pipeline的包装器。它与任何其他pipeline一样实例化，但可以提供额外的生活质量。

对一个项目进行简单调用：

```python
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

如果你想要使用hub上的特定模型，可以忽略hub上的模型是否定义了任务：

```python
>>> pipe = pipeline(model="roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

要在多个项上调用pipeline，可以使用*list*调用它。

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

在迭代整个数据集时，建议直接使用`dataset`。这意味着你不需要一次分配整个数据集，也不需要自己进行批处理。这应该与在GPU上的自定义循环一样快。如果不是，请随时提出问题。

```python
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

# KeyDataset（只有*pt*）将仅简单将数据集返回的字典中的项作为数据集项返回
# 因为我们对数据集的*target*部分不感兴趣。对于句对使用KeyPairDataset
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

为了使用方便，还可以使用generator：

```python
from transformers import pipeline

pipe = pipeline("text-classification")


def data():
    while True:
        # 这可能来自数据集、数据库、队列或HTTP请求
        # 服务器中
        # 注意：因为这是迭代的，所以你不能使用`num_workers > 1`变量
        # 同时使用多个线程对数据进行预处理。你仍然可以有1个线程
        # 做预处理，而主要运行大推理
        yield "This is a test"


for out in pipe(data()):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

[[autodoc]] pipeline

## pipeline批处理

所有pipeline都可以使用批处理。这将在pipeline使用其流式处理功能（因此传递列表、数据集或生成器时）起作用。

```python
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
    # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
    # 与之前完全相同的输出，但内容被批量传递给模型
```

<Tip warning={true}>

然而，这不是自动的性能提升。它可能是10倍的性能提升，也可能是5倍的性能降低，具体取决于硬件、数据和实际使用的模型。

基本上是速度更快的示例：

</Tip>

```python
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)


class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"


dataset = MyDataset()

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
```

```
# On GTX 970
------------------------------
Streaming no batching
100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
------------------------------
Streaming batch_size=64
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
------------------------------
Streaming batch_size=256
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
（递减回报，GPU已饱和）
```

主要是速度变慢的示例：

```python
class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        if i % 64 == 0:
            n = 100
        else:
            n = 1
        return "This is a test" * n
```

与其他句子相比，这是一个特别长的句子。在这种情况下，**整个**批次将需要长度为400个令牌，因此整个批次将为[64, 400]，而不是[64, 4]，导致较高的性能下降。更糟糕的是，在更大的批次上，程序会崩溃。

```
------------------------------
Streaming no batching
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 183.69it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 265.74it/s]
------------------------------
Streaming batch_size=64
100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.80it/s]
------------------------------
Streaming batch_size=256
  0%|                                                                                 | 0/1000 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/nicolas/src/transformers/test.py", line 42, in <module>
    for out in tqdm(pipe(dataset, batch_size=256), total=len(dataset)):
....
    q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 3.95 GiB total capacity; 1.72 GiB already allocated; 354.88 MiB free; 2.46 GiB reserved in total by PyTorch)
```

对于这个问题没有好的（通用）解决方案，你的使用情况可能会有所不同。经验法则：

对于用户来说，经验法则是：

- **测量你的负载的性能，通过硬件测量，不断测量，并继续测量。真实的数字是唯一的方法。**
- 如果你受到延迟限制（现场产品进行推断），不要进行批处理。
- 如果你使用CPU，请不要进行批处理。
- 如果你正在使用吞吐量（你想要运行一堆静态数据的模型），在GPU上，则：

  - 如果你对sequence_length的大小没有概念（"自然"数据），默认情况下不要批处理，测量并尝试添加它，添加OOM检查以在失败时恢复（如果你不控制sequence_length，则它最终会失败。）
  - 如果你的sequence_length非常规律，那么批处理更有可能非常有趣，测量并将其推至直到获得OOM。
  - GPU越大，批处理越有可能更有趣
- 一旦启用了批处理，请确保你可以很好地处理OOM。

## pipeline块批处理

`zero-shot-classification`和`question-answering`在一个输入中可能会产生多个模型的前向传递，这在正常情况下会导致`batch_size`参数出现问题。

为了避免这个问题，这两个pipeline是有点特殊的，它们是`ChunkPipeline`而不是普通的`Pipeline`。简而言之：

```python
对于预处理的东西，pipeline.preprocess(输入)
对于预处理的东西，pipeline.forward(预处理的东西)
对于模型的输出，pipeline.postprocess(模型的输出)
```

现在变成了：

```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

这对你的代码来说应该是非常透明的，因为pipeline的使用方式是相同的。

这是一个简化的视图，因为pipeline可以自动处理批次！这意味着你无需关心你的输入实际上会触发多少前向传递，你可以独立于输入优化`batch_size`。前一节的注意事项仍然适用。

## pipeline自定义代码

如果要覆盖特定pipeline。

请随时创建问题以处理手头的任务，pipeline的目标是易于使用并支持大多数情况，所以`transformers`也许可以支持你的用例。

如果要尝试简单的方式，可以：

- 子类化你选择的pipeline

```python
class MyPipeline(TextClassificationPipeline):
    def postprocess():
        # 你的代码放在这里
        scores = scores * 100
        # 以及在这里


my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# 或者如果你使用*pipeline*函数，那么：
my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)
```

这样可以使你能够进行想要的所有自定义代码。

## pipeline实现

[实现新pipeline](../add_new_pipeline.md)

## 音频

可用于音频任务的pipeline包括以下内容。

### AudioClassificationPipeline

[[autodoc]] AudioClassificationPipeline
- __call__
- all

### AutomaticSpeechRecognitionPipeline

[[autodoc]] AutomaticSpeechRecognitionPipeline
- __call__
- all

### TextToAudioPipeline

[[autodoc]] TextToAudioPipeline
- __call__
- all


### ZeroShotAudioClassificationPipeline

[[autodoc]] ZeroShotAudioClassificationPipeline
- __call__
- all

## 计算机视觉

可用于计算机视觉任务的pipeline包括以下内容。

### DepthEstimationPipeline
[[autodoc]] DepthEstimationPipeline
- __call__
- all

### ImageClassificationPipeline

[[autodoc]] ImageClassificationPipeline
- __call__
- all

### ImageSegmentationPipeline

[[autodoc]] ImageSegmentationPipeline
- __call__
- all

### ObjectDetectionPipeline

[[autodoc]] ObjectDetectionPipeline
- __call__
- all

### VideoClassificationPipeline

[[autodoc]] VideoClassificationPipeline
- __call__
- all

### ZeroShotImageClassificationPipeline

[[autodoc]] ZeroShotImageClassificationPipeline
- __call__
- all

### ZeroShotObjectDetectionPipeline

[[autodoc]] ZeroShotObjectDetectionPipeline
- __call__
- all

## 自然语言处理

可用于自然语言处理任务的pipeline包括以下内容。

### ConversationPipeline

[[autodoc]] ConversationPipeline
- __call__
- all

### FillMaskPipeline

[[autodoc]] FillMaskPipeline
- __call__
- all

### NerPipeline

[[autodoc]] NerPipeline

有关所有详细信息，请参阅[`TokenClassificationPipeline`]。

### QuestionAnsweringPipeline

[[autodoc]] QuestionAnsweringPipeline
- __call__
- all

### SummarizationPipeline

[[autodoc]] SummarizationPipeline
- __call__
- all

### TableQuestionAnsweringPipeline

[[autodoc]] TableQuestionAnsweringPipeline
- __call__

### TextClassificationPipeline

[[autodoc]] TextClassificationPipeline
- __call__
- all

### TextGenerationPipeline

[[autodoc]] TextGenerationPipeline
- __call__
- all

### Text2TextGenerationPipeline

[[autodoc]] Text2TextGenerationPipeline
- __call__
- all

### TokenClassificationPipeline

[[autodoc]] TokenClassificationPipeline
- __call__
- all

### TranslationPipeline

[[autodoc]] TranslationPipeline
- __call__
- all

### ZeroShotClassificationPipeline

[[autodoc]] ZeroShotClassificationPipeline
- __call__
- all

## 多模态

可用于多模态任务的pipeline包括以下内容。

### DocumentQuestionAnsweringPipeline

[[autodoc]] DocumentQuestionAnsweringPipeline
- __call__
- all

### FeatureExtractionPipeline

[[autodoc]] FeatureExtractionPipeline
- __call__
- all

### ImageToTextPipeline

[[autodoc]] ImageToTextPipeline
- __call__
- all

### VisualQuestionAnsweringPipeline

[[autodoc]] VisualQuestionAnsweringPipeline
- __call__
- all

## 父类：`Pipeline`

[[autodoc]] Pipeline