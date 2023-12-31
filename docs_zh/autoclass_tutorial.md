<!--版权 2022 年 HuggingFace 团队。保留所有权利。

根据 Apache 许可证第2版（“许可证”）的规定，你只有在遵守许可证的情况下才能使用此文件，
你可在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本。除非适用法律要求或书面同意，否则根据许可证分发的软件以“如实”为基础，
不附带任何明示或暗示的担保或条件。请参阅许可证了解具体的管辖权限和限制。

⚠️ 请注意，此文件是用 Markdown 格式编写的，
但包含了我们的文档生成器（类似于 MDX）的特定语法，
可能无法在你的 Markdown 查看器中正确显示。-->

# 使用 AutoClass 加载预训练实例

由于存在如此多不同的 Transformer 架构，为你的检查点创建一个可能是具有挑战性的任务。使库易于使用、简单灵活，是🤗Transformers 核心理念的一部分。`AutoClass` 会自动推断并从给定的检查点中加载正确的架构。`from_pretrained()` 方法允许你快速加载任何架构的预训练模型，因此你无需花费时间和资源从头开始训练模型。生成这种类型的与检查点无关的代码意味着，如果你的代码适用于一个检查点，它将适用于另一个检查点-只要它是针对类似任务进行训练的-即使架构不同。

注意：

>请记住，架构指的是模型的框架，而检查点是给定架构的权重。例如，[BERT](https://huggingface.co/bert-base-uncased)是一个架构，而 `bert-base-uncased` 是一个检查点。"模型" 是一个通用术语，可以指代架构或检查点。


在本教程中，你将学习：

- 加载预训练的分词器。
- 加载预训练的图像处理器。
- 加载预训练的特征提取器。
- 加载预训练的处理器。
- 加载预训练的模型。

## AutoTokenizer

几乎每个 NLP 任务都始于一个分词器。分词器将你的输入转换为可以被模型处理的格式。

使用 [`AutoTokenizer.from_pretrained`] 加载一个分词器：

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

然后按照下面的示例对输入进行分词：

```python
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoImageProcessor

对于视觉任务，图像处理器会将图像处理为正确的输入格式。

```python
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

## AutoFeatureExtractor

对于音频任务，特征提取器会将音频信号处理为正确的输入格式。

使用 [`AutoFeatureExtractor.from_pretrained`] 加载一个特征提取器：

```python
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor

多模态任务需要一个处理器，它结合了两种类型的预处理工具。例如，[LayoutLMV2](model_doc/layoutlmv2.md) 模型需要一个图像处理器来处理图像和一个分词器来处理文本；处理器将两者结合起来。

使用 [`AutoProcessor.from_pretrained`] 加载一个处理器：

```python
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

**1、pytorch写法**

最后，`AutoModelFor` 类让你可以加载一个给定任务的预训练模型（请参阅[这里]（http://www.liuzard.com/model_doc/auto）以获取可用任务的完整列表）。例如，使用 [`AutoModelForSequenceClassification.from_pretrained`] 加载一个用于序列分类的模型：

```python
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

轻松地重复使用相同的检查点，加载不同任务的架构：

```python
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

注意:

>对于 PyTorch 模型，`from_pretrained()` 方法使用 `torch.load()`，它在内部使用 `pickle`，已知存在一些安全问题。通常情况下，不要加载可能来自不受信任的来源或可能被篡改的模型。对于托管在 Hugging Face Hub 上的公共模型，这种安全风险在一定程度上得到了缓解，每次提交时都会进行 [恶意软件扫描]（https://huggingface.co/docs/hub/security-malware）。有关最佳实践（如使用 GPG 进行 [签名提交验证]（https://huggingface.co/docs/hub/security-gpg#signing-commits-with-gpg））请参阅 [Hub 文档]（https://huggingface.co/docs/hub/security）。

>TensorFlow 和 Flax 检查点不受影响，可以使用 `from_pretrained` 方法的 `from_tf` 和 `from_flax` 参数在 PyTorch 架构中加载这些检查点，以避免此问题。

通常情况下，我们建议使用 `AutoTokenizer` 类和 `AutoModelFor` 类加载预训练的模型实例。这样可以确保每次都加载正确的架构。在下一个[教程]（http://www.liuzard.com/preprocessing）中，你将学习如何使用新加载的分词器、图像处理器、特征提取器和处理器对数据集进行预处理，以便进行微调。

**2、tensorflow写法**

最后，`TFAutoModelFor` 类使你可以加载给定任务的预训练模型（有关可用任务的完整列表，请参阅[此处](http://www.liuzard.com/model_doc/auto)）。例如，使用 [`TFAutoModelForSequenceClassification.from_pretrained`] 加载一个用于序列分类的模型：

```python
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

轻松地重复使用相同的检查点，加载不同任务的架构：

```python
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

通常情况下，我们建议使用 `AutoTokenizer` 类和 `TFAutoModelFor` 类加载预训练的模型实例。这样可以确保每次都正确加载架构。在下一个[教程](http://www.liuzard.com/preprocessing)中，你将学习如何使用新加载的分词器、图像处理器、特征提取器和处理器对数据集进行预处理，以进行微调。
