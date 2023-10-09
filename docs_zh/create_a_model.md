<!--版权2022祝大家越来越好，保持开心，免责声明，不要用这个代码哦-->

# 创建自定义架构

[`AutoClass`](model_doc/auto)会自动推断模型架构并下载预训练配置和权重。通常，我们建议使用`AutoClass`来产生与检查点无关的代码。但对于希望对特定模型参数有更多控制的用户来说，可以从几个基类创建一个自定义🤗 Transformers模型。这对于对🤗 Transformers模型进行研究、训练或实验的任何人来说都非常有用。在本指南中，深入了解如何创建没有`AutoClass`的自定义模型。学习如何：

- 加载和自定义模型配置。
- 创建模型架构。
- 为文本创建慢速和快速分词器。
- 为视觉任务创建图像处理器。
- 为音频任务创建特征提取器。
- 为多模式任务创建处理器。

## 配置

[配置](main_classes/configuration)是指模型的特定属性。每个模型配置都有不同的属性；例如，所有NLP模型都具有`hidden_size`、`num_attention_heads`、`num_hidden_layers`和`vocab_size`属性。这些属性指定了构建模型所需的注意力头或隐藏层的数量。

通过访问 [`DistilBertConfig`] 来查看 [DistilBERT](model_doc/distilbert) 的更多详细信息以检查它的属性：

```py
>>> from transformers import DistilBertConfig

>>> config = DistilBertConfig()
>>> print(config)
DistilBertConfig {
  "activation": "gelu",
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

[`DistilBertConfig`] 显示了用于构建基础 [`DistilBertModel`] 的所有默认属性。所有属性都是可自定义的，为实验提供了空间。例如，你可以使用 `activation` 参数尝试其他不同的激活函数，或者使用 `attention_dropout` 参数来设置更高的注意力概率的丢弃比例。

```py
>>> my_config = DistilBertConfig(activation="relu", attention_dropout=0.4)
>>> print(my_config)
DistilBertConfig {
  "activation": "relu",
  "attention_dropout": 0.4,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

预训练模型属性可以在 [`~PretrainedConfig.from_pretrained`] 函数中进行修改：

```py
>>> my_config = DistilBertConfig.from_pretrained("distilbert-base-uncased", activation="relu", attention_dropout=0.4)
```

一旦你对模型配置感到满意，就可以使用 [`~PretrainedConfig.save_pretrained`] 来保存配置。你的配置文件将作为一个 JSON 文件存储在指定的保存目录中：

```py
>>> my_config.save_pretrained(save_directory="./your_model_save_path")
```

要重用配置文件，请使用 [`~PretrainedConfig.from_pretrained`] 加载它：

```py
>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
```

<Tip>

你还可以将配置文件保存为字典，甚至只保存自定义配置属性与默认配置属性之间的差异！有关更多详细信息，请参阅[配置](main_classes/configuration)文档。

</Tip>

## 模型

下一步是创建一个[模型](main_classes/models)。模型（也可以宽泛地称为架构）定义了每个层的工作和操作。例如，使用配置中的`num_hidden_layers`定义了架构。每个模型都共享基类 [`PreTrainedModel`] 和一些常见的方法，如调整输入嵌入的大小和修剪自注意力头。此外，所有模型都是 [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)、[`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 或 [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#module) 的子类。这意味着各自框架的模型可以与其各自框架的用法兼容。

<frameworkcontent>
<pt>
将自定义配置属性加载到模型中：

```py
>>> from transformers import DistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
>>> model = DistilBertModel(my_config)
```

这将创建一个具有随机值而不是预训练权重的模型。在你训练模型之前，你将无法对其进行任何有用的操作。训练是一个昂贵且耗时的过程。通常情况下，建议使用预训练模型，以便更快地获得更好的结果，同时仅使用训练所需资源的一小部分。

使用 [`~PreTrainedModel.from_pretrained`] 创建预训练模型：

```py
>>> model = DistilBertModel.from_pretrained("distilbert-base-uncased")
```

当加载预训练权重时，如果模型由 🤗 Transformers 提供，则会自动加载默认模型配置。但是，如果你愿意，仍然可以替换-某些或全部-默认模型配置属性：

```py
>>> model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=my_config)
```
</pt>
<tf>
将自定义配置属性加载到模型中：

```py
>>> from transformers import TFDistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
>>> tf_model = TFDistilBertModel(my_config)
```

这将创建一个具有随机值而不是预训练权重的模型。在你训练模型之前，你将无法对其进行任何有用的操作。训练是一个昂贵且耗时的过程。通常情况下，建议使用预训练模型，以便更快地获得更好的结果，同时仅使用训练所需资源的一小部分。

使用 [`~TFPreTrainedModel.from_pretrained`] 创建预训练模型：

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
```

当加载预训练权重时，如果模型由 🤗 Transformers 提供，则会自动加载默认模型配置。但是，如果你愿意，仍然可以替换-某些或全部-默认模型配置属性：

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=my_config)
```
</tf>
</frameworkcontent>

### 模型头

此时，你已经有了一个基本的 DistilBERT 模型，它输出 *隐藏状态*。隐藏状态作为输入传递给模型头以产生最终的输出。只要模型支持任务，🤗 Transformers 为每个任务提供了一个不同的模型头（例如，你不能为 DistilBERT 这样的序列到序列任务（如翻译）使用它）。

<frameworkcontent>
<pt>
例如，[`DistilBertForSequenceClassification`] 是一个带有序列分类头的基本 DistilBERT 模型。序列分类头是位于池化输出之上的线性层。

```py
>>> from transformers import DistilBertForSequenceClassification

>>> model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

通过切换到不同的模型头，可以轻松地将此检查点用于另一个任务。对于问题回答任务，你将使用 [`DistilBertForQuestionAnswering`] 模型头。问题回答头与序列分类头类似，只是它是位于隐藏状态输出之上的线性层。

```py
>>> from transformers import DistilBertForQuestionAnswering

>>> model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```
</pt>
<tf>
例如，[`TFDistilBertForSequenceClassification`] 是一个带有序列分类头的基本 DistilBERT 模型。序列分类头是位于池化输出之上的线性层。

```py
>>> from transformers import TFDistilBertForSequenceClassification

>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

通过切换到不同的模型头，可以轻松地将此检查点用于另一个任务。对于问题回答任务，你将使用 [`TFDistilBertForQuestionAnswering`] 模型头。问题回答头与序列分类头类似，只是它是位于隐藏状态输出之上的线性层。

```py
>>> from transformers import TFDistilBertForQuestionAnswering

>>> tf_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```
</tf>
</frameworkcontent>

## 分词器

在使用模型处理文本数据之前，你需要使用一个[分词器](main_classes/tokenizer)将原始文本转换为张量。🤗 Transformers 提供了两种类型的分词器：

- [`PreTrainedTokenizer`]：分词器的 Python 实现。
- [`PreTrainedTokenizerFast`]：来自我们的基于 Rust 的 [🤗 Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/) 库的分词器。由于其 Rust 实现，这种分词器类型在批量分词时速度明显更快。快速分词器还提供了额外的方法，如 *offset mapping*，用于将标记映射到它们的原始单词或字符。

这两种分词器都支持常见的方法，如编码和解码、添加新的标记、管理特殊标记。

<Tip warning={true}>

并非每个模型都支持快速分词器。请查看此[表格](index_zh#supported-frameworks)以检查模型是否支持快速分词器。

</Tip>

如果你训练了自己的分词器，可以根据你的 *词汇* 文件创建一个分词器：

```py
>>> from transformers import DistilBertTokenizer

>>> my_tokenizer = DistilBertTokenizer(vocab_file="my_vocab_file.txt", do_lower_case=False, padding_side="left")
```

重要的是要记住，自定义分词器的词汇将与预训练模型分词器生成的词汇不同。如果你使用预训练模型，你需要使用预训练模型的词汇，否则输入将毫无意义。使用预训练模型的词汇创建一个分词器，使用 [`DistilBertTokenizer`] 类：

```py
>>> from transformers import DistilBertTokenizer

>>> slow_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
```

使用 [`DistilBertTokenizerFast`] 类创建快速分词器：

```py
>>> from transformers import DistilBertTokenizerFast

>>> fast_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
```

<Tip>

默认情况下，[`AutoTokenizer`] 将尝试加载快速分词器。你可以通过在 `from_pretrained` 中设置 `use_fast=False` 来禁用此行为。

</Tip>

## 图像处理器

图像处理器处理视觉输入。它是从基类 [`~image_processing_utils.ImageProcessingMixin`] 继承的。

要使用，创建与你正在使用的模型相关联的图像处理器。例如，如果你在使用 [ViT](model_doc/vit) 进行图像分类，则可以创建一个默认的 [`ViTImageProcessor`]：

```py
>>> from transformers import ViTImageProcessor

>>> vit_extractor = ViTImageProcessor()
>>> print(vit_extractor)
ViTImageProcessor {
  "do_normalize": true,
  "do_resize": true,
  "image_processor_type": "ViTImageProcessor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "size": 224
}
```

<Tip>

如果你不打算进行任何自定义操作，只需使用 `from_pretrained` 方法加载默认图像处理器参数即可。

</Tip>

修改任何 [`ViTImageProcessor`] 参数以创建自定义图像处理器：

```py
>>> from transformers import ViTImageProcessor

>>> my_vit_extractor = ViTImageProcessor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])
>>> print(my_vit_extractor)
ViTImageProcessor {
  "do_normalize": false,
  "do_resize": true,
  "image_processor_type": "ViTImageProcessor",
  "image_mean": [
    0.3,
    0.3,
    0.3
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": "PIL.Image.BOX",
  "size": 224
}
```

## 特征提取器

特征提取器处理音频输入。它是从基类 [`~feature_extraction_utils.FeatureExtractionMixin`] 继承的，并且还可以从 [`SequenceFeatureExtractor`] 类继承以处理音频输入。

要使用，创建与你正在使用的模型相关联的特征提取器。例如，如果你在使用 [Wav2Vec2](model_doc/wav2vec2) 进行音频分类，则可以创建一个默认的 [`Wav2Vec2FeatureExtractor`]：

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor()
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": true,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 16000
}
```

<Tip>

如果你不打算进行任何自定义操作，只需使用 `from_pretrained` 方法加载默认特征提取器参数即可。

</Tip>

修改任何 [`Wav2Vec2FeatureExtractor`] 参数以创建自定义特征提取器：

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor(sampling_rate=8000, do_normalize=False)
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": false,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 8000
}
```


## 处理器

对于支持多模态任务的模型，🤗 Transformers 提供了一个处理器类，方便地将特征提取器和标记器等处理类封装成一个单一对象。例如，让我们为自动语音识别任务（ASR）使用 [`Wav2Vec2Processor`]。ASR 将语音转录为文本，因此你需要一个特征提取器和一个标记器。

创建一个特征提取器来处理音频输入:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> feature_extractor = Wav2Vec2FeatureExtractor(padding_value=1.0, do_normalize=True)
```

创建一个标记器来处理文本输入:

```py
>>> from transformers import Wav2Vec2CTCTokenizer

>>> tokenizer = Wav2Vec2CTCTokenizer(vocab_file="my_vocab_file.txt")
```

将特征提取器和标记器组合在 [`Wav2Vec2Processor`] 中:

```py
>>> from transformers import Wav2Vec2Processor

>>> processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

通过配置和模型这两个基本类，以及一个额外的预处理类（标记器、图像处理器、特征提取器或处理器），你可以创建🤗 Transformers 支持的任何模型。每个基类都是可配置的， allowing you to use the specific attributes you want。 你可以轻松设置一个用于训练的模型或修改一个现有的预训练模型进行微调。