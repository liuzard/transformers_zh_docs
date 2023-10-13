# 🤗Transformers能做什么

🤗Transformers是一个预训练的先进模型库，用于自然语言处理（NLP）、计算机视觉、音频和语音处理任务。该库不仅包含了Transformer模型，还包含了现代卷积网络等非Transformer模型，用于计算机视觉任务。如果你看一下当今最流行的消费产品，如智能手机、应用程序和电视机，可以想到某种形式的深度学习技术在其中发挥作用。想要从你的智能手机拍摄的照片中去除背景对象？这就是全景分割任务的一个例子（如果你还不知道这是什么意思，不用担心，我们将在接下来的章节中进行介绍！）。

本页面提供了关于如何只用三行代码解决🤗Transformers库中的不同语音和音频、计算机视觉和NLP任务的概述！

## 音频

音频和语音处理任务与其他模态有所不同，主要因为音频作为输入是一个连续的信号。与文本不同，原始音频波形不能像句子可以划分为离散的块那样整齐地划分。为了解决这个问题，通常会在固定时间间隔内对原始音频信号进行采样。如果在一个时间间隔内采样更多的样本，采样率就会更高，音频就更接近于原始音频来源。

以往的方法通过从音频中提取有用的特征来对其进行预处理。现在更常见的做法是通过将原始音频波形直接输入到特征编码器中以提取音频表示。这简化了预处理步骤，并允许模型学习最重要的特征。

### 音频分类

音频分类是将音频数据标记为预定义的类别集的任务。这是一个广泛的类别，其中包含许多特定应用，其中一些包括：

* 声景分类：为音频打上一个场景标签（“办公室”，“海滩”，“体育场”）
* 声事件检测：为音频打上一个声音事件的标签（“汽车喇叭”，“鲸鱼呼喊”，“玻璃破裂”）
* 标签化：为包含多个声音的音频打上标签（鸟鸣，会议中的发言人识别）
* 音乐分类：为音乐打上流派标签（“金属”，“嘻哈”，“乡村”）

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
>>> preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

### 自动语音识别

自动语音识别（ASR）将语音转录为文本。由于语音是一种人类通信的自然形式，它是一种最常见的音频任务。今天，ASR系统嵌入在“智能”技术产品中，如扬声器、手机和汽车中。我们可以让虚拟助手播放音乐，设置提醒，并告诉我们天气。

但是，Transformer架构的一个关键挑战是在资源匮乏的语言方面。通过在大量语音数据上进行预训练，并将模型在仅有一个小时的低资源语言标注语音数据上进行调优，仍然可以得到与之前使用100倍更多标注数据训练的ASR系统相比，质量更高的结果。

```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

## 计算机视觉

计算机视觉任务中最早和最成功的任务之一是使用卷积神经网络（CNN）识别邮政编码数字的图像。图像由像素组成，每个像素都有一个数值。这使得将图像表示为像素值矩阵变得容易。每个特定的像素值组合描述了图像的颜色。

计算机视觉任务可以通过两种一般方式解决：

1. 使用卷积操作从低级特征到高级抽象的方式来学习图像的层次特征。
2. 将图像分割为补丁，并使用Transformer逐渐学习每个图像补丁如何相互关联以形成图像。与CNN喜欢的自底向上的方法不同，这有点像从一个模糊图像开始，然后逐渐使其清晰起来。

### 图像分类

图像分类将整个图像从预定义的类别集中进行标记。与大多数分类任务一样，图像分类有许多实际应用，其中一些包括：

* 医疗保健：标记医学图像以检测疾病或监测患者健康状况
* 环境：标记卫星图像以监测森林砍伐、通知野生管理或检测野火
* 农业：标记作物图像以监测植物健康或用于土地利用监测的卫星图像
* 生态学：标记动物或植物物种图像以监测野生动物种群或追踪濒危物种

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="image-classification")
>>> preds = classifier(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.4335, 'label': 'lynx, catamount'}
{'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
{'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
{'score': 0.0239, 'label': 'Egyptian cat'}
{'score': 0.0229, 'label': 'tiger cat'}
```

### 物体检测

与图像分类不同，物体检测识别图像中的多个对象及其在图像中的位置（由边界框定义）。物体检测的一些示例应用包括：

* 自动驾驶车辆：检测日常交通对象，如其他车辆、行人和交通信号灯
* 遥感：灾害监测、城市规划和天气预测
* 缺陷检测：检测建筑物的裂缝或结构损坏，以及制造缺陷

```py
>>> from transformers import pipeline

>>> detector = pipeline(task="object-detection")
>>> preds = detector(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
>>> preds
[{'score': 0.9865,
  'label': 'cat',
  'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
```

### 图像分割

图像分割是一个像素级任务，它将图像中的每个像素分配给一个类别。它与物体检测不同，后者使用边界框来标记和预测图像中的对象，因为分割更加精细。分割可以在像素级别检测对象。图像分割有几种类型：

* 实例分割：除了标记对象的类别外，还标记每个不同对象的实例（“狗-1”，“狗-2”）
* 全景分割：将语义分割和实例分割组合起来；它为每个像素都标记一个语义类别和每个不同对象的实例

分割任务对于自动驾驶车辆来说非常有帮助，可以创建一个像素级地图，用来安全地围绕行人和其他车辆进行导航。在医学成像中也非常有用，任务的细粒度可以帮助识别异常细胞或器官特征。图像分割也可以在电子商务中用于虚拟试衣或通过摄像头在真实世界中叠加对象来创建增强现实体验。

```py
>>> from transformers import pipeline

>>> segmenter = pipeline(task="image-segmentation")
>>> preds = segmenter(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.9879, 'label': 'LABEL_184'}
{'score': 0.9973, 'label': 'snow'}
{'score': 0.9972, 'label': 'cat'}
```

### 深度估计

深度估计预测图像中每个像素相对于摄像机的距离。这是一项对场景理解和重建特别重要的计算机视觉任务。例如，在自动驾驶汽车中，车辆需要了解诸如行人、交通标志和其他车辆之类的物体距离以避免障碍物和碰撞。深度信息还有助于从2D图像构建3D表示，并可用于创建生物结构或建筑物的高质量3D表示。

深度估计有两种方法：

* 立体：通过比较略有不同角度拍摄的两个相同图像来估计深度
* 单眼：从单个图像估计深度

```py
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation")
>>> preds = depth_estimator(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
```

## 自然语言处理

NLP任务是最常见的任务类型之一，因为文本是我们之间进行自然交流的一种自然方式。要将文本转换为模型可识别的格式，需要进行标记化处理。这意味着将一个文本序列分成单独的单词或子单词（标记），然后将这些标记转换为数字。因此，可以将文本序列表示为数字序列，一旦拥有了数字序列，就可以将其输入到模型中解决各种NLP任务！

### 文本分类

与任何模态中的分类任务一样，文本分类从预定义的类别集中对文本序列（可以是句子级、段落或文档级别）进行标记。文本分类有许多实际应用，其中一些包括：

* 情感分析：根据某种极性对文本进行标记，比如`积极`或`消极`，可以在政治、金融和市场营销等领域进行决策支持和支持
* 内容分类：根据文本的主题对其进行标记，以帮助组织和过滤新闻和社交媒体提要中的信息（`天气`，`体育`，`金融`等）

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="sentiment-analysis")
>>> preds = classifier("Hugging Face is the best thing since sliced bread!")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.9991, 'label': 'POSITIVE'}]
```

### 标记分类

在任何NLP任务中，文本通过将文本序列分割成单个单词或子单词进行预处理。这些被称为[标记](/docs_zh/glossary.md#token)。标记分类为每个标记分配一个来自预定义类别集的标签。

标记分类的两种常见类型是：

* 命名实体识别（NER）：根据实体类别（如机构、人物、地点或日期）对标记进行标记。NER在生物医学环境中特别流行，可以对基因、蛋白质和药物名称进行标记。
* 词性标注（POS）：根据其词性（如名词、动词或形容词）对标记进行标记。POS有助于帮助翻译系统理解两个相同单词在语法上的差异（作为名词与动词的“银行”）。

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="ner")
>>> preds = classifier("Hugging Face is a French company based in New York City.")
>>> preds = [
...     {
...         "entity": pred["entity"],
...         "score": round(pred["score"], 4),
...         "index": pred["index"],
...         "word": pred["word"],
...         "start": pred["start"],
...         "end": pred["end"],
...     }
...     for pred in preds
... ]
>>> print(*preds, sep="\n")
{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```

### 问答

问答是另一个基于标记的任务，可以根据问题返回一个答案，有时候带有上下文（开放域），有时候没有上下文（封闭域）。这个任务在我们向虚拟助手询问餐厅是否营业之类的问题时发生。它还可以提供客户或技术支持，并帮助搜索引擎检索你正在寻找的相关信息。

常见的问答类型有两种：

* 抽取型：给定一个问题和一些上下文，答案是模型必须提取的上下文中的一段文本
* 生成型：给定一个问题和一些上下文，答案是由上下文生成的；这种方法由[`Text2TextGenerationPipeline`]处理，而不是下面显示的[`QuestionAnsweringPipeline`]

```py
>>> from transformers import pipeline

### 问题回答

问题回答是一项任务，它从给定的上下文中回答自然语言问题。你可以使用预训练的问题回答模型来回答各种类型的问题，例如问题回答、FAQ、搜索引擎或虚拟助手。给定一个问题和一个上下文，问题回答模型将输出一个答案。

下面是一个使用问题回答模型的示例。

```python
>>> from transformers import pipeline

>>> question_answerer = pipeline(task="question-answering")
>>> preds = question_answerer(
...     question="仓库的名称是什么？",
...     context="仓库的名称是huggingface/transformers",
... )
>>> print(
...     f"得分: {round(preds['score'], 4)}, 起始位置: {preds['start']}, 结束位置: {preds['end']}, 答案: {preds['answer']}"
... )
得分: 0.9327, 起始位置: 30, 结束位置: 54, 答案: huggingface/transformers
```

### 摘要

摘要任务将长文本生成一个较短版本，同时尽可能保留原始文档的大部分含义。摘要是一个序列到序列的任务；它输出比输入更短的文本序列。有很多长篇文档可以摘要，以帮助读者快速了解主要要点。立法法案、法律和金融文件、专利和科学论文就是一些可以摘要以节省读者时间的文件类型，并作为阅读辅助工具的例子。

和问题回答一样，摘要也有两种类型：

* 提取式：识别和提取原始文本中最重要的句子
* 抽象型：从原始文本中生成目标摘要（可能包括输入文档中没有的新单词）；[`SummarizationPipeline`]使用抽象型方法

下面是一个使用摘要模型的示例。

```python
>>> from transformers import pipeline

>>> summarizer = pipeline(task="summarization")
>>> summarizer(
...     "在这项工作中，我们提出了Transformer，这是第一个完全基于注意力机制的序列转导模型，它将在编码器-解码器架构中最常用的循环层替换为多头自注意力机制。对于翻译任务，Transformer的训练速度比基于循环或卷积层的架构快得多。在WMT 2014英-德和WMT 2014英-法翻译任务中，我们实现了一个新的最佳结果。在前一项任务中，我们最好的模型甚至优于以前报告的所有组合模型。"
... )
[{'summary_text': 'Transformer是第一个完全基于注意力机制的序列转导模型。它将在编码器-解码器架构中最常用的循环层替换为多头自注意力机制。对于翻译任务，Transformer的训练速度比基于循环或卷积层的架构快得多。'}]
```

### 翻译

翻译将一种语言的文本序列转换为另一种语言。它对帮助来自不同背景的人们相互沟通、帮助翻译内容以扩大受众范围，甚至作为帮助人们学习一门新语言的学习工具具有重要意义。与摘要一样，翻译也是一个序列到序列的任务，即模型接收一个输入序列并返回一个目标输出序列。

在早期，翻译模型主要是单语的，但最近对能够在许多语言对之间进行翻译的多语言模型越来越感兴趣。

下面是一个使用翻译模型的示例。

```python
>>> from transformers import pipeline

>>> text = "将英语翻译成法语：Hugging Face是一个面向机器学习的基于社区的开源平台。"
>>> translator = pipeline(task="translation", model="t5-small")
>>> translator(text)
[{'translation_text': "Hugging Face est une plateforme open-source basée sur la communauté pour l'apprentissage automatique."}]
```

### 语言建模

语言建模是一项任务，用于预测文本序列中的一个单词。它已成为非常流行的NLP任务，因为预训练的语言模型可以用于许多其他下游任务的微调。近来，对于可以实现零样本或少样本学习的大型语言模型（LLM）的兴趣越来越高。这意味着模型可以解决它没有明确训练的任务！语言模型可以用于生成流畅和令人信服的文本，但你需要小心，因为文本可能并不总是准确的。

有两种类型的语言建模：

* 因果型：模型的目标是预测序列中的下一个标记，未来标记被mask

    ```python
    >>> from transformers import pipeline

    >>> prompt = "Hugging Face是一个面向机器学习的基于社区的开源平台。"
    >>> generator = pipeline(task="text-generation")
    >>> generator(prompt)  # doctest: +SKIP
    ```

* 掩码型：模型的目标是预测序列中的一个掩码标记，可以完全访问序列中的标记
    
    ```python
    >>> text = "Hugging Face是一个面向机器学习的基于社区的<mask>平台。"
    >>> fill_mask = pipeline(task="fill-mask")
    >>> preds = fill_mask(text, top_k=1)
    >>> preds = [
    ...     {
    ...         "score": round(pred["score"], 4),
    ...         "token": pred["token"],
    ...         "token_str": pred["token_str"],
    ...         "sequence": pred["sequence"],
    ...     }
    ...     for pred in preds
    ... ]
    >>> preds
    [{'score': 0.2236,
      'token': 1761,
      'token_str': ' platform',
      'sequence': 'Hugging Face是一个面向机器学习的基于社区的平台。'}]
    ```

## 多模态

多模态任务要求模型处理多种数据模态（文本、图像、音频、视频）以解决特定问题。图像描述是一个多模态任务的示例，其中模型以图像作为输入，并输出描述图像或图像某些属性的文本序列。

尽管多模态模型处理不同的数据类型或模态，但在内部，预处理步骤帮助模型将所有数据类型转换为嵌入（向量或包含有关数据的有意义信息的数字列表）。对于像图像描述这样的任务，模型学习图像嵌入和文本嵌入之间的关系。

### 文档问题回答

文档问题回答是一种从文档中回答自然语言问题的任务。与基于标记的问题回答任务不同，文档问题回答任务接受一个文档的图像作为输入，并搭配一个关于文档的问题，返回一个回答。文档问题回答可以用于解析结构化文档并从中提取关键信息。在下面的例子中，可以从收据中提取总金额和找零金额。

```python
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> url = "https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/2/image/image.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> doc_question_answerer = pipeline("document-question-answering", model="magorshunov/layoutlm-invoices")
>>> preds = doc_question_answerer(
...     question="总金额是多少？",
...     image=image,
... )
>>> preds
[{'score': 0.8531, 'answer': '17,000', 'start': 4, 'end': 4}]
```

希望这个页面能为你提供关于每种模态中所有任务类型的更多背景信息，并且这些任务的实际重要性。在下一个[部分](tasks_explained.md)中，你将了解到**如何** 🤗使用 Transformers 来解决这些任务。