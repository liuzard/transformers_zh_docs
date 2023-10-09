<!--版权 2020 HuggingFace团队。版权所有。

根据Apache许可证，第2版 (the "License") ；除非符合License，否则不得使用此文件
您可以在以下位置获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以编写的软件在
"AS IS"基础上分发，无论是明示的还是暗示的保证或条件。请参阅许可证，获取
指定语言的特定权限和限制的权限

⚠️请注意，此文件以Markdown格式编写，但包含的特定语法为我writer工具 的语法(类似于MDX)，这些语法在你的Markdown查看器中可能无法正常呈现。

-->

# 术语表

本术语表定义了一般的机器学习和🤗 Transformers术语以帮助您更好地理解文档。

## A

### 注意力掩码

注意力掩码是在将序列批量处理时使用的可选参数。

<Youtube id="M6adb1j2jPI"/>

该参数指示模型应关注哪些标记，而不应关注哪些标记。

例如，考虑下面这两个序列：

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

>>> sequence_a = "This is a short sequence."
>>> sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

>>> encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
>>> encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
```

编码后的版本长度不同：

```python
>>> len(encoded_sequence_a), len(encoded_sequence_b)
(8, 19)
```

因此，我们不能将它们直接放在同一个张量中。第一个序列需要填充到与第二个序列相同的长度，或者第二个序列需要被截断为与第一个序列相同的长度。

在第一种情况下，id列表将被填充索引扩展。我们可以将列表传递给tokenizer，并要求它进行填充，就像这样：

```python
>>> padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
```

我们可以看到在第一句句子的右边增加了零，以使其与第二句句子具有相同的长度：

```python
>>> padded_sequences["input_ids"]
[[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]
```

接下来，可以将其转换为PyTorch或TensorFlow中的张量。注意力掩码是一个二进制张量，指示填充索引的位置，以便模型不关注它们。对于[`BertTokenizer`]，`1`表示应注意的值，而`0`表示填充值。这个注意力掩码在tokenizer返回的字典中的“attention_mask”键下：

```python
>>> padded_sequences["attention_mask"]
[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

### 自编码模型

参见[编码器模型](#encoder-models)和[掩码语言模型](#masked-language-modeling-mlm)。

### 自回归模型

参见[因果语言模型](#causal-language-modeling)和[解码器模型](#decoder-models)。

## B

### 主干网络

主干网络是输出原始隐藏状态或特征的网络(嵌入和层)。通常它与一个[头部](#head)相连，后者接受特征作为输入进行预测。例如，[`ViTModel`]是没有特定头部的主干网络。其他模型也可以使用[`ViTModel`]作为主干网络，例如[DPT](model_doc/dpt)。

## C

### 因果语言模型

一种预训练任务，模型按顺序阅读文本并预测下一个单词。通常通过读取整个句子，并使用模型内部的掩码来隐藏某个时间步长的未来标记来实现。

### 通道

彩色图像由三个通道 - 红色、绿色和蓝色(RGB) - 的某种组合组成，灰度图像只有一个通道。在🤗 Transformers中，通道可以是图像张量的第一个或最后一个维度：[`n_channels`, `height`, `width`]或[`height`, `width`, `n_channels`]。

### 连接主义时间分类(CTC)

一种允许模型在不知道输入和输出如何对齐的情况下进行学习的算法；CTC计算给定输入的所有可能输出的分布，并从中选择最可能的输出。CTC在语音识别任务中经常被使用，因为由于发音速度不同等原因，语音和转录之间并不总是完全对齐。

### 卷积

神经网络中的一种层类型，其中输入矩阵与较小的矩阵(卷积核或过滤器)逐元素相乘，并在新矩阵中求和。这被称为卷积操作，它在整个输入矩阵上重复。每个操作应用于输入矩阵的不同片段。卷积神经网络(CNNs)常用于计算机视觉中。

## D

### 解码器输入ID

这个输入是特定于编码器-解码器模型的，包含将被馈送给解码器的输入ID。这些输入应该用于序列到序列的任务，比如翻译或摘要，并且通常以每个模型特定的方式构建。

大多数编码器-解码器模型(BART，T5)会根据标准从`labels`自动构建`decoder_input_ids`。在这样的模型中，传递`labels`是处理训练的首选方式。

请查看每个模型的文档，了解它们如何处理序列到序列训练中的这些输入ID。

### 解码器模型

也称为自回归模型，解码器模型涉及到一个预训练任务(称为因果语言模型)，在该任务中，模型按顺序阅读文本并预测下一个单词。通常通过读取整个句子，并在某个时间步处使用掩码来隐藏未来标记来实现。

<Youtube id="d_ixlCubqQw"/>

### 深度学习(DL)

使用多层神经网络的机器学习算法。

## E

### 编码器模型

也称为自编码模型，编码器模型接受输入(如文本或图像)并将其转换为被称为嵌入的压缩数值表示形式。编码器模型通常使用[掩码语言建模](#masked-language-modeling-mlm)等技术进行预训练，这些技术会对输入序列的一部分进行掩码，并迫使模型生成更有意义的表示。

<Youtube id="H39Z_720T5s"/>

## F

### 特征提取

将原始数据选择和转换为一组特征的过程，这些特征对机器学习算法更具信息量和有用性。一些特征提取的例子包括将原始文本转换为词嵌入，从图像/视频数据中提取重要特征（如边缘或形状）等。

### 前馈分块

在transformers中，每个残差注意力块的自注意力层通常后跟2个前馈层。
前馈层的中间嵌入大小通常大于模型的隐藏大小(例如，对于`bert-base-uncased`)。

对于长度为`[batch_size，sequence_length]`的输入，存储中间前馈嵌入`[batch_size，sequence_length，config.intermediate_size]`所需的内存占用可能占有很大比例。[《Reformer: The Efficient Transformer》](https://arxiv.org/abs/2001.04451)的作者注意到，由于计算与`sequence_length`维无关，从数学上讲，分别计算前馈层的输出嵌入`[batch_size，config.hidden_size]_0，...，[batch_size，config.hidden_size]_n`，然后连接它们到`[batch_size，sequence_length，config.hidden_size]`，n = `sequence_length`，这样可以以增加的计算时间换取减少的内存使用，但产生的结果在数学上是**等效的**。

对于使用[`apply_chunking_to_forward`]函数的模型，`chunk_size`定义了并行计算的输出嵌入的数量，因此定义了内存和时间复杂度之间的权衡。如果将`chunk_size`设置为0，则不进行前馈分块。

### 微调模型

微调是迁移学习的一种形式，它涉及采用预训练模型，冻结其权重，并用新添加的[模型头部](#head)替换输出层。模型头部根据目标数据集进行训练。

有关详细信息，请参见[用于微调的预训练模型](https://huggingface.co/docs/transformers/training)教程，并了解如何使用🤗 Transformers进行模型微调。

## H

### 头部

模型头部是神经网络的最后一层，它接受原始隐藏状态并将其投影到不同的维度。每个任务都有一个不同的模型头部。例如：

  * [`GPT2ForSequenceClassification`]是基于基础[`GPT2Model`]上的一个序列分类头 - 一个线性层。
  * [`ViTForImageClassification`]是基于[`ViTModel`]上的`CLS`标记的最终隐藏状态上的图像分类头 - 一个线性层。
  * [`Wav2Vec2ForCTC`]是`Wav2Vec2Model`基础上的语言模型头部，带有[CTC](#connectionist-temporal-classification-(CTC))。

## I

### 图像块

以视觉为基础的Transformers模型将图像分割成较小的块，将其线性嵌入，然后将其作为一个序列传递给模型。您可以在其配置中找到模型的`patch_size` - 或分辨率。

### 推断

推断是在训练完成后对新数据评估模型的过程。请参阅[用于推断的Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial)教程，了解如何使用🤗 Transformers进行推断。

### 输入ID

输入ID通常是传递给模型的唯一必需参数。它们是令牌索引，也就是由构建输入序列的标记的数字表示形式。

<Youtube id="VFp38yj8h3A"/>

每个tokenizer的工作方式都不同，但基本机制是相同的。这里有一个使用BERT tokenizer的例子，它是一个[WordPiece](https://arxiv.org/pdf/1609.08144.pdf) tokenizer：

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

>>> sequence = "A Titan RTX has 24GB of VRAM"
```

tokenizer负责将序列拆分为tokenizer词汇表中可用的标记。

```python
>>> tokenized_sequence = tokenizer.tokenize(sequence)
```

标记可以是单词或子词。例如，在这里，"VRAM"不在模型词汇表中，所以它被拆分为 "V"，"RA" 和 "M"。为了表示这些标记不是独立的单词，而是相同单词的一部分，在 "RA" 和 "M" 的前面加入了双哈希前缀：

```python
>>> print(tokenized_sequence)
['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
```

然后，可以将这些标记转换为模型可理解的ID。可以通过直接将句子提供给tokenizer来完成这个操作，它利用[🤗 Tokenizers](https://github.com/huggingface/tokenizers)的Rust实现，以实现最佳性能。

```python
>>> inputs = tokenizer(sequence)
```

tokenizer返回一个带有所有必要参数的字典，以便其对应的模型能够正常工作。标记ID在键"input_ids"下：

```python
>>> encoded_sequence = inputs["input_ids"]
>>> print(encoded_sequence)
[101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]
```

请注意，tokenizer会自动添加一些"特殊令牌"(如果相关模型需要)，这些特殊令牌是模型有时使用的特殊ID。

如果我们解码先前的ID序列，

```python
>>> decoded_sequence = tokenizer.decode(encoded_sequence)
```

我们会看到

```python
>>> print(decoded_sequence)
[CLS] A Titan RTX has 24GB of VRAM [SEP]
```

因为这是[`BertModel`]期望其输入的方式。

## L

### 标签

标签是可选的参数，模型可以使用它来计算损失值。这些标签应该是模型的预期预测结果：它将使用标准损失函数计算其预测值和预期值(标签)之间的损失。

这些标签根据模型头部的不同而不同，例如：

- 对于序列分类模型([`BertForSequenceClassification`])，模型期望具有`(batch_size)`维度的张量，批的每个值对应整个序列的预期标签。
- 对于令牌分类模型([`BertForTokenClassification`])，模型期望具有`(batch_size, seq_length)`维度的张量，每个值对应每个个体令牌的预期标签。
- 对于掩蔽语言建模([`BertForMaskedLM`])，模型期望具有`(batch_size, seq_length)`维度的张量，每个值对应每个个体令牌的预期标签：标签为掩蔽令牌的标记ID，其余值为待忽略的(通常为-100)。
- 对于序列到序列任务([`BartForConditionalGeneration`], [`MBartForConditionalGeneration`])，模型期望具有`(batch_size, tgt_seq_length)`维度的张量，每个值对应每个输入序列的目标序列。在训练期间，BART和T5将在内部生成适当的`decoder_input_ids`和解码器注意力掩码。通常情况下，它们不需要被提供。这不适用于使用编码器-解码器框架的模型。
- 对于图像分类模型([`ViTForImageClassification`])，模型期望具有`(batch_size)`维度的张量，每个值对应每个个体图像的预期标签。
- 对于语义分割模型([`SegformerForSemanticSegmentation`])，模型期望具有`(batch_size, height, width)`维度的张量，每个值对应每个个体像素的预期标签。
- 对于物体检测模型([`DetrForObjectDetection`])，模型期望具有字典的列表，其中包含`class_labels`和`boxes`键，批的每个值对应每个个体图像的预期标签和边界框数量。
- 对于自动语音识别模型([`Wav2Vec2ForCTC`])，模型期望具有`(batch_size, target_length)`维度的张量，每个值对应每个个体令牌的预期标签。
  
<Tip>

每个模型的标签可能不同，所以一定要始终检查每个模型的文档，以了解有关其特定标签的更多信息！

</Tip>

## M

### 屏蔽语言建模（MLM）

预训练任务，模型会看到一段文本的损坏版本，通常是通过随机屏蔽一些标记来实现的，并且模型需要预测原始文本。

### 多模态

将文本与其他类型的输入（例如图像）结合在一起的任务。

## N

### 自然语言生成（NLG）

与生成文本相关的所有任务（例如[与Transformer一起写作](https://transformer.huggingface.co/)，翻译）。

### 自然语言处理（NLP）

一个泛指“处理文本”的术语。

### 自然语言理解（NLU）

与理解文本中的内容相关的所有任务（例如对整个文本和单个单词进行分类）。

## P

### pipeline

在🤗 Transformers中，pipeline是一个抽象，指的是按特定顺序执行的一系列步骤，以预处理和转换数据，并从模型返回预测。pipeline中可能包含一些示例阶段，例如数据预处理、特征提取和归一化。

有关更多详细信息，请参阅[用于推断的pipeline]（https://huggingface.co/docs/transformers/pipeline_tutorial）。

### 像素值

传递给模型的图像的数值表示的张量。像素值的形状为[`batch_size`，`num_channels`，`height`，`width`]，并且是由图像处理器生成的。

### 汇合

将矩阵缩小为更小的矩阵的操作，可以通过选择汇总维度的最大值或平均值来实现。汇合层通常出现在卷积层之间，用于对特征表示进行下采样。

### 位置ID

与将每个标记的位置嵌入到RNN不同，transformer不知道每个标记的位置。因此，模型使用位置ID（`position_ids`）来识别列表中每个标记的位置。

它们是一个可选参数。如果没有将`position_ids`传递给模型，ID将自动创建为绝对位置嵌入。

绝对位置嵌入被选在范围`[0, config.max_position_embeddings - 1]`内。一些模型使用其他类型的位置嵌入，例如正弦位置嵌入或相对位置嵌入。

### 预处理

将原始数据准备为机器学习模型易于使用的格式的任务。例如，文本通常通过标记化进行预处理。要了解其他输入类型的预处理是什么样子，请查看[预处理](https://huggingface.co/docs/transformers/preprocessing)教程。

### 预训练模型

已经在一些数据上进行了预训练的模型（例如全部维基百科）。预训练方法包括自监督目标，可以是阅读文本并尝试预测下一个词（参见[因果语言模型](#因果语言建模)）或者屏蔽一些词并尝试预测它们（参见[屏蔽语言建模](#屏蔽语言建模-mlm)）。

语音和视觉模型有各自的预训练目标。例如，Wav2Vec2是一个在对比任务上进行预训练的语音模型，其要求模型从一组“错误”的语音表示中识别出“真实”的语音表示。另一方面，BEiT是一个在遮挡图像建模任务上进行预训练的视觉模型，它会遮挡一部分图像补丁，并要求模型预测遮挡的补丁（类似于屏蔽语言建模目标）。

## R

### 循环神经网络（RNN）

一种使用循环层处理文本的模型类型。

### 表示学习

机器学习的一个子领域，侧重于学习原始数据的有意义的表示。一些表示学习技术的示例包括词嵌入，自动编码器和生成对抗网络（GAN）。

## S

### 采样率

每秒采样数（音频信号）的赫兹测量。采样率是将连续信号（例如语音）离散化的结果。

### 自注意力

输入的每个元素会找出它们应该注意的其他元素。

### 自监督学习

一类机器学习技术，其中模型从未标记的数据中创建其自己的学习目标。它与[无监督学习](#无监督学习)和[有监督学习](#有监督学习)不同，因为学习过程是有监督的，但不是显式地来自用户。

自监督学习的一个示例是[屏蔽语言建模](#屏蔽语言建模-mlm)，其中模型接收到一些标记被移除的句子，并学习预测缺失的标记。

### 半监督学习

一类广义的机器学习训练技术，利用了少量有标签数据和大量无标签数据来提高模型的准确性，与[有监督学习](#有监督学习)和[无监督学习](#无监督学习)不同。

半监督学习方法的一个示例是“自训练”，模型在有标签数据上进行训练，然后对无标签数据进行预测。模型置信度最高的一部分无标签数据将被添加到有标签数据集并用于重新训练模型。

### 序列到序列（seq2seq）

从输入生成新序列的模型，例如翻译模型或摘要模型（如[Bart](model_doc/bart)或[T5](model_doc/t5)）。

### 步幅

在[卷积](#卷积)或[汇合](#汇合)中，步幅是内核在矩阵上移动的距离。步幅为1表示内核每次移动一个像素，步幅为2表示内核每次移动两个像素。

### 监督学习

一种模型训练形式，直接使用带有标签的数据来纠正和指导模型性能。数据被送入正在训练的模型中，其预测结果与已知标签进行比较。模型根据其预测的错误情况更新权重，并重复该过程以优化模型性能。

## T

### 标记

句子的一部分，通常是一个单词，但也可以是子词（不常见的单词通常会被分割成子词）或标点符号。

### 标记类型ID

某些模型的任务是对句子对进行分类或问答。

<Youtube id="0u3ioSwev3s"/>

这些任务要求将两个不同的序列合并为单个“input_ids”输入，通常使用特殊标记（例如分类器（`[CLS]`）和分隔符（`[SEP]`）标记）来完成。例如，BERT模型将其两个序列输入构建为：

```python
>>> # [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]
```

我们可以使用我们的分词器通过将两个序列作为两个参数（而不是像之前那样作为列表）传递给`tokenizer`来自动生成这样的句子：

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
>>> sequence_a = "HuggingFace is based in NYC"
>>> sequence_b = "Where is HuggingFace based?"

>>> encoded_dict = tokenizer(sequence_a, sequence_b)
>>> decoded = tokenizer.decode(encoded_dict["input_ids"])
```

这将返回：

```python
>>> print(decoded)
[CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
```

这对于某些模型足以理解一个序列在哪里结束，另一个序列在哪里开始。然而，其他模型（如BERT）还会使用标记类型ID（也称为段ID）。它们是表示模型中两种序列的二进制掩码。

分词器将此掩码作为“token_type_ids”条目返回：

```python
>>> encoded_dict["token_type_ids"]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

第一个序列，用于问题的“上下文”，其所有标记都用`0`表示，而第二个序列，对应于“问题”，其所有标记都用`1`表示。

一些模型，如[`XLNetModel`]，使用额外的标记，用`2`来表示。

### 迁移学习

一种技术，涉及使用预训练模型并将其调整为特定于任务的数据集。与从头开始训练模型不同，您可以利用现有模型获得的知识作为起点。这加快了学习过程并减少了所需的训练数据量。

### 变压器

基于自注意的深度学习模型结构。

## U

### 无监督学习

一种模型训练形式，其中提供给模型的数据没有标签。无监督学习技术利用数据分布的统计信息来寻找对任务有用的模式。