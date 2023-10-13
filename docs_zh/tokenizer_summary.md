<!--版权 2020年HuggingFace团队保留所有权利。

根据Apache License，Version 2.0（“许可证”）许可； 除非符合许可证，否则你不得使用此文件。
你可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本

除非适用的法律要求或书面同意，否则根据许可证分发的软件以“原样”分发，
不附带任何明示或暗示的担保或条件。有关许可的更多详细信息，请参阅许可证
在许可证下有关特定语法与我们的文档生成器（类似于MDX）的内容不同的Markdown中的该文件
不能在你的Markdown查看器中正确渲染。

-->

＃分词器摘要

[[open-in-colab]]

在这个页面上，我们将更详细地了解分词。

<Youtube id="VFp38yj8h3A"/>

如我们在[预处理教程](preprocessing.md)中所看到的，对文本进行分词处理是将其拆分为单词或
子词，然后通过查找表将其转换为ID。将单词或子词转换为ID感觉很简单，因此在本摘要中，我们将重点放在将文本拆分为单词或子词上（即对文本进行分词）。
更具体地说，我们将介绍🤗转换器中使用的三种主要类型的分词器：[字节对编码
（BPE）](#byte-pair-encoding)，[WordPiece](#wordpiece)和[SentencePiece]（#sentencepiece），并显示示例
对于哪种模型使用了token器类型。

请注意，在每个模型页面上，你可以查看相关分词器的文档，以了解预训练模型使用的分词器
类型。例如，如果我们查看[`BertTokenizer`]，我们可以看到
该模型使用[WordPiece](#wordpiece)。

## 介绍

将文本拆分为较小的块是一项比它看起来更困难的任务，在这方面有多种方法。例如，让我们看一下句子`“Don't you love 🤗Transformers? We sure do."`

<Youtube id="nhJxYji1aho"/>

将此文本分词处理的一个简单方法是通过空格分割，这将给出：

```
["Don't", "you", "love", "🤗", "Transformers?", "We", "sure", "do."]
```

这是一个明智的第一步，但是如果我们看一下token`"Transformers？"`和`"do."`，我们会发现
标点符号连接到了单词`"Transformer"`和`"do"`，这是不太理想的。我们应该考虑标点符号，以便模型不必学习一个不同的单词表示以及可能的任何一个
可能跟随它的标点符号，这将导致模型必须学习的表示数量暴增。
考虑标点符号，对我们的示例文本进行标记化将会给出：

```
["Don", "'", "t", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

更好了。但是，令人不满意的是分词处理如何处理单词`"Don't"`。`"Don't"` 意味着`"do not"`，所以最好是将其标记化为`["Do", "n't"]`。这是事情开始变得复杂的地方，也是每个模型都有自己的分词器类型的一部分原因。根据我们用于对文本进行分词处理的规则，对于相同的文本生成了不同的分词处理输出。仅当你将经过相同规则分词处理的输入提供给预训练模型时，预训练模型才能得到正确的结果。

[spaCy](https://spacy.io/)和[Moses](http://www.statmt.org/moses/?n=Development.GetStarted)是两个常用的规则
基于分词处理器。将它们应用于我们的示例上，*spaCy*和*Moses*会输出如下结果：

```
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

如所示，这里使用了空格和标点符号分词处理以及基于规则的分词处理。空格和
标点符号分词处理和基于规则的分词处理都是单词分词处理的示例，它是一个宽泛定义的概念
将句子拆分为单词。尽管这是将文本分割为较小块的最直观方法，但是此
分词处理方法可能会导致大文本语料库的问题。在这种情况下，空格和标点符号分词处理
通常会生成一个非常大的词汇表（使用的所有唯一单词和标记的集合）。*例*
[Transformer XL]（model_doc/transformerxl）使用空格和标点符号标记化，导致词汇表大小为267,735！

如此庞大的词汇表大小强制模型具有巨大的嵌入矩阵作为输入和输出层，从而
增加内存和时间复杂性。通常，变压器模型的词汇表大小很少
大于50,000，尤其是如果它们仅在单个语言上进行了预训练。

因此，如果简单的空格和标点符号分词处理是不令人满意的，为什么不仅通过字符进行分词呢？

<Youtube id="ssLq_EK2jLE"/>

虽然字符标记化非常简单，并且可以极大地减少内存和时间复杂性，但是它使模型学习有意义的输入表示更加困难。
*例如*，学习字母`t`的有意义的上下文独立
表示比学习单词`"today"`的上下文独立表示要困难得多。因此，字符分词处理通常与性能损失相伴。因此，为了同时获得最佳
这两个世界，变压器模型使用了字级别和字符级别分词之间的混合，称为**子词**
分词。

## 子词分词

<Youtube id="zHvTiHr506c"/>

子词分词算法依赖于一个原则，即常用单词不应该拆分成更小的
子词，但是罕见的单词应该被分解为有意义的子词。例如，`"annoyingly"`可能是一个
被认为是罕见的单词，并且可以分解为`"annoying"`和`"ly"`。两者`"annoying"`和`"ly"`作为
独立的子词出现的次数比较多，同时`"annoyingly"`的含义也保持不变
“annoying”和“ly”的合成意义。这对于像土耳其语这样的粘聚语言特别有用，你可以通过将子词串连在一起来形成（几乎）任意长的复杂单词。

子词分词允许模型拥有合理的词汇量，同时能够学习有意义的
上下文独立表示。此外，子词分词使模型能够处理以前从未见过的单词，将其分解为已知子词。例如，[`~transformers.BertTokenizer`]将
`"I have a new GPU!"` 分词如下：

```py
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> tokenizer.tokenize("I have a new GPU!")
["i", "have", "a", "new", "gp", "##u", "!"]
```

因为我们正在考虑无大小写的模型，所以首先将句子转换为小写。我们可以看到单词 `["i", "have", "a", "new"]` 出现在分词器的词汇表中，但是单词`"gpu"`没有。因此，
分词器将`"gpu"`分割为已知的子词：`["gp"和"##u"]`。`"##"`表示剩下的标记应该
附加到前一个标记上，没有空格（用于解码或颠倒分词处理）。

另一个例子，[`~transformers.XLNetTokenizer`]如下分词我们之前的示例文本：

```py
>>> from transformers import XLNetTokenizer

>>> tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
>>> tokenizer.tokenize("Don't you love 🤗Transformers? We sure do.")
["▁Don", "'", "t", "▁you", "▁love", "▁", "🤗", "▁", "Transform", "ers", "?", "▁We", "▁sure", "▁do", "."]
```

我们将在查看[SentencePiece](#sentencepiece)时回到那些`"▁"`的含义。正如可以看到的那样，
罕见的单词`"Transformers"`已经分解为更常见的子词`"Transform"`和`"ers"`。

现在让我们来看看不同的子词分词处理算法是如何工作的。请注意，所有这些分词处理
算法都依赖于某种形式的训练，通常是在对应模型将要训练的语料库上进行的训练。

<a id='byte-pair-encoding'></a>

### 字节对编码（BPE）

字节对编码（BPE）是在[神经机器翻译中使用字词单元翻译罕见单词（Sennrich et
al.，2015）](https://arxiv.org/abs/1508.07909)中引入的。BPE依赖于一个预分词器，将训练数据拆分为
单词。预标记可以简单地是空格标记化，例如[GPT-2](model_doc/gpt2)，[RoBERTa](model_doc/roberta)。更高级的预分词处理包括基于规则的分词处理，例如[XLM](model_doc/xlm)，
[FlauBERT](model_doc/flaubert)中的Moses用于大多数语言，或者[GPT](model_doc/gpt)中使用
Spacy和ftfy，以计算训练语料库中每个单词的频率。

在预标记之后，已经创建了一组唯一单词，并且已确定了每个单词在
训练数据中出现的频率。接下来，BPE创建一个基本词汇，其中包含在唯一单词中出现的所有符号，并学习合并规则以从基本词汇的两个符号中形成一个新符号。直到词汇表达到所需的大小为止，不断执行此操作。请注意，所需的词汇表大小是在训练分词器之前定义的超参数。

例如，让我们假设预分词之后，已经确定了以下单词集，包括其频率：

```
（“hug”，10），（“pug”，5），（“pun”，12），（“bun”，4），（“hugs”，5）
```

因此，基本词汇表是`["b"，"g"，"h"，"n"，"p"，"s"，"u"]`。将所有单词拆分为基本词汇的符号，我们获得：

```
（“h” “u” “g”，10），（“p” “u” “g”，5），（“p” “u” “n”，12），（“b” “u” “n”，4），（“h” “u” “g” “s”，5）
```

然后，BPE计算每对可能的符号对的频率，并选择出现频率最高的符号对。在上面的示例中，`"h"`后跟`"u"`的次数为_10 + 5 = 15_次（在“hug”出现的10次中，__`"hugs"`中的5次）。但是，最常见的符号对是后跟的`"u"`，并且后跟的`"g"`，总共出现_10 + 5 + 5 = 20_次。因此，token器学到的第一个合并规则是将所有的`"u"`符号与一个`"g"`符号组合在一起。然后，`"ug"`添加到词汇表中。单词集然后变成

```
（“h”“ug”，10），（“p”“ug”，5），（“p”“u”“n”，12），（“b”“u”“n”，4），（“h”“ug”“s”，5）
```

BPE然后识别出下一个最常见的符号对。它是`"u"`后面跟着`"n"`，出现了16次。 `"u"`和`"n"`合并为`"un"`并添加到词汇表中。下一个最频繁的符号对是`"h"`后面跟着`"ug"`，总共出现15次。再次合并该对，就可以将`"hug"`添加到词汇表中。

在这个阶段，词汇表是`["b"，"g"，"h"，"n"，"p"，"s"，"u"，"ug"，"un"，"hug"]`，而我们的唯一单词集
表示为

```
（“hug”，10），（“p”“ug”，5），（“p”“un”，12），（“b”“un”，4），（“hug”“s”，5）
```

假设字节对编码训练会在这点停止，那么将应用已学习的合并规则到新单词（只要这些新单词中不包含基础词汇中没有的符号）。例如，单词`"bug"`将被标记化为`["b"，"ug"]`，但是`"mug"`将被标记化为`["<unk>"，"ug"]`，因为
符号`"m"`不在基础词汇中。通常情况下，像`"m"`这样的单个字母不会被`"<unk>"`符号替换，因为训练数据通常至少包含每个字母的一次出现，但是对于非常特殊的字符（例如表情符号），可能会发生这种情况。

如前所述，词汇表大小，即基本词汇表大小+合并数量，是要选择的超参数。例如[GPT](model_doc/gpt)的词汇表大小为40,478，因为它们有478个基本字符，并选择在40,000次合并之后停止训练。

#### 字节级BPE

如果*例如*将所有Unicode字符都视为基本字符，则包含所有可能基本字符的基本词汇表可能相当大。为了获得更好的基本词汇表，[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)使用字节
作为基本词汇表，这是强制基本词汇表大小为256的巧妙的技巧，同时确保
包括在词汇中的每个基本字符。通过一些额外的规则来处理标点符号，GPT2的
分词器可以处理每个文本而无需<unk>符号。 [GPT-2](model_doc/gpt)的词汇量
50,257，对应于256字节的基本token，特殊的文本结束标记和使用50,000次合并学到的符号。

<a id='wordpiece'></a>

### WordPiece

WordPiece是用于[BERT](model_doc/bert)、[DistilBERT](model_doc/distilbert)和[Electra](model_doc/electra)的子词分词处理算法。该算法在[日语和韩语
语音搜索（Schuster et al.，2012）](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)中概述，非常类似
BPE。WordPiece首先将词汇表初始化为包含训练数据中的每个字符，并
逐渐学习给定数量的合并规则。与BPE不同，WordPiece不选择最常见
符号对，但选择使其添加到词汇表后训练数据的似然性最大化的符号对。

那么这到底是什么意思？参考之前的示例，最大化训练数据的似然性相当于找到使概率除以其第一个符号后跟其第二个符号的概率最大化的符号对，其中这个最大化量是所有符号对中最大的。*例如*，如果在`"ug"`之后`"u"`的概率分别为`"ug"`和`"u"`的概率除以`"g"`的概率，则只有当该概率高于任何其他符号对的概率时，才会合并`"u"`和`"g"`。直观地说，WordPiece与BPE略有不同，它通过评估由于合并两个符号而_丢失_的内容来确保它的确_值得_。

<a id='unigram'></a>

### Unigram

Unigram是在[Subword Regularization: Improving Neural Network Translation
Models with Multiple Subword Candidates（Kudo，2018）](https://arxiv.org/pdf/1804.10959.pdf)中介绍的子词分词处理算法。与BPE或
WordPiece不同，Unigram将其基本词汇表初始化为大量的符号，然后逐渐修剪每个
符号以获得较小的词汇表。基本词汇表可以对应于所有预分分词处理的单词和
最常见的子字符串。Unigram不直接用于transformers中的任何模型，但与[SentencePiece](#sentencepiece)结合使用。

在每个训练步骤中，Unigram算法根据当前词汇表和unigram语言模型对训练数据定义损失（通常定义为对数似然）。然后，对于词汇表中的每个符号，算法计算如果该符号从词汇表中删除，整体损失会增加多少。Unigram然后删除损失增加最小的p（p通常为10％或20％）百分比的符号，即那些对训练数据整体损失影响最小的符号。该过程重复执行，直到词汇表达到所需大小。Unigram算法始终保留基本字符，以便可以对任何单词进行分词。

由于Unigram不基于合并规则（与BPE和WordPiece相反），训练后的算法有几种方法可以对新文本进行分词。例如，如果训练后的Unigram分词器具有以下词汇表：

```
["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"],
```

`"hugs"`可以被分词为`["hug", "s"]`，`["h", "ug", "s"]`或`["h", "u", "g", "s"]`。那么应该选择哪一个？Unigram在保存词汇表的同时保存了训练语料库中每个标记的概率，以便可以在训练后计算每种可能分词的概率。实际上，该算法只选择最可能的分词方式，但也可以根据概率随机选择可能的分词方式。

这些概率是由分词器在训练时定义的损失来确定的。假设训练数据由单词 \\(x_{1}, \dots, x_{N}\\) 组成，对于单词 \\(x_{i}\\) 的所有可能分词的集合定义为 \\(S(x_{i})\\)，则整体损失定义为

$$\mathcal{L} = -\sum_{i=1}^{N} \log \left ( \sum_{x \in S(x_{i})} p(x) \right )$$

<a id='sentencepiece'></a>

### SentencePiece

迄今为止描述的所有分词算法都有同样的问题：假设输入文本使用空格来分隔单词。然而，并非所有语言都使用空格来分隔单词。解决这个问题的一种可能方法是使用特定语言的预分词器，例如[XLM](model_doc/xlm)使用特定的中文、日文和泰文预分词器。为了更普遍地解决这个问题，[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing (Kudo et al., 2018)](https://arxiv.org/pdf/1808.06226.pdf)将输入视为原始输入流，因此包括空格在内的字符集。然后，它使用BPE或unigram算法构建适当的词汇表。

[`XLNetTokenizer`]例如使用SentencePiece，这也是为什么前面的示例中包含了`"▁"`字符的原因。使用SentencePiece进行解码非常简单，因为所有标记只需连接在一起，而`"▁"`被替换为一个空格。

该库中使用SentencePiece的所有transformer模型包括[ALBERT](model_doc/albert)、[XLNet](model_doc/xlnet)、[Marian](model_doc/marian)和[T5](model_doc/t5)。