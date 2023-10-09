<!--版权所有2020年HuggingFace团队保留。

根据Apache License，Version 2.0（“许可证”）许可，除非符合许可证条款，否则你不得使用此文件。你可以在下面获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意软件按“原样”分发，不附带任何形式的担保或条件。详细了解许可证中的特定语言，请参阅许可证。

⚠️请注意，此文件采用Markdown格式，但包含用于我们的文档构建器（类似于MDX）的特定语法，在你的Markdown查看器中可能无法正确渲染。-->

# 使用🤗 Tokenizers的分词器

[`PreTrainedTokenizerFast`](https://huggingface.co/docs/tokenizer/installation)依赖于[🤗 Tokenizers](https://huggingface.co/docs/tokenizers)库。从🤗 Tokenizers库获取的分词器可以很简单地加载到🤗 Transformers中。

首先，让我们通过几行代码创建一个虚拟分词器：

```python
>>> from tokenizers import Tokenizer
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import BpeTrainer
>>> from tokenizers.pre_tokenizers import Whitespace

>>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
>>> trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

>>> tokenizer.pre_tokenizer = Whitespace()
>>> files = [...]
>>> tokenizer.train(files, trainer)
```

现在我们有一个训练好的分词器。我们可以在当前运行时继续使用它，或将其保存为JSON文件以备将来使用。

## 直接从分词器对象加载

让我们看看如何在🤗 Transformers库中利用这个分词器对象。[`PreTrainedTokenizerFast`](https://huggingface.co/docs/tokenizers/quicktour#loading-the-tokenizer)类可以通过接受初始化的*tokenizer*对象作为参数来轻松实例化：

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

现在，这个对象可以与🤗 Transformers分词器共享的所有方法一起使用！请前往[分词器页面](main_classes/tokenizer)了解更多信息。

## 从JSON文件加载

为了从JSON文件加载分词器，让我们首先保存我们的分词器：

```python
>>> tokenizer.save("tokenizer.json")
```

我们保存这个文件的路径可以通过`tokenizer_file`参数传递给[`PreTrainedTokenizerFast`](https://huggingface.co/docs/tokenizers/quicktour#loading-the-tokenizer)初始化方法：

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

现在，这个对象可以与🤗 Transformers分词器共享的所有方法一起使用！请前往[分词器页面](main_classes/tokenizer)了解更多信息。