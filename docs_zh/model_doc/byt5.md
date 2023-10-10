<!--
版权所有2021年The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），除非符合许可证的规定，否则不得使用此文件。
你可以在http://www.apache.org/licenses/LICENSE-2.0获得许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”的基础上进行分发的，不附带任何形式的明示或暗示保证，也不包括任何担保和条件。详情请参阅许可证，在许可证下限制事项。

⚠️注意，此文件是Markdown格式的，但包含我们doc-builder的特定语法（类似MDX），可能无法在Markdown查看器中正确渲染。

-->

# ByT5

## 概览

ByT5模型在Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel的论文《ByT5: Towards a token-free future with pre-trained byte-to-byte models》中提出。
论文摘要如下：

*大多数常用的预训练语言模型是基于与单词或子词对应的标记序列进行操作的。将文本编码为标记序列需要一个分词器，通常分词器是与模型独立创建的。而可以直接处理原始文本（字节或字符）的无标记模型具有许多优势：它们可以处理任何语言的文本，对噪声更加鲁棒，并通过删除复杂且容易出错的文本预处理流程来减少技术负债。由于字节或字符序列比标记序列要长，过去研究中的无标记模型通常引入了新的模型架构，以分散直接处理原始文本的成本。在本文中，我们展示了标准的Transformer架构可以通过最小修改来处理字节序列。我们仔细评估了在参数数量、训练FLOPs和推理速度方面的权衡，并表明字节级模型与标记级模型具有竞争力。我们还证明了字节级模型在噪声环境下具有更高的鲁棒性，并在对拼写和发音敏感的任务上表现更好。作为我们的贡献的一部分，我们发布了基于T5架构的一组新的预训练字节级Transformer模型，以及我们实验中使用的所有代码和数据。*

该模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。原始代码可以在[这里](https://github.com/google-research/byt5)找到。

ByT5的架构基于T5v1.1模型，因此可以参考[T5v1.1的文档页面](t5v1.1)。它们在输入准备方面有所不同，请参阅下面的代码示例。

由于ByT5是无监督预训练的，单任务微调时使用任务前缀并没有实际优势。如果进行多任务微调，应使用前缀。


### 示例

ByT5可直接在原始UTF-8字节上操作，因此可以在没有分词器的情况下使用：

```python
>>> from transformers import T5ForConditionalGeneration
>>> import torch

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

>>> num_special_tokens = 3
>>> # 模型有3个特殊标记，它们占据了ByT5的输入id 0、1、2。
>>> # => 在将id传递给模型之前，需要将utf-8字符编码左移3位。

>>> input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens

>>> labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens

>>> loss = model(input_ids, labels=labels).loss
>>> loss.item()
2.66
```

对于批量推断和训练，建议使用分词器：

```python
>>> from transformers import T5ForConditionalGeneration, AutoTokenizer

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

>>> model_inputs = tokenizer(
...     ["Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt"
... )
>>> labels_dict = tokenizer(
...     ["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt"
... )
>>> labels = labels_dict.input_ids

>>> loss = model(**model_inputs, labels=labels).loss
>>> loss.item()
17.9
```

与[T5](t5)类似，ByT5在范围掩码去噪任务上进行了训练。然而，由于该模型直接在字符上工作，因此预训练任务有所不同。我们对输入句子“The dog chases a ball in the park.”进行一些字符破坏，并要求ByT5预测它们。

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")

>>> input_ids_prompt = "The dog chases a ball in the park."
>>> input_ids = tokenizer(input_ids_prompt).input_ids

>>> # 请注意，无法直接将"{extra_id_...}"添加到字符串中
>>> # 因为字节分词器会错误地合并这些标记
>>> # 对于ByT5，我们需要直接在字符级别上进行操作
>>> # 与T5不同，ByT5没有使用特殊标记进行掩码，而是使用最终的utf字符id。
>>> # UTF-8由8位表示，ByT5有3个特殊标记。
>>> # => 有259个输入id和掩码token从索引258开始。
>>> # => 掩码为"The dog [258]a ball [257]park."

>>> input_ids = torch.tensor([input_ids[:8] + [258] + input_ids[14:21] + [257] + input_ids[28:]])
>>> input_ids
tensor([[ 87, 107, 104,  35, 103, 114, 106,  35, 258,  35, 100,  35, 101, 100, 111, 111, 257,  35, 115, 100, 117, 110,  49,   1]])

>>> # ByT5每次只生成一个字符，因此在这里我们需要产生更多的输出字符-> 设置`max_length=100`。
>>> output_ids = model.generate(input_ids, max_length=100)[0].tolist()
>>> output_ids
[0, 258, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118, 257,  35, 108, 113,  35, 119, 107, 104,  35, 103, 108, 118, 102, 114, 256, 108, 113,  35, 119, 107, 104, 35, 115, 100, 117, 110,  49,  35,  87, 107, 104,  35, 103, 114, 106, 35, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118,  35, 100,  35, 101, 100, 111, 111,  35, 108, 113, 255,  35, 108, 113,  35, 119, 107, 104,  35, 115, 100, 117, 110,  49]

>>> # ^- 请注意258降至257、256、255

>>> # 现在我们需要在分隔符标记上进行拆分，让我们写一个简短的循环来实现这一点
>>> output_ids_list = []
>>> start_token = 0
>>> sentinel_token = 258
>>> while sentinel_token in output_ids:
...     split_idx = output_ids.index(sentinel_token)
...     output_ids_list.append(output_ids[start_token:split_idx])
...     start_token = split_idx
...     sentinel_token -= 1

>>> output_ids_list.append(output_ids[start_token:])
>>> output_string = tokenizer.batch_decode(output_ids_list)
>>> output_string
['<pad>', 'is the one who does', ' in the disco', 'in the park. The dog is the one who does a ball in', ' in the park.']
```


## ByT5Tokenizer

[[autodoc]] ByT5Tokenizer

有关详细信息，请参见[`ByT5Tokenizer`]。