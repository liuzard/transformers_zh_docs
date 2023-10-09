<!--版权所有2022年亿欧金融 團隊.版权所有。

根据Apache许可证第2版(the "License")，除非你遵守许可证，否则你不可以使用这个文件。你可以获得许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则以"原样"分布的软件是基于Apache许可的，无任何担保或条件。 
请阅读许可证的具体限制。

⚠️注意：这个文件是Markdown格式的，但包含我们的文档生成器（类似MDX的语法），可能在您的Markdown查看器中无法正确显示。

-->

# MarkupLM

## 概述

MarkupLM模型在[MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document
Understanding](https://arxiv.org/abs/2110.08518)中由Junlong Li，Yiheng Xu，Lei Cui，Furu Wei提出。与原始的BERT不同，MarkupLM模型应用于HTML页面而不是原始文本文档。该模型增加了额外的嵌入层，以提高性能，类似于LayoutLM。

该模型可用于Web页面上的问题回答或Web页面上的信息提取等任务。它在两个重要基准测试中取得了最先进的结果：
- [WebSRC](https://x-lance.github.io/WebSRC/)是一个与Web页面结构阅读理解（类似于SQuAD，但针对Web页面）类似的数据集
- [SWDE](https://www.researchgate.net/publication/221299838_From_one_tree_to_a_forest_a_unified_solution_for_structured_web_data_extraction)，一个用于从Web页面提取信息的数据集

论文的摘要如下：

*多模态文本、布局和图像的预训练在可视化丰富的文档理解(VrDU)方面取得了显著进展，尤其是对于选择固定布局文档，例如扫描文档图像。然而，仍然有大量的数字文档，其中布局信息不是固定的，需要进行互动和动态渲染以用于可视化，从而使现有的基于布局的预训练方法难于应用。在本文中，我们提出了作为HTML/XML其他标记语言的骨干的MarkupLM，用于具有标记语言（如基于HTML/XML的文档）的文档理解任务的预训练，即预训练文本和标记信息。实验结果表明，预训练的MarkupLM在几个文档理解任务上显著优于现有的强基线模型。预训练模型和代码将公开可用。*

提示：
- 除了`input_ids`之外，[`~MarkupLMModel.forward`]还期望有两个额外的输入，即`xpath_tags_seq`和`xpath_subs_seq`。这些是输入序列中每个令牌的XPATH标记和下标。
- 一个可以使用[`MarkupLMProcessor`]来为模型准备所有数据。有关详细信息，请参阅[使用指南](#usage-markuplmprocessor)。
- 可以在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MarkupLM)找到演示笔记本。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/markuplm_architecture.jpg"
alt="drawing" width="600"/> 

<small>MarkupLM架构。取自 <a href="https://arxiv.org/abs/2110.08518">原始论文。</a></small>

这个模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可以在[这里](https://github.com/microsoft/unilm/tree/master/markuplm)找到。

## 使用: MarkupLMProcessor

为模型准备数据的最简单方法是使用[`MarkupLMProcessor`]，它在内部结合了特征提取器（[`MarkupLMFeatureExtractor`]）和标记器（[`MarkupLMTokenizer`]或[`MarkupLMTokenizerFast`]）。特征提取器用于从HTML字符串中提取所有节点和XPATH，并将它们提供给标记器，将它们转换为模型的令牌级输入（例如`input_ids`等）。注意，您仍然可以单独使用特征提取器和标记器，如果您只想处理其中一个任务。

```python
from transformers import MarkupLMFeatureExtractor, MarkupLMTokenizerFast, MarkupLMProcessor

feature_extractor = MarkupLMFeatureExtractor()
tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base")
processor = MarkupLMProcessor(feature_extractor, tokenizer)
```

简而言之，您可以将HTML字符串（和可能的其他数据）提供给[`MarkupLMProcessor`]，它将创建模型所需的输入。在内部，处理器首先使用[`MarkupLMFeatureExtractor`]获取节点和相应的XPATH列表。然后将节点和XPATH提供给[`MarkupLMTokenizer`]或[`MarkupLMTokenizerFast`]，将它们转换为令牌级的`input_ids`、`attention_mask`、`token_type_ids`、`xpath_subs_seq`、`xpath_tags_seq`。可以选择性地向处理器提供节点标签，它们将被转换为令牌级的`labels`。

[`MarkupLMFeatureExtractor`]在内部使用[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)，一个用于从HTML和XML文件中提取数据的Python库。请注意，您仍然可以使用您选择的自定义解析解决方案，并将节点和XPATH自己提供给[`MarkupLMTokenizer`]或[`MarkupLMTokenizerFast`]。

总共有5种用例支持处理器。下面我们列出了所有这些用例。请注意，每个用例都适用于批处理和非批处理输入（我们为非批处理输入进行说明）。

**用例1：网页分类（训练，推理）+令牌分类（推理），parse_html = True**

这是最简单的情况，处理器将使用特征提取器从HTML中获取所有节点和XPATH。

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")

>>> html_string = """
...  <!DOCTYPE html>
...  <html>
...  <head>
...  <title>Hello world</title>
...  </head>
...  <body>
...  <h1>Welcome</h1>
...  <p>Here is my website.</p>
...  </body>
...  </html>"""

>>> # 注意，您还可以在此处添加所有标记器参数，如填充、截断等
>>> encoding = processor(html_string, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq'])
```

**用例2：网页分类（训练，推理）+令牌分类（推理），parse_html=False**

如果已经获取了所有节点和XPATH，就不需要特征提取器。在这种情况下，应将节点和相应的XPATH直接提供给处理器，并确保将`parse_html`设置为`False`。

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
>>> processor.parse_html = False

>>> nodes = ["hello", "world", "how", "are"]
>>> xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
>>> encoding = processor(nodes=nodes, xpaths=xpaths, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq'])
```

**用例3：令牌分类（训练），parse_html=False**

对于令牌分类任务（例如[SWDE](https://paperswithcode.com/dataset/swde)），您还可以提供相应的节点标签，以便训练模型。处理器将把它们转换为令牌级别的`labels`。默认情况下，它只会为一个词的第一个子词进行标记，并将其余的子词标记为-100，这是PyTorch的CrossEntropyLoss的`ignore_index`。如果您希望标记一个词的所有子词，可以将标记器初始化为`only_label_first_subword`设置为`False`。

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
>>> processor.parse_html = False

>>> nodes = ["hello", "world", "how", "are"]
>>> xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
>>> node_labels = [1, 2, 2, 1]
>>> encoding = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq', 'labels'])
```

**用例4：网页问答（推理），parse_html=True**

对于Web页面上的问答任务，您可以向处理器提供问题。默认情况下，处理器将使用特征提取器获取所有节点和XPATH，并创建[CLS]问题令牌[SEP]单词令牌[SEP]。

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")

>>> html_string = """
...  <!DOCTYPE html>
...  <html>
...  <head>
...  <title>Hello world</title>
...  </head>
...  <body>
...  <h1>Welcome</h1>
...  <p>My name is Niels.</p>
...  </body>
...  </html>"""

>>> question = "What's his name?"
>>> encoding = processor(html_string, questions=question, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq'])
```

**用例5：网页问答（推理），parse_html=False**

对于问答任务（例如WebSRC），您可以向处理器提供问题。如果您已经自己提取了所有节点和XPATH，请直接提供它们给处理器。确保将`parse_html`设置为`False`。

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
>>> processor.parse_html = False

>>> nodes = ["hello", "world", "how", "are"]
>>> xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
>>> question = "What's his name?"
>>> encoding = processor(nodes=nodes, xpaths=xpaths, questions=question, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq'])
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)

## MarkupLMConfig

[[autodoc]] MarkupLMConfig
    - all

## MarkupLMFeatureExtractor

[[autodoc]] MarkupLMFeatureExtractor
    - __call__

## MarkupLMTokenizer

[[autodoc]] MarkupLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## MarkupLMTokenizerFast

[[autodoc]] MarkupLMTokenizerFast
    - all

## MarkupLMProcessor

[[autodoc]] MarkupLMProcessor
    - __call__

## MarkupLMModel

[[autodoc]] MarkupLMModel
    - forward

## MarkupLMForSequenceClassification

[[autodoc]] MarkupLMForSequenceClassification
    - forward

## MarkupLMForTokenClassification

[[autodoc]] MarkupLMForTokenClassification
    - forward

## MarkupLMForQuestionAnswering

[[autodoc]] MarkupLMForQuestionAnswering
    - forward
