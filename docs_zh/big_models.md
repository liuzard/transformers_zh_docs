<!--版权 2022 HuggingFace团队。版权所有。

根据Apache许可证第2.0版（“许可证”），您不得在遵循许可证的情况下使用此文件。您可以在以下网址获得许可证的副本：

http：//www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以软件形式分发的软件根据许可证分发
一个“AS IS”基础，没有任何种类的保证或条件，无论是明示的还是暗示的。请参阅许可证以获取
特定语言的详细信息，以及许可证下的限制。

⚠️请注意，该文件以Markdown格式提供，但包含我们的doc-builder（类似于MDX）的特定语法，这可能在您的Markdown查看器中无法正确渲染。

-->

# 实例化一个大模型

当您希望使用一个非常大的预训练模型时，一个挑战是尽量减少RAM的使用。 PyTorch中的常规工作流程如下：

1. 创建具有随机权重的模型。
2. 加载预训练的权重。
3. 将这些预训练的权重放入您的随机模型中。

第1步和第2步都需要在内存中具有完整版本的模型，这在大多数情况下不是问题，但是如果您的模型开始达到几个GigaBytes，这两个副本可能会使您的RAM不足。更糟糕的是，如果您正在使用`torch.distributed`来启动分布式培训，则每个进程将加载预训练模型并将这两个副本存储在RAM中。

<Tip>

请注意，随机创建的模型使用“空”张量进行初始化，在不填充内存的情况下占用内存空间（因此随机值是给定时间内内存块中的任何内容）。根据实例化的模型/参数类型的适当分布来执行随机初始化（例如正常分布），仅在步骤3之后对非初始化的权重进行，以尽可能快地执行！

</Tip>

在本指南中，我们将探讨Transformers提供的解决此问题的解决方案。请注意，这是一个正在积极开发的领域，因此此处解释的API可能在将来稍有变化。

## 分片检查点

自版本4.18.0以来，占用超过10GB空间的模型检查点会自动分片成较小的片段。在执行`model.save_pretrained(save_dir)`时，您将获得几个部分检查点（每个部分的大小都小于10GB）和一个将参数名称与存储它们的文件相映射的索引。

您可以使用`max_shard_size`参数控制分片之前的最大大小，所以为了举例，我们将使用具有小分片大小的普通大小模型：让我们选择传统的BERT模型。

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
```

如果使用[`~PreTrainedModel.save_pretrained`]保存它，您将获得一个包含两个文件的新文件夹：模型的配置和权重：

```py
>>> import os
>>> import tempfile

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir)
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model.bin']
```

现在让我们使用最大分片大小为200MB：

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin', 'pytorch_model.bin.index.json']
```

除了模型的配置之外，我们可以看到三个不同的权重文件和一个`index.json`文件，它是我们的索引。可以使用[`~PreTrainedModel.from_pretrained`]方法完全重新加载此类检查点：

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     new_model = AutoModel.from_pretrained(tmp_dir)
```

这样做对于大型模型的主要优点是，在上述工作流程的第2步中，检查点的每个片段在前一个片段之后加载，将RAM中的内存使用限制为模型大小加上最大分片的大小。

在幕后，索引文件用于确定哪些键在检查点中以及相应权重存储在哪里。我们可以像加载任何json一样加载该索引并得到一个字典：

```py
>>> import json

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     with open(os.path.join(tmp_dir, "pytorch_model.bin.index.json"), "r") as f:
...         index = json.load(f)

>>> print(index.keys())
dict_keys(['metadata', 'weight_map'])
```

目前，metadata只包含模型的总大小。我们计划在将来添加其他信息：

```py
>>> index["metadata"]
{'total_size': 433245184}
```

权重映射是此索引的主要部分，它将每个参数名称（如PyTorch模型`state_dict`中通常找到的）映射到其存储的文件：

```py
>>> index["weight_map"]
{'embeddings.LayerNorm.bias': 'pytorch_model-00001-of-00003.bin',
 'embeddings.LayerNorm.weight': 'pytorch_model-00001-of-00003.bin',
 ...
```

如果要在不使用[`~PreTrainedModel.from_pretrained`]（与完整检查点一样使用`model.load_state_dict（）`）的情况下直接加载此类分片检查点到模型中，您应该使用[`~modeling_utils.load_sharded_checkpoint`]：

```py
>>> from transformers.modeling_utils import load_sharded_checkpoint

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     load_sharded_checkpoint(model, tmp_dir)
```

## 低内存加载

分片检查点减少了上述工作流程的第2步中的内存使用，但为了在低内存设置中使用该模型，我们建议利用基于Accelerate库的工具。

请阅读以下指南以获取更多信息：[使用Accelerate进行大模型加载](main_classes/model#large-model-loading)