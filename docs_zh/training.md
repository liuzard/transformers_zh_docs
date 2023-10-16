<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 微调预训练模型

[[open-in-colab]]

使用预训练模型有很多优势。它降低了计算成本，有利于环境保护，并且无需从头开始训练模型即可使用最先进的模型。🤗Transformers为各种任务提供了成千上万个预训练模型。当你使用预训练模型时，你将其训练到与你的任务相关的数据集上。这就是所谓的微调操作，这是一种非常强大的训练技术。在本教程中，你将使用自己选择的深度学习框架来微调预训练模型：

- 使用🤗Transformers [`Trainer`]来微调预训练模型。
- 在TensorFlow中使用Keras来微调预训练模型。
- 在原生PyTorch中微调预训练模型。

<a id='data-processing'></a>

## 准备数据集

<Youtube id="_BZearw7f0w"/>

在微调预训练模型之前，首先下载数据集并对其进行处理以供训练使用。前面的教程展示了如何处理训练数据，现在你有机会将这些技能付诸实践！

首先加载[Yelp评论](https://huggingface.co/datasets/yelp_review_full)数据集：

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

### DataLoader

创建一个`DataLoader`来读取训练和测试数据集，以便可以遍历数据的批次：

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

使用期望的标签数量加载模型：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

### 优化器和学习率调度器

创建一个优化器和学习率调度器来微调模型。让我们使用PyTorch中的[`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)优化器：

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

从[`Trainer`]中创建默认的学习率调度器：

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

最后，指定使用GPU的`device`，如果有的话。否则，在CPU上进行训练可能需要几个小时而不是几分钟。

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

如果你没有GPU，可以使用像[Colaboratory](https://colab.research.google.com/)或[SageMaker StudioLab](https://studiolab.sagemaker.aws/)这样的托管笔记本来免费访问云GPU。

</Tip>

好了，现在你已经准备好训练了！ 🥳

### 训练循环

为了跟踪你的训练进度，使用[tqdm](https://tqdm.github.io/)库在训练步骤的数量上添加一个进度条：

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k, v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### 评估

就像你在[`Trainer`]中添加了一个评估函数一样，在编写自己的训练循环时，你需要做同样的事情。但是，与在每个epoch的末尾计算和报告指标不同，这次你将使用[`~evaluate.add_batch`]累积所有的批次，并在最后计算指标。

```py
>>> import evaluate

>>> metric = evaluate.load("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k, v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits, dim=-1)
...     metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```

## 其他资源

更多微调示例，请参考：

- [🤗Transformers 示例](https://github.com/huggingface/transformers/tree/main/examples) 包含了在 PyTorch 和 TensorFlow 中训练常见 NLP 任务的脚本。

- [🤗Transformers 笔记本](notebooks) 包含了使用 PyTorch 和 TensorFlow 对特定任务进行模型微调的各种笔记本。
