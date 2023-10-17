<!--版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）获得许可。除非符合许可证的规定，否则你不得使用此文件。你可以在以下链接中获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按“原样”分发，不附带任何担保或条件，无论是明示的还是默示的。有关许可证下的特定语言限制和限制，请参阅许可证。

⚠️注意，此文件采用Markdown格式，但包含特定于我们doc-builder（类似于MDX）的语法，可能在Markdown查看器中无法正确呈现。-->

# 微调预训练模型

[[open-in-colab]]

使用预训练模型有很多好处。它可以降低计算成本和碳排放，并且无需从头开始训练即可使用最先进的模型。🤗Transformers提供了对各种任务的数千种预训练模型的访问。当你使用预训练模型时，你需要使用特定于任务的数据集训练它。这个过程被称为微调，是一种非常强大的技术。在本教程中，你可以使用特定的深度学习框架对预训练模型进行微调：

* 使用🤗Transformers [`Trainer`] 微调预训练模型。
* 使用Keras和TensorFlow微调预训练模型。
* 使用PyTorch微调预训练模型。

<a id='data-processing'></a>

## 准备数据集

<Youtube id="_BZearw7f0w"/>

在微调预训练模型之前，需下载数据集并准备进行训练。前一篇教程向你展示了如何处理训练数据，现在你将有机会将这些技能投入到实际运用中！

首先加载[Yelp Reviews](https://huggingface.co/datasets/yelp_review_full)数据集：

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

如你所知，你需要一个分词器来处理文本，并使用一种填充和截断策略来处理任意长度的序列。为了一次性处理数据集，你可以使用🤗Datasets的 [`map`](https://huggingface.co/docs/datasets/process.html#map)方法来应用一个预处理函数在整个数据集上：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

如果愿意，你可以创建全量数据集的一个较小子集，这样可以缩短微调的时间：

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## 训练

现在，你可以选择你所用框架的部分进行阅读。你可以使用右侧边栏中的链接跳转到所需部分，如果你想隐藏给定框架的所有内容，只需使用该框架块右上方的按钮！

1、pytorch框架

<Youtube id="nvBXf7s7vTI"/>

## 使用PyTorch Trainer训练

🤗Transformers提供了一个经过优化的 [`Trainer`]类，用于训练🤗Transformers模型，使你能够轻松启动训练，而无需手动编写自己的训练循环。[`Trainer`] API支持多种训练选项和功能，如日志记录、梯度累积和混合精度。

首先，加载模型并指定预期标签数。从 Yelp Review [数据集卡片](https://huggingface.co/datasets/yelp_review_full#data-fields)中，你知道有5个标签：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

注意：

>你会看到一个警告，内容是一些预训练权重未被使用，一些权重被随机初始化的信息。别担心，这是完全正常的！BERT模型的预训练头部被丢弃，替换为随机初始化的分类头。在你的序列分类任务上，新的模型头部将对该任务进行微调，从而将预训练模型的知识转移到它上面。



### 训练超参数

接下来，创建一个包含你可以调整的所有超参数以及激活不同训练选项的标志的 [`TrainingArguments`]类。在本教程中，你可以使用默认的训练 [hyperparameters](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)，你也可以自由地尝试调整这些参数以找到最佳设置。

指定训练检查点保存位置：

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### 评估

[`Trainer`] 不会在训练过程中自动评估模型性能。你需要向 [`Trainer`] 传递一个函数来计算和报告指标。[🤗Evaluate](https://huggingface.co/docs/evaluate/index)库提供了一个简单的 [`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy)函数，你可以使用 [`evaluate.load`]（有关更多信息，请参见此[快速浏览](https://huggingface.co/docs/evaluate/a_quick_tour)）函数加载它：

```py
>>> import numpy as np
>>> import evaluate

>>> metric = evaluate.load("accuracy")
```

对`compute`函数调用[`~evaluate.compute`]计算预测的准确性值之前，需将预测值转换为logits（请记住，所有🤗Transformers模型返回logits）：

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

如果你想在微调过程中监视评估指标，请在训练参数中的`evaluation_strategy`参数中指定在每个epoch结束时报告评估指标：

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
```

### Trainer

使用你的模型、训练参数、训练和测试数据集以及评估函数创建一个 [`Trainer`] 对象：

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

然后通过调用 [`~transformers.Trainer.train`]微调你的模型：

```py
>>> trainer.train()
```

2、tensorflow框架
<a id='keras'></a>

<Youtube id="rnTGBy2ax1c"/>

## 使用Keras训练TensorFlow模型

使用Keras API，你也可以训练基于🤗Transformers的TensorFlow模型！

### 为Keras加载数据

使用Keras API训练🤗Transformers模型时，你需要将数据集转换为Keras可理解的格式。如果你的数据集很小，你可以将整个数据集转换为NumPy数组并传递给Keras。在进行其他更复杂的操作之前，先试试这个简单的方法。

首先，加载一个数据集。我们将使用[GLUE基准测试]中的CoLA数据集，因为这是一个简单的二元文本分类任务，目前只使用训练集。

```py
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  # 目前只使用训练集
```

接下来，加载一个分词器并将数据分词处理为NumPy数组。请注意，标签已经是一个由0和1组成的列表，因此我们可以直接将其转换为NumPy数组而无需进行分词！

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)
# 分词器返回的是一个BatchEncoding，但我们将其转换为Keras的字典型输入
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # 标签已经是一个由0和1组成的数组
```

最后，加载模型并[`compile`](https://keras.io/api/models/model_training_apis/#compile-method)和[`fit`](https://keras.io/api/models/model_training_apis/#fit-method)。请注意，Transformers模型都有一个默认的与任务相关的损失函数，因此你无需指定损失函数，除非你希望指定：

```py
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# 加载和编译模型
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")
# 对于微调transformers来说，较小的学习率通常更好
model.compile(optimizer=Adam(3e-5))  # 不需要指定损失参数！

model.fit(tokenized_data, labels)
```

注意：

>在`compile()`模型时，你不必传递损失参数给模型！Hugging Face模型会根据适用于其任务和模型架构的默认损失来选择一个合适的损失函数，如果参数为空，默认选择合适的损失函数。如果你想覆盖该默认选择，只需指定自己的损失函数即可！

对于小型数据集，这种方法效果很好，但对于大型数据集，你可能会发现开始变得很困难。为什么？因为tokenized数组和标签必须完全加载到内存中，并且由于NumPy不处理“嵌套”数组，所以每个分词处理样本都必须填充为整个数据集中最长样本的长度。这将使你的数组变得更大，并且所有这些填充token也会减慢训练速度！

### 将数据加载为tf.data.Dataset

如果你想避免训练速度变慢，你可以将数据加载为`tf.data.Dataset`。虽然你可以根据需要自己编写`tf.data` pipeline，但我们为此提供了两个方便的方法：

- [`~TFPreTrainedModel.prepare_tf_dataset`]：这是我们在大多数情况下推荐的方法。因为是模型的一个方法，所以它可以检查模型以自动确定哪些列可用作模型输入，并丢弃其他列，从而使得数据集更简单、更高性能。
- [`~datasets.Dataset.to_tf_dataset`]：这个方法更低级一些，当你想完全控制数据集创建方式时很有用；通过明确指定包含的 'columns'和 'label_cols'来指定。

在使用[`~TFPreTrainedModel.prepare_tf_dataset`]之前，你需要将分词器的输出作为列添加到你的数据集中，如以下代码示例所示：

```py
def tokenize_dataset(data):
    # 字典的键将作为列添加到数据集中
    return tokenizer(data["text"])


dataset = dataset.map(tokenize_dataset)
```

请记住，默认情况下，Hugging Face数据集存储在磁盘上，因此这不会增加内存使用！一旦添加了这些列，你就可以从数据集中读取批次，并对每个批次进行填充，这将大大减少与对整个数据集进行填充相比的填充token数量。

```py
>>> tf_dataset = model.prepare_tf_dataset(dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer)
```

请注意，在上面的代码示例中，你需要将分词器传递给`prepare_tf_dataset`，以便在加载批次时正确进行填充。如果数据集中的所有样本都具有相同的长度且不需要填充，则可以跳过此参数。如果你需要执行的操作比仅填充样本复杂（例如，对蒙版语言建模的token进行损坏），那么你可以使用`collate_fn`参数来传递一个函数，该函数在将样本列表转换为批次并应用任何所需的预处理时将被调用。请参阅我们的[示例](https://github.com/huggingface/transformers/tree/main/examples)或[笔记本](https://huggingface.co/docs/transformers/notebooks)以查看此方法的具体应用。

一旦创建了`tf.data.Dataset`，你可以像以前一样编译和训练模型：

```py
model.compile(optimizer=Adam(3e-5))  # 不需要指定损失参数！

model.fit(tf_dataset)
```


<a id='pytorch_native'></a>

## 在原生PyTorch中训练


<Youtube id="Dh9CL8fyG80"/>

[`Trainer`]负责训练循环，并允许在一行代码中微调模型。对于喜欢编写自己的训练循环的用户，你还可以在原生PyTorch中微调🤗Transformers模型。

在这一点上, 你需要重启笔记本或执行以下代码以释放一些内存：

```py
del model
del trainer
torch.cuda.empty_cache()
```

然后，手动对`tokenized_dataset`进行后处理，以准备进行训练。

1. 删除`text`列，因为模型不接受原始文本作为输入：

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. 将`label`列重命名为`labels`，因为模型期望参数名称为`labels`：

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```

3. 设置数据集的格式，以返回PyTorch张量而不是列表：

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

然后，创建数据集的较小子集以加快微调速度，如之前所示：

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

```
### 数据加载

创建 `DataLoader` 来对训练和测试数据集进行批次迭代：

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

使用预期标签数量加载模型：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

### 优化器和学习率调度器

创建一个优化器和学习率调度器来对模型进行微调。让我们使用 PyTorch 中的 [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 优化器：

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

使用 [`Trainer`] 创建默认的学习率调度器：

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

最后，如果你有 GPU，指定 `device` 使用 GPU。否则，使用 CPU 进行训练可能需要几个小时而不是几分钟。

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

注意：

>如果你没有 GPU，则可以使用像 [Colaboratory](https://colab.research.google.com/) 或 [SageMaker StudioLab](https://studiolab.sagemaker.aws/) 这样的托管笔记本来免费访问云 GPU。



太棒了，现在你可以开始训练了！🥳

### 训练循环

为了跟踪训练进度，使用 [tqdm](https://tqdm.github.io/) 库在训练步骤数量上添加一个进度条：

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

就像你在 [`Trainer`] 中添加了一个评估函数一样，当编写自己的训练循环时，你也需要这样做。但是，与在每个 epoch 结束时计算和报告指标不同，这次你将使用 [`~evaluate.add_batch`] 累积所有批次，并在最后计算指标。

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

有关更多微调示例，请参阅：

- [🤗Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) 包括在 PyTorch 和 TensorFlow 中训练常见 NLP 任务的脚本。

- [🤗Transformers Notebooks](notebooks) 包含有关如何针对特定任务在 PyTorch 和 TensorFlow 中微调模型的各种笔记本。
```