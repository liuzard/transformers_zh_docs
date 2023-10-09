<!--版权2022年HuggingFace团队。版权所有。

根据Apache License，版本2.0（“许可证”）的规定，除非符合许可证的规定，否则不得使用此文件。
你可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除non适用法律要求或书面同意外，软件按“按原样”分发，
不附带任何明示或暗示的条件和保证。请参阅许可证以获取有关
特定语言下限制和特殊条款的保

⚠️请注意，此文件使用Markdown格式，但包含了特定的语法，我们的doc-builder（类似于MDX）无法正确渲染。-->

# 多选题

[[open-in-colab]]

多选题类似于问答题，不同之处在于上下文中除了提供一个问题，还提供了若干个候选答案，模型的任务是选择正确答案。

本指南将向你展示如何进行以下操作：

1. 对[SWAG](https://huggingface.co/datasets/swag)数据集的`regular`配置使用[BERT](https://huggingface.co/bert-base-uncased)进行微调，以选择最佳答案。
2. 使用你微调的模型进行推理。

<Tip>
本教程中所示任务支持以下模型架构：

<!--此提示由`make fix-copies`自动生成，不要手动填写！-->

[ALBERT](../model_doc/albert), [BERT](../model_doc/bert), [BigBird](../model_doc/big_bird), [CamemBERT](../model_doc/camembert), [CANINE](../model_doc/canine), [ConvBERT](../model_doc/convbert), [Data2VecText](../model_doc/data2vec-text), [DeBERTa-v2](../model_doc/deberta-v2), [DistilBERT](../model_doc/distilbert), [ELECTRA](../model_doc/electra), [ERNIE](../model_doc/ernie), [ErnieM](../model_doc/ernie_m), [FlauBERT](../model_doc/flaubert), [FNet](../model_doc/fnet), [Funnel Transformer](../model_doc/funnel), [I-BERT](../model_doc/ibert), [Longformer](../model_doc/longformer), [LUKE](../model_doc/luke), [MEGA](../model_doc/mega), [Megatron-BERT](../model_doc/megatron-bert), [MobileBERT](../model_doc/mobilebert), [MPNet](../model_doc/mpnet), [MRA](../model_doc/mra), [Nezha](../model_doc/nezha), [Nyströmformer](../model_doc/nystromformer), [QDQBert](../model_doc/qdqbert), [RemBERT](../model_doc/rembert), [RoBERTa](../model_doc/roberta), [RoBERTa-PreLayerNorm](../model_doc/roberta-prelayernorm), [RoCBert](../model_doc/roc_bert), [RoFormer](../model_doc/roformer), [SqueezeBERT](../model_doc/squeezebert), [XLM](../model_doc/xlm), [XLM-RoBERTa](../model_doc/xlm-roberta), [XLM-RoBERTa-XL](../model_doc/xlm-roberta-xl), [XLNet](../model_doc/xlnet), [X-MOD](../model_doc/xmod), [YOSO](../model_doc/yoso)

<!--生成提示的结尾-->

</Tip>

开始之前，请确保你已安装所有必需的库：

```bash
pip install transformers datasets evaluate
```

我们建议登录你的Hugging Face账户，这样你可以上传并与社区共享你的模型。在提示时，输入你的令牌登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载SWAG数据集

首先，从🤗数据集库加载SWAG数据集的`regular`配置：

```py
>>> from datasets import load_dataset

>>> swag = load_dataset("swag", "regular")
```

然后查看一个示例：

```py
>>> swag["train"][0]
{'ending0': 'passes by walking down the street playing their instruments.',
 'ending1': 'has heard approaching them.',
 'ending2': "arrives and they're outside dancing and asleep.",
 'ending3': 'turns the lead singer watches the performance.',
 'fold-ind': '3416',
 'gold-source': 'gold',
 'label': 0,
 'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
 'sent2': 'A drum line',
 'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
 'video-id': 'anetv_jkn6uvmqwh4'}
```

尽管看起来字段很多，但实际上很简单：

- `sent1`和`sent2`：这些字段显示了句子的开头，并且如果将它们连接起来，你将得到`startphrase`字段。
- `ending`：为句子的可能结尾提供了一些建议，但只有一个是正确答案。
- `label`：标识正确的句子结尾。

## 预处理

接下来，加载BERT tokenizer来处理句子的开头和四个可能的结尾：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

你要创建的预处理函数需要：

1. 复制`sent1`字段的四个副本，并将每个副本与`sent2`组合以重新创建句子的开头。
2. 将`sent2`与四个可能的句子结尾组合。
3. 扁平化这两个列表，以便对它们进行分词，然后在分词后重新给它们定义形状，使每个示例都有相应的`input_ids`，`attention_mask`和`labels`字段。

```py
>>> ending_names = ["ending0", "ending1", "ending2", "ending3"]


>>> def preprocess_function(examples):
...     first_sentences = [[context] * 4 for context in examples["sent1"]]
...     question_headers = examples["sent2"]
...     second_sentences = [
...         [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
...     ]

...     first_sentences = sum(first_sentences, [])
...     second_sentences = sum(second_sentences, [])

...     tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
...     return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
```

使用🤗数据集的[`~datasets.Dataset.map`]方法将预处理函数应用于整个数据集，通过将`batched=True`设置为同时处理数据集的多个元素，可以加快`map`函数的处理速度：

```py
tokenized_swag = swag.map(preprocess_function, batched=True)
```

🤗 Transformers没有适用于多选题的数据整理器，因此你需要修改[`DataCollatorWithPadding`]以创建一批示例。在整理过程中，将句子动态填充到批处理中的最长长度，而不是将整个数据集填充到最大长度。

`DataCollatorForMultipleChoice`对所有模型输入进行扁平化、填充，然后恢复结果：

<frameworkcontent>
<pt>
```py
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import torch


>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="pt",
...         )

...         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
...         batch["labels"] = torch.tensor(labels, dtype=torch.int64)
...         return batch
```
</pt>
<tf>
```py
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import tensorflow as tf


>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="tf",
...         )

...         batch = {k: tf.reshape(v, (batch_size, num_choices, -1)) for k, v in batch.items()}
...         batch["labels"] = tf.convert_to_tensor(labels, dtype=tf.int64)
...         return batch
```
</tf>
</frameworkcontent>

## 评估

在训练过程中包括一个指标通常有助于评估模型的性能。你可以使用🤗评估库快速加载一个评估方法。对于这个任务，加载[accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)指标（请参阅🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour)以了解更多有关加载和计算指标的信息）：

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

然后创建一个函数，将你的预测和标签传递给[`~evaluate.EvaluationModule.compute`]以计算准确性：

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

现在你的`compute_metrics`函数已经准备好了，当设置训练时将返回它。

## 训练
<frameworkcontent>
<pt>
<Tip>

如果你对使用[`Trainer`]微调模型不熟悉，请查看[这里](../training.md#train-with-pytorch-trainer)的基本教程。

</Tip>

现在，你可以开始训练模型了！使用[`AutoModelForMultipleChoice`]加载BERT：

```py
>>> from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

>>> model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
```

此时，只剩下三个步骤：

1. 在[`TrainingArguments`]中定义你的训练超参数。唯一需要的参数是`output_dir`，它指定保存你的模型的位置。通过设置`push_to_hub=True`，你将该模型上传到Hub（你需要登录Hugging Face以上传你的模型）。在每个epoch结束时，[`Trainer`]将评估准确性并保存训练检查点。
2. 将训练参数与模型、数据集、tokenizer、数据整理器和`compute_metrics`函数一起传递给[`Trainer`]。
3. 调用[`~Trainer.train`]进行微调。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_swag_model",
...     evaluation_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_swag["train"],
...     eval_dataset=tokenized_swag["validation"],
...     tokenizer=tokenizer,
...     data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

完成训练后，使用[`~transformers.Trainer.push_to_hub`]方法将模型推送到Hub，以便每个人都可以使用你的模型：

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

如果你对使用Keras微调模型不熟悉，请查看[这里](../training.md#train-a-tensorflow-model-with-keras)的基本教程。

</Tip>
在TensorFlow中微调模型，首先设置一个优化器函数、学习率计划和一些训练超参数：

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 2
>>> total_train_steps = (len(tokenized_swag["train"]) // batch_size) * num_train_epochs
>>> optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

然后，使用[`TFAutoModelForMultipleChoice`]加载BERT：

```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
```

使用[`~transformers.TFPreTrainedModel.prepare_tf_dataset`]将数据集转换为`tf.data.Dataset`格式：

```py
>>> data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_swag["train"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_swag["validation"],
...     shuffle=False,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )
```

使用[`compile`](https://keras.io/api/models/model_training_apis/#compile-method)为训练配置模型。注意，Transformer模型都有一个默认的与任务相关的损失函数，因此你不需要指定损失函数，除非你想要使用其他的：

```py
>>> model.compile(optimizer=optimizer)  # 没有损失参数！
```

在开始训练之前的最后两件事是从预测中计算准确性，并提供一种将模型上传到Hub的方法。这两个都可以使用[Keras回调](../main_classes/keras_callbacks)来完成。

将`compute_metrics`函数传递给[`~transformers.KerasMetricCallback`]：

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

在[`~transformers.PushToHubCallback`]中指定要推送模型和tokenizer的位置：

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(push_to_hub_model_id="your-model-id", push_to_hub_organization="your-organization")
```

运行[`model.fit`](https://keras.io/api/models/model_training_apis/#fit-method)开始训练：

```py
>>> model.fit(
...     tf_train_set,
...     epochs=num_train_epochs,
...     callbacks=[metric_callback, push_to_hub_callback],
...     validation_data=tf_validation_set,
... )
```

一旦训练完成，使用[`push_to_hub_callback`](https://huggingface.co/docs/datasets/package_reference/main_classes/transformers.PushToHubCallback)方法将你的模型和tokenizer推送到Hub，以便每个人都可以使用你的模型和tokenizer。

```py
>>> model.push_to_hub(push_to_hub_organization="your-organization")
```
</tf>
</frameworkcontent>

```markdown
>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model",
...     tokenizer=tokenizer,
... )
```

然后将你的回调函数打包在一起：

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，你准备好开始训练模型了！使用你的训练和验证数据集，指定训练轮数和回调函数来微调模型，调用 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) 方法：

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=callbacks)
```

训练完成后，你的模型会自动上传到 Hub，这样每个人都可以使用它！
</tf>
</frameworkcontent>


<Tip>

如果想要更深入地了解如何对模型进行多项选择的微调，请参考相应的[PyTorch笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)或[TensorFlow笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)。

</Tip>

## 推理

太好了，现在你已经对模型进行了微调，可以用它进行推理了！

编写一些文本和两个候选答案：

```py
>>> prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
>>> candidate1 = "The law does not apply to croissants and brioche."
>>> candidate2 = "The law applies to baguettes."
```

<frameworkcontent>
<pt>
对每个提示和候选答案对进行标记化，并返回PyTorch张量。同时你还需要创建一些`labels`：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
>>> labels = torch.tensor(0).unsqueeze(0)
```

将输入数据和`labels`传递给模型，并返回`logits`：

```py
>>> from transformers import AutoModelForMultipleChoice

>>> model = AutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
>>> logits = outputs.logits
```

获取具有最高概率的类别：

```py
>>> predicted_class = logits.argmax().item()
>>> predicted_class
'0'
```
</pt>
<tf>
对每个提示和候选答案对进行标记化，并返回TensorFlow张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="tf", padding=True)
```

将输入数据传递给模型，并返回`logits`：

```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in inputs.items()}
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

获取具有最高概率的类别：

```py
>>> predicted_class = int(tf.math.argmax(logits, axis=-1)[0])
>>> predicted_class
'0'
```
</tf>
</frameworkcontent>
```