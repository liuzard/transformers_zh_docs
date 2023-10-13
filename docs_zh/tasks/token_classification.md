<!--版权所有2022 The HuggingFace Team。保留所有权利。

根据Apache License, Version 2.0（“许可证”）许可; 在遵守许可证的情况下，你不得使用此文件。你可以在下面的位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面约定，否则根据许可证分发的软件是基于“按原样”分发的基础上，不附带任何形式的明示或暗示担保。详细了解权限和限制，请参阅许可证。

⚠️请注意，此文件为Markdown文件，但包含我们的文档生成器（类似于MDX）的特定语法，这可能无法在Markdown查看器中正确渲染。

-->

# Token分类

[[在colab中打开]]

<Youtube id="wVHdVlPScxA"/>

Token分类为句子中的每个标记分配一个标签。最常见的Token分类任务之一是命名实体识别（NER）。NER旨在为句子中的每个实体（如人、位置或组织）找到一个标签。

本指南将向你展示如何：

1. 使用[DistilBERT](https://huggingface.co/distilbert-base-uncased)对[WNUT 17](https://huggingface.co/datasets/wnut_17)数据集进行微调，以检测新的实体。
2. 使用微调后的模型进行推理。

<Tip>
本教程中所示的任务由以下模型架构支持：

<!--此提示由 `make fix-copies` 自动生成，请勿手动填写!-->

[ALBERT](../model_doc/albert), [BERT](../model_doc/bert), [BigBird](../model_doc/big_bird), [BioGpt](../model_doc/biogpt), [BLOOM](../model_doc/bloom), [BROS](../model_doc/bros), [CamemBERT](../model_doc/camembert), [CANINE](../model_doc/canine), [ConvBERT](../model_doc/convbert), [Data2VecText](../model_doc/data2vec-text), [DeBERTa](../model_doc/deberta), [DeBERTa-v2](../model_doc/deberta-v2), [DistilBERT](../model_doc/distilbert), [ELECTRA](../model_doc/electra), [ERNIE](../model_doc/ernie), [ErnieM](../model_doc/ernie_m), [ESM](../model_doc/esm), [Falcon](../model_doc/falcon), [FlauBERT](../model_doc/flaubert), [FNet](../model_doc/fnet), [Funnel Transformer](../model_doc/funnel), [GPT-Sw3](../model_doc/gpt-sw3), [OpenAI GPT-2](../model_doc/gpt2), [GPTBigCode](../model_doc/gpt_bigcode), [GPT Neo](../model_doc/gpt_neo), [GPT NeoX](../model_doc/gpt_neox), [I-BERT](../model_doc/ibert), [LayoutLM](../model_doc/layoutlm), [LayoutLMv2](../model_doc/layoutlmv2), [LayoutLMv3](../model_doc/layoutlmv3), [LiLT](../model_doc/lilt), [Longformer](../model_doc/longformer), [LUKE](../model_doc/luke), [MarkupLM](../model_doc/markuplm), [MEGA](../model_doc/mega), [Megatron-BERT](../model_doc/megatron-bert), [MobileBERT](../model_doc/mobilebert), [MPNet](../model_doc/mpnet), [MPT](../model_doc/mpt), [MRA](../model_doc/mra), [Nezha](../model_doc/nezha), [Nyströmformer](../model_doc/nystromformer), [QDQBert](../model_doc/qdqbert), [RemBERT](../model_doc/rembert), [RoBERTa](../model_doc/roberta), [RoBERTa-PreLayerNorm](../model_doc/roberta-prelayernorm), [RoCBert](../model_doc/roc_bert), [RoFormer](../model_doc/roformer), [SqueezeBERT](../model_doc/squeezebert), [XLM](../model_doc/xlm), [XLM-RoBERTa](../model_doc/xlm-roberta), [XLM-RoBERTa-XL](../model_doc/xlm-roberta-xl), [XLNet](../model_doc/xlnet), [X-MOD](../model_doc/xmod), [YOSO](../model_doc/yoso)

<!--自动生成的提示结束-->

</Tip>

开始之前，请确保已安装所有必要的库：

```bash
pip install transformers datasets evaluate seqeval
```

我们建议你登录到你的Hugging Face账户，这样你可以上传和共享你的模型给社区。提示输入你的token以登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载WNUT 17数据集

首先从🤗Datasets库中加载WNUT 17数据集：

```py
>>> from datasets import load_dataset

>>> wnut = load_dataset("wnut_17")
```

然后看一个示例：

```py
>>> wnut["train"][0]
{'id': '0',
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
}
```

`ner_tags`中的每个数字表示一个实体。将数字转换为其标签名称以了解实体是什么：

```py
>>> label_list = wnut["train"].features[f"ner_tags"].feature.names
>>> label_list
[
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]
```

`ner_tags`中的每个标记前缀字母表示实体的token位置：

- `B-`表示实体的开始。
- `I-`表示token包含在同一个实体中（例如，`State`token是`Empire State Building`实体的一部分）。
- `0`表示该token不对应任何实体。

## 预处理

<Youtube id="iY2AZYdZAr0"/>

下一步是加载DistilBERT分词器以预处理`tokens`字段：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

正如你在上面的示例`tokens`字段中看到的那样，它看起来像已经进行了标记化的输入。但是实际上输入尚未标记化，你需要设置`is_split_into_words=True`将单词标记化为子单词。例如：

```py
>>> example = wnut["train"][0]
>>> tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
>>> tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
>>> tokens
['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living', 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad', 'storm', 'here', 'last', 'evening', '.', '[SEP]']
```

但是，这会添加一些特殊标记`[CLS]`和`[SEP]`，而子词标记会导致输入和标签之间的不匹配。现在，一个对应于单个标签的单个单词可能被分割为两个子词。你需要通过以下方式对齐标记和标签：

1. 使用[`word_ids`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.word_ids)方法将所有标记映射到相应的单词。
2. 将特殊标记`[CLS]`和`[SEP]`的标签设置为`-100`，这样它们将被忽略掉用于PyTorch损失函数的计算（请参见[CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)）。
3. 仅为给定单词的第一个标记进行标记。将同一单词的其他子词分配为`-100`。

以下是你可以创建以对齐标记和标签的函数，并截断序列为不超过DistilBERT的最大输入长度的方法：

```py
>>> def tokenize_and_align_labels(examples):
...     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

...     labels = []
...     for i, label in enumerate(examples[f"ner_tags"]):
...         word_ids = tokenized_inputs.word_ids(batch_index=i)  # 将标记映射到它们对应的单词。
...         previous_word_idx = None
...         label_ids = []
...         for word_idx in word_ids:  # 将特殊标记设置为-100。
...             if word_idx is None:
...                 label_ids.append(-100)
...             elif word_idx != previous_word_idx:  # 仅对给定单词的第一个标记进行标记。
...                 label_ids.append(label[word_idx])
...             else:
...                 label_ids.append(-100)
...             previous_word_idx = word_idx
...         labels.append(label_ids)

...     tokenized_inputs["labels"] = labels
...     return tokenized_inputs
```

要将预处理函数应用于整个数据集，请使用🤗Datasets [`~datasets.Dataset.map`]函数。通过设置`batched=True`可以加速`map`函数，以便一次处理数据集的多个元素：

```py
>>> tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
```

现在使用[`DataCollatorWithPadding`]创建一个示例批次。在整理期间将句子动态填充到批次中的最大长度，而不是将整个数据集填充到最大长度。

<frameworkcontent>
<pt>
```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
```
</tf>
</frameworkcontent>

## 评估

在训练过程中包含度量标准通常有助于评估模型的性能。你可以使用🤗[评估](https://huggingface.co/docs/evaluate/index)库快速加载一个评估方法。对于本任务，请加载[seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval)框架（请参阅🤗Evaluate [快速导览](https://huggingface.co/docs/evaluate/a_quick_tour)以了解有关如何加载和计算度量标准的更多信息）。Seqeval实际上产生了几个分数：精确度（precision）、召回率（recall）、F1和准确度（accuracy）。

```py
>>> import evaluate

>>> seqeval = evaluate.load("seqeval")
```

首先获取NER标签，然后创建一个函数，该函数将你的真实预测和真实标签传递给[`~evaluate.EvaluationModule.compute`]以计算分数：

```py
>>> import numpy as np

>>> labels = [label_list[i] for i in example[f"ner_tags"]]


>>> def compute_metrics(p):
...     predictions, labels = p
...     predictions = np.argmax(predictions, axis=2)

...     true_predictions = [
...         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]
...     true_labels = [
...         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]

...     results = seqeval.compute(predictions=true_predictions, references=true_labels)
...     return {
...         "precision": results["overall_precision"],
...         "recall": results["overall_recall"],
...         "f1": results["overall_f1"],
...         "accuracy": results["overall_accuracy"],
...     }
```

现在你的`compute_metrics`函数已经准备好，当设置训练时将会返回它。

## 训练

在开始训练模型之前，请创建一个预期的ID到标签的映射以及ID到标签的映射`id2label`和`label2id`：

```py
>>> id2label = {
...     0: "O",
...     1: "B-corporation",
...     2: "I-corporation",
...     3: "B-creative-work",
...     4: "I-creative-work",
...     5: "B-group",
...     6: "I-group",
...     7: "B-location",
...     8: "I-location",
...     9: "B-person",
...     10: "I-person",
...     11: "B-product",
...     12: "I-product",
... }
>>> label2id = {
...     "O": 0,
...     "B-corporation": 1,
...     "I-corporation": 2,
...     "B-creative-work": 3,
...     "I-creative-work": 4,
...     "B-group": 5,
...     "I-group": 6,
...     "B-location": 7,
...     "I-location": 8,
...     "B-person": 9,
...     "I-person": 10,
...     "B-product": 11,
...     "I-product": 12,
... }
```

<frameworkcontent>
<pt>
<Tip>

如果你对使用[`Trainer`]对模型进行微调不熟悉，请查看[此处](../training.md#train-with-pytorch-trainer)的基本教程！

</Tip>

现在，你可以开始训练模型了！使用[`AutoModelForTokenClassification`]加载DistilBERT，同时指定期望的标签数量以及标签映射：

```py
>>> from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

>>> model = AutoModelForTokenClassification.from_pretrained(
...     "distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

此时，仅剩下三个步骤：

1. 在[`TrainingArguments`]中定义你的训练超参数。`output_dir`是唯一需要的参数，它指定要保存模型的位置。你可以设置`push_to_hub=True`将模型推送到Hub（上传模型需要登录到Hugging Face）。在每个epoch结束时，[`Trainer`]将评估seqeval分数并保存训练检查点。
2. 将训练参数与模型、数据集、分词器、数据整理器和`compute_metrics`函数一起传递给[`Trainer`]。
3. 调用[`~Trainer.train`]以微调你的模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_wnut_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=2,
...     weight_decay=0.01,
...     evaluation_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_wnut["train"],
...     eval_dataset=tokenized_wnut["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

训练完成后，使用[`~transformers.Trainer.push_to_hub`]方法将你的模型分享到Hub，以便每个人都可以使用你的模型：

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

如果你对使用Keras进行模型微调不熟悉，请查看[此处](../training.md#train-a-tensorflow-model-with-keras)的基本教程！

</Tip>
要在TensorFlow中微调模型，请首先设置一个优化器函数、学习率计划和一些训练超参数：

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 3
>>> num_train_steps = (len(tokenized_wnut["train"]) // batch_size) * num_train_epochs
>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=2e-5,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=0.01,
...     num_warmup_steps=0,
... )
```

然后，你可以使用[`TFAutoModelForTokenClassification`]加载DistilBERT，同时指定期望的标签数量以及标签映射：

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained(
...     "distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

使用[`~transformers.TFPreTrainedModel.prepare_tf_dataset`]将数据集转换为`tf.data.Dataset`格式：

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_wnut["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

```md
将模型配置为使用[`compile`](https://keras.io/api/models/model_training_apis/#compile-method)进行训练。注意，Transformers模型都有一个默认的与任务相关的损失函数，因此除非你想要指定一个，否则不需要再指定了：

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # 没有损失参数！
```

在开始训练之前，还有最后两件事要做，即从预测中计算seqeval分数，并提供将模型上传到Hub的方法。这两件事都是通过使用[Keras回调](../main_classes/keras_callbacks)来完成的。

将你的`compute_metrics`函数传递给[`~transformers.KerasMetricCallback`]：

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

在[`~transformers.PushToHubCallback`]中指定将模型和分词处理器上传到哪：

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_wnut_model",
...     tokenizer=tokenizer,
... )
```

然后将回调捆绑在一起：

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，你可以开始训练模型了！使用训练和验证数据集、训练轮数以及回调函数来调用[`fit`](https://keras.io/api/models/model_training_apis/#fit-method)来微调模型：

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)
```

一旦训练完成，你的模型将自动上传到Hub，这样每个人都可以使用它！
</tf>
</frameworkcontent>

<Tip>

要了解有关如何为token分类微调模型的更详细示例，请参阅相应的[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)或[TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)。

</Tip>

## 推理

太棒了，现在你已经微调了模型，可以用它进行推理了！

选择一些你想要进行推理的文本：

```py
>>> text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
```

尝试使用[`pipeline`]中的模型进行推理是最简单的方法。使用NER实例化一个`pipeline`，并将文本传递给它：

```py
>>> from transformers import pipeline

>>> classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
>>> classifier(text)
[{'entity': 'B-location',
  'score': 0.42658573,
  'index': 2,
  'word': 'golden',
  'start': 4,
  'end': 10},
 {'entity': 'I-location',
  'score': 0.35856336,
  'index': 3,
  'word': 'state',
  'start': 11,
  'end': 16},
 {'entity': 'B-group',
  'score': 0.3064001,
  'index': 4,
  'word': 'warriors',
  'start': 17,
  'end': 25},
 {'entity': 'B-location',
  'score': 0.65523505,
  'index': 13,
  'word': 'san',
  'start': 80,
  'end': 83},
 {'entity': 'B-location',
  'score': 0.4668663,
  'index': 14,
  'word': 'francisco',
  'start': 84,
  'end': 93}]
```

如果需要，你也可以手动复制`pipeline`的结果：

<frameworkcontent>
<pt>
对文本进行分词并返回PyTorch张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

将输入传递给模型并返回`logits`：

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

获取具有最高概率的类别，并使用模型的`id2label`映射将其转换为文本标签：

```py
>>> predictions = torch.argmax(logits, dim=2)
>>> predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
>>> predicted_token_class
['O',
 'O',
 'B-location',
 'I-location',
 'B-group',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'B-location',
 'B-location',
 'O',
 'O']
```
</pt>
<tf>
对文本进行分词并返回TensorFlow张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text, return_tensors="tf")
```

将输入传递给模型并返回`logits`：

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> logits = model(**inputs).logits
```

获取具有最高概率的类别，并使用模型的`id2label`映射将其转换为文本标签：

```py
>>> predicted_token_class_ids = tf.math.argmax(logits, axis=-1)
>>> predicted_token_class = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
>>> predicted_token_class
['O',
 'O',
 'B-location',
 'I-location',
 'B-group',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'B-location',
 'B-location',
 'O',
 'O']
```
</tf>
</frameworkcontent>