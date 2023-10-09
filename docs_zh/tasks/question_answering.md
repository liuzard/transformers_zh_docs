<!--版权所有2022年的HuggingFace团队。保留所有权利。

根据Apache License Version 2.0（“许可证”），除非你遵守许可证规定，否则你不得使用此文件。你可以在以下位置获取许可证的副本

http：//www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于“按原样”分布，没有任何形式的明示或暗示保证。有关许可下限制的详细信息，请参阅许可证。

⚠️ 请注意，此文件为Markdown格式，但包含特定语法，用于我们的doc-builder（类似于MDX），在你的Markdown查看器中可能无法正确呈现。

-->

# 问答

[[open-in-colab]]

<Youtube id="ajPx5LwJD-I"/>

问答任务根据问题得到一个答案。如果你曾经向Alexa、Siri或Google等虚拟助手询问天气情况，那么你之前使用过问答模型。常见的问答任务有两种类型：

- 抽取式：从给定的上下文中抽取答案。
- 生成式：从上下文生成一个正确回答问题的答案。

本指南将介绍如何：

1. 使用[SQuAD](https://huggingface.co/datasets/squad)数据集对[DistilBERT](https://huggingface.co/distilbert-base-uncased)进行微调，用于抽取式问答。
2. 使用微调后的模型进行推理。

<Tip>
本教程中展示的任务由以下模型架构支持：

<!--此提示由`make fix-copies`自动生成，勿自行填写！-->


[ALBERT](../model_doc/albert), [BART](../model_doc/bart), [BERT](../model_doc/bert), [BigBird](../model_doc/big_bird), [BigBird-Pegasus](../model_doc/bigbird_pegasus), [BLOOM](../model_doc/bloom), [CamemBERT](../model_doc/camembert), [CANINE](../model_doc/canine), [ConvBERT](../model_doc/convbert), [Data2VecText](../model_doc/data2vec-text), [DeBERTa](../model_doc/deberta), [DeBERTa-v2](../model_doc/deberta-v2), [DistilBERT](../model_doc/distilbert), [ELECTRA](../model_doc/electra), [ERNIE](../model_doc/ernie), [ErnieM](../model_doc/ernie_m), [Falcon](../model_doc/falcon), [FlauBERT](../model_doc/flaubert), [FNet](../model_doc/fnet), [Funnel Transformer](../model_doc/funnel), [OpenAI GPT-2](../model_doc/gpt2), [GPT Neo](../model_doc/gpt_neo), [GPT NeoX](../model_doc/gpt_neox), [GPT-J](../model_doc/gptj), [I-BERT](../model_doc/ibert), [LayoutLMv2](../model_doc/layoutlmv2), [LayoutLMv3](../model_doc/layoutlmv3), [LED](../model_doc/led), [LiLT](../model_doc/lilt), [Longformer](../model_doc/longformer), [LUKE](../model_doc/luke), [LXMERT](../model_doc/lxmert), [MarkupLM](../model_doc/markuplm), [mBART](../model_doc/mbart), [MEGA](../model_doc/mega), [Megatron-BERT](../model_doc/megatron-bert), [MobileBERT](../model_doc/mobilebert), [MPNet](../model_doc/mpnet), [MPT](../model_doc/mpt), [MRA](../model_doc/mra), [MT5](../model_doc/mt5), [MVP](../model_doc/mvp), [Nezha](../model_doc/nezha), [Nyströmformer](../model_doc/nystromformer), [OPT](../model_doc/opt), [QDQBert](../model_doc/qdqbert), [Reformer](../model_doc/reformer), [RemBERT](../model_doc/rembert), [RoBERTa](../model_doc/roberta), [RoBERTa-PreLayerNorm](../model_doc/roberta-prelayernorm), [RoCBert](../model_doc/roc_bert), [RoFormer](../model_doc/roformer), [Splinter](../model_doc/splinter), [SqueezeBERT](../model_doc/squeezebert), [T5](../model_doc/t5), [UMT5](../model_doc/umt5), [XLM](../model_doc/xlm), [XLM-RoBERTa](../model_doc/xlm-roberta), [XLM-RoBERTa-XL](../model_doc/xlm-roberta-xl), [XLNet](../model_doc/xlnet), [X-MOD](../model_doc/xmod), [YOSO](../model_doc/yoso)


<!--End of the generated tip-->

</Tip>

开始之前，请确保你已经安装了所有必需的库：

```bash
pip install transformers datasets evaluate
```

我们鼓励你登录到你的Hugging Face账号，这样你就可以将你的模型上传并共享给社区用户。当提示时，请输入你的令牌以登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 SQuAD 数据集

首先，通过🤗 Datasets库加载SQuAD数据集的一个较小子集。这将给你一个机会在使用完整数据集进行训练之前进行实验和确保一切工作正常。

```py
>>> from datasets import load_dataset

>>> squad = load_dataset("squad", split="train[:5000]")
```

使用[`~datasets.Dataset.train_test_split`]方法将数据集的“train”拆分为训练集和测试集：

```py
>>> squad = squad.train_test_split(test_size=0.2)
```

然后看一个例子：

```py
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

这里有几个重要的字段：

- `answers`：答案标记的起始位置和答案文本。
- `context`：模型需要从中提取答案的背景信息。
- `question`：模型应该回答的问题。

## 预处理

<Youtube id="qgaM0weJHpA"/>

下一步是加载DistilBERT tokenizer以处理`question`和`context`字段：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

还有一些特定于问答任务的预处理步骤需要注意：

1. 数据集中的一些示例可能具有非常长的`context`，超过了模型的最大输入长度。为了处理更长的序列，只截断`context`，将`truncation`设置为"only_second"。
2. 接下来，通过设置`return_offsets_mapping=True`，将回答的开始和结束位置映射到原始的`context`。
3. 有了映射后，可以找到答案的开始和结束标记。使用[`~tokenizers.Encoding.sequence_ids`]方法找出哪部分偏移对应于`question`，哪部分对应于`context`。

下面是如何创建函数来截断和映射`answer`的开始和结束标记到`context`的方法：

```py
>>> def preprocess_function(examples):
...     questions = [q.strip() for q in examples["question"]]
...     inputs = tokenizer(
...         questions,
...         examples["context"],
...         max_length=384,
...         truncation="only_second",
...         return_offsets_mapping=True,
...         padding=True,
...         max_length=(64, 384),  # 扩展输入以适应新的输入token
...         stride=128  # 测试时按照128 stride
...     )

...     offset_mapping = inputs.pop("offset_mapping")
...     answers = examples["answers"]
...     start_positions = []
...     end_positions = []

...     for i, offset in enumerate(offset_mapping):
...         answer = answers[i]
...         start_char = answer["answer_start"][0]
...         end_char = answer["answer_start"][0] + len(answer["text"][0])
...         sequence_ids = inputs.sequence_ids(i)

...         # 找到上下文的开始和结束位置
...         idx = 0
...         while sequence_ids[idx] != 1:
...             idx += 1
...         context_start = idx
...         while sequence_ids[idx] == 1:
...             idx += 1
...         context_end = idx - 1

...         # 如果答案没有完全在上下文内，则标记为(0, 0)
...         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
...             start_positions.append(0)
...             end_positions.append(0)
...         else:
...             # 否则是开始和结束标记的位置
...             idx = context_start
...             while idx <= context_end and offset[idx][0] <= start_char:
...                 idx += 1
...             start_positions.append(idx - 1)

...             idx = context_end
...             while idx >= context_start and offset[idx][1] >= end_char:
...                 idx -= 1
...             end_positions.append(idx + 1)

...     inputs["start_positions"] = start_positions
...     inputs["end_positions"] = end_positions
...     return inputs
```

要在整个数据集上应用预处理函数，使用🤗 Datasets的[`~datasets.Dataset.map`]函数即可。你可以通过将`batched=True`设置为一次处理数据集的多个元素来加快`map`函数的速度。删除你不需要的任何列：

```py
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

然后使用[`DefaultDataCollator`]创建一批示例。与🤗 Transformers中的其他数据整理器不同，[`DefaultDataCollator`]不会应用任何额外的预处理，例如填充。

<frameworkcontent>
<pt>
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
</pt>
<tf>
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```
</tf>
</frameworkcontent>

## 训练

<frameworkcontent>
<pt>
<Tip>

如果你不熟悉使用[`Trainer`]微调模型，请参阅[此处](../training.md#train-with-pytorch-trainer)的基础教程！

</Tip>

现在你可以开始训练模型了！使用[`AutoModelForQuestionAnswering`]加载DistilBERT：

```py
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```

在这一点上，只剩下三个步骤：

1. 在[`TrainingArguments`]中定义你的训练超参数。唯一需要的参数是`output_dir`，指定保存模型的位置。你可以通过设置`push_to_hub=True`将模型推送到Hub（你需要登录Hugging Face才能上传模型）。
2. 将训练参数与模型、数据集、tokenizer和数据整理器一起传递给[`Trainer`]。
3. 调用[`~Trainer.train`]进行微调模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     evaluation_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

训练完成后，使用[`~transformers.Trainer.push_to_hub`]方法将模型分享给Hub，这样每个人都可以使用你的模型：

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

如果你不熟悉使用Keras微调模型，请参阅[此处](../training.md#train-a-tensorflow-model-with-keras)的基础教程！

</Tip>
要在TensorFlow中微调模型，请首先设置优化器、学习率计划和一些训练超参数：

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_epochs = 2
>>> total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs
>>> optimizer, schedule = create_optimizer(
...     init_lr=2e-5,
...     num_warmup_steps=0,
...     num_train_steps=total_train_steps,
... )
```

然后使用[`TFAutoModelForQuestionAnswering`]加载DistilBERT：

```py
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering("distilbert-base-uncased")
```

使用[`~transformers.TFPreTrainedModel.prepare_tf_dataset`]将数据集转换为`tf.data.Dataset`格式：

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_squad["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_squad["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

使用[`compile`](https://keras.io/api/models/model_training_apis/#compile-method)为训练配置模型：

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

在开始训练之前，你还需要提供一种将模型推送到Hub的方法。这可以通过在[`~transformers.PushToHubCallback`]中指定要推送模型和tokenizer的位置来完成：

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_qa_model",
...     tokenizer=tokenizer,
... )
```

最后，你已经准备好开始训练模型了！调用[`fit`](https://keras.io/api/models/model_training_apis/#fit-method)与训练集、验证集的样本数量、回调函数来微调模型：

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=[callback])
```
训练完成后，你的模型将自动上传到Hub，以便每个人都可以使用它！
</tf>
</frameworkcontent>

<Tip>

要了解如何对问答模型进行评估并了解其性能，请参阅🤗 Hugging Face课程中的[问答](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing)章节。

</Tip>

## 推理

太好了，你已经微调了一个模型，现在可以用它进行推理了！

提出一个问题和一些你希望模型预测的上下文：

在使用你的微调模型进行推理时，最简单的方法是在[`pipeline`]中使用它。使用你的模型实例化一个问题回答的`pipeline`，并将你的文本传递给它：

```py
>>> from transformers import pipeline

>>> question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
>>> question_answerer(question=question, context=context)
{'score': 0.2058267742395401,
 'start': 10,
 'end': 95,
 'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
```

如果你愿意，你也可以手动复制`pipeline`的结果：

<frameworkcontent>
<pt>
对文本进行标记化并返回PyTorch张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

将你的输入传递给模型并返回`logits`：

```py
>>> import torch
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

从模型输出中获取开始和结束位置的最高概率：

```py
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

将预测的标记解码为答案：

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
</pt>
<tf>
对文本进行标记化并返回TensorFlow张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, text, return_tensors="tf")
```

将你的输入传递给模型并返回`logits`：

```py
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> outputs = model(**inputs)
```

从模型输出中获取开始和结束位置的最高概率：

```py
>>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
>>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
```

将预测的标记解码为答案：

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
</tf>
</frameworkcontent>