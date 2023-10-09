版权©2022 HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），您除非符合许可证的规定，否则不得使用此文件。您可以在下面的链接找到许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

请注意，此文件以Markdown格式编写，但包含用于我们的文档构建器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确显示。

音频分类

[[open-in-colab]]

<Youtube id="KWwzcmG98Ds"/>

音频分类-与文本一样-将类标签输出分配给输入数据。唯一的区别是，使用原始音频波形而不是文本输入。音频分类的一些实际应用包括识别说话者意图，语言分类，甚至通过声音识别动物物种。

本指南将向您展示如何：

1.在[MInDS-14](https://huggingface.co/datasets/PolyAI/minds14)数据集上微调[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base)，以实现说话者意图的分类。
2.使用您微调的模型进行推理。

提示
本教程中所示的任务支持以下模型架构：

音频频谱图变换器，Data2VecAudio，Hubert，SEW，SEW-D，UniSpeech，UniSpeechSat，Wav2Vec2，Wav2Vec2-Conformer，WavLM，Whisper

在开始之前，请确保您已安装所有必要的库：

```bash
pip install transformers datasets evaluate
```

我们鼓励您登录到Hugging Face账户，这样您就可以上传和共享您的模型。当提示时，输入您的令牌进行登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

加载MInDS-14数据集

首先从🤗 Datasets库加载MInDS-14数据集：

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

将数据集的“train”拆分为较小的训练集和测试集，使用[`~datasets.Dataset.train_test_split`]方法。这样您可以有机会在处理完整数据集之前进行实验和确认一切正常。

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

然后查看数据集：

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 450
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 113
    })
})
```

数据集包含许多有用的信息，例如`lang_id`和`english_transcription`，但在本指南中，您将专注于`audio`和`intent_class`。使用[`~datasets.Dataset.remove_columns`]方法删除其他列：

```py
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

现在查看一个示例：

```py
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

有两个字段：

- `audio`：一个一维的`array`，存储了必须调用以加载和重新采样音频文件的语音信号。
- `intent_class`：代表说话者意图的类别id。

为了使模型可以从标签id获取标签名称，创建一个将标签名称映射到整数和相反的字典：

```py
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

现在可以将标签id转换为标签名称：

```py
>>> id2label[str(2)]
'app_error'
```

预处理

下一步是加载Wav2Vec2特征提取器以处理音频信号：

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

MInDS-14数据集的采样率为8000khz（您可以在其[数据集卡片](https://huggingface.co/datasets/PolyAI/minds14)中找到此信息），这意味着您需要将数据集重采样为16000kHz以使用预训练的Wav2Vec2模型：

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

现在创建一个预处理函数：

1. 调用`audio`列以加载和（如果必要）重采样音频文件。
2. 检查音频文件的采样率是否与音频数据模型的采样率匹配。您可以在Wav2Vec2[模型卡片](https://huggingface.co/facebook/wav2vec2-base)中找到此信息。
3. 设置最大输入长度，以便批处理更长的输入而不会将其截断。

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

使用🤗 Datasets [`~datasets.Dataset.map`]函数将预处理函数应用于整个数据集。可以通过设置`batched=True`加快`map`的速度，以一次处理数据集的多个元素。删除不需要的列，并将`intent_class`重命名为`label`，因为模型期望的名称是`label`：

```py
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
```

评估

在训练过程中包含指标通常有助于评估模型的性能。您可以使用🤗[Evaluate](https://huggingface.co/docs/evaluate/index)库快速加载评估方法。对于此任务，加载[accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)指标（请参阅🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour)以了解有关如何加载和计算指标的更多信息）：

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

然后创建一个函数，将您的预测和标签传递给[`~evaluate.EvaluationModule.compute`]以计算准确率：

```py
>>> import numpy as np

>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

现在您的`compute_metrics`函数已准备就绪，当您设置训练时将返回它。

训练

现在您准备开始训练模型了！使用[`AutoModelForAudioClassification`]加载Wav2Vec2模型以及期望标签的数量和标签映射：

```py
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

到此为止，只剩下三个步骤：

1. 在[`TrainingArguments`]中定义您的训练超参数。唯一需要的参数是`output_dir`，它指定了保存模型的位置。通过设置`push_to_hub=True`将此模型推送到Hub（需要登录到Hugging Face上载您的模型）。在每个epoch结束时，[`Trainer`]将评估准确性并保存训练检查点。
2. 将训练参数与模型、数据集、标记器、数据整理器和`compute_metrics`函数一起传递给[`Trainer`]。
3. 调用[`~Trainer.train`]以微调模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_mind_model",
...     evaluation_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=3e-5,
...     per_device_train_batch_size=32,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=32,
...     num_train_epochs=10,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     tokenizer=feature_extractor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

训练完成后，使用[`~transformers.Trainer.push_to_hub`]方法将模型分享到Hub，以便所有人都可以使用您的模型：

```py
>>> trainer.push_to_hub()
```

推理

很好，现在您已经微调了一个模型，可以将其用于推理！

加载要运行推理的音频文件。记得要根据需要重新采样音频文件的采样率以匹配模型的采样率。

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

尝试使用[`pipeline`]在推理中使用微调的模型最简单的方法是将其用于音频分类。用您的模型实例化一个音频分类的`pipeline`，并将音频文件传递给它：

```py
>>> from transformers import pipeline

>>> classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model")
>>> classifier(audio_file)
[
    {'score': 0.09766869246959686, 'label': 'cash_deposit'},
    {'score': 0.07998877018690109, 'label': 'app_error'},
    {'score': 0.0781070664525032, 'label': 'joint_account'},
    {'score': 0.07667109370231628, 'label': 'pay_bill'},
    {'score': 0.0755252093076706, 'label': 'balance'}
]
```

如果愿意，您也可以手动复制`pipeline`的结果：

加载一个特征提取器来预处理音频文件并将`input`作为PyTorch张量返回：

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

将inputs传递给模型并返回logits：

```py
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

获取概率最高的类别，并使用模型的`id2label`映射将其转换为标签：

```py
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```