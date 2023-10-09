# 视觉问答

[[open-in-colab]]

视觉问答（Visual Question Answering，VQA）是根据图像回答开放式问题的任务。支持这个任务的模型的输入通常是图像和一个问题的组合，输出是以自然语言表达的答案。

一些值得注意的VQA应用案例包括：
* 面向视力障碍个体的辅助应用程序。
* 教育：对讲座或教科书中呈现的视觉材料提问。VQA还可以在互动博物馆展示或历史遗址中使用。
* 客户服务和电子商务：VQA可以通过允许用户提问产品来提升用户体验。
* 图像检索：VQA模型可以用于检索具有特定特征的图像。例如，用户可以询问“有一只狗吗？”以从一组图像中找到所有包含狗的图像。

在本指南中，您将学习以下内容：

- 在[`Graphcore/vqa`数据集](https://huggingface.co/datasets/Graphcore/vqa)上微调分类VQA模型，特别是[ViLT](../model_doc/vilt)。
- 使用经微调的ViLT进行推理。
- 使用生成模型（如BLIP-2）进行零-shot VQA推理。

## 微调ViLT

ViLT模型将文本嵌入集成到视觉Transformer（ViT）中，使其可以在视觉和语言预训练（VLP）中具有最小的设计。该模型可以用于多个下游任务。对于VQA任务，将在顶部放置一个分类器头（放置在`[CLS]`标记的最终隐藏状态之上的线性层）并进行随机初始化。因此，视觉问答被视为**分类问题**。

较新的模型，如BLIP、BLIP-2和InstructBLIP，将VQA作为生成任务处理。在本指南的后面部分，我们将说明如何使用它们进行零-shot VQA推理。

在开始之前，请确保您已安装了所有必要的库。

```bash
pip install -q transformers datasets
```

我们鼓励您与社区共享您的模型。登录到您的Hugging Face帐号，将其上传到🤗 Hub。在提示时，输入您的令牌以登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

让我们将模型检查点定义为全局变量。

```py
>>> model_checkpoint = "dandelin/vilt-b32-mlm"
```

## 加载数据

为了说明目的，在本指南中，我们使用了`Graphcore/vqa`数据集的非常小的样本。您可以在[🤗 Hub](https://huggingface.co/datasets/Graphcore/vqa)上找到完整的数据集。

作为[`Graphcore/vqa`数据集](https://huggingface.co/datasets/Graphcore/vqa)的替代，您可以从官方的[VQA数据集页面](https://visualqa.org/download.html)手动下载相同的数据。如果您想使用自定义数据跟随本教程，请查看在🤗 Datasets文档中的[创建图像数据集](https://huggingface.co/docs/datasets/image_dataset#loading-script)指南。

让我们加载验证集的前200个示例，并探索数据集的特征：

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
>>> dataset
Dataset({
    features: ['question', 'question_type', 'question_id', 'image_id', 'answer_type', 'label'],
    num_rows: 200
})
```

让我们查看一个示例以了解数据集的特征：

```py
>>> dataset[0]
{'question': 'Where is he looking?',
 'question_type': 'none of the above',
 'question_id': 262148000,
 'image_id': '/root/.cache/huggingface/datasets/downloads/extracted/ca733e0e000fb2d7a09fbcc94dbfe7b5a30750681d0e965f8e0a23b1c2f98c75/val2014/COCO_val2014_000000262148.jpg',
 'answer_type': 'other',
 'label': {'ids': ['at table', 'down', 'skateboard', 'table'],
  'weights': [0.30000001192092896,
   1.0,
   0.30000001192092896,
   0.30000001192092896]}}
```

与任务相关的特征包括：
* `question`：需要从图像中回答的问题
* `image_id`：问题相关的图像的路径
* `label`：注释

由于这些特征并不是必需的，请将其删除：

```
>>> dataset = dataset.remove_columns(['question_type', 'question_id', 'answer_type'])
```

正如您所见，`label`特征包含对相同问题的多个答案（此处称为`ids`），它们是由不同的人类标注者收集到的。这是因为对问题的答案可能是主观的。在这种情况下，问题是“他在哪里看？”。有人用“下面”进行注释，有人用“在桌子上”，还有人用“滑板”等等。

查看图像并考虑一下您会给出哪个答案：

```python
>>> from PIL import Image

>>> image = Image.open(dataset[0]['image_id'])
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/vqa-example.png" alt="VQA Image Example"/>
</div>

由于问题和答案的不确定性，像这样的数据集被视为多标签分类问题（因为可能有多个答案有效）。此外，与其只创建一个one-hot编码向量，不如创建一个软编码，基于某个答案在注释中出现的次数。

例如，在上面的示例中，因为答案“下面”经常被选择，所以它的分数（在数据集中称为`weight`）为1.0，而其他答案的分数<1.0。

为了以后使用适当的分类头实例化模型，让我们创建两个字典：一个将标签名称映射到整数，另一个将整数映射回到标签名称：

```py
>>> import itertools

>>> labels = [item['ids'] for item in dataset['label']]
>>> flattened_labels = list(itertools.chain(*labels))
>>> unique_labels = list(set(flattened_labels))

>>> label2id = {label: idx for idx, label in enumerate(unique_labels)}
>>> id2label = {idx: label for label, idx in label2id.items()} 
```

现在我们有了映射，我们可以用它来用它们的id替换字符串答案，并将数据集扁平化以进行更方便的进一步预处理。

```python
>>> def replace_ids(inputs):
...   inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
...   return inputs


>>> dataset = dataset.map(replace_ids)
>>> flat_dataset = dataset.flatten()
>>> flat_dataset.features
{'question': Value(dtype='string', id=None),
 'image_id': Value(dtype='string', id=None),
 'label.ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
 'label.weights': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}
```

## 数据预处理

下一步是加载一个ViLT处理器，以准备图像和文本数据以供模型使用。[`ViltProcessor`]将BERT标记器和ViLT图像处理器封装成一个方便的单一处理器：

```py 
>>> from transformers import ViltProcessor

>>> processor = ViltProcessor.from_pretrained(model_checkpoint)
```

为了预处理数据，我们需要使用[`ViltProcessor`]对图像和问题进行编码。处理器将使用[`BertTokenizerFast`]对文本进行标记化，并为文本数据创建`input_ids`、`attention_mask`和`token_type_ids`。对于图像，处理器将利用[`ViltImageProcessor`]对图像进行调整大小和归一化，并创建`pixel_values`和`pixel_mask`。

所有这些预处理步骤都在幕后完成，我们只需要调用处理器即可。但是，我们仍然需要准备目标标签。在这个表示中，每个元素对应一个可能的答案（标签）。对于正确的答案，该元素保存它们的相应分数（权重），而其他元素设置为零。

下面的函数应用了`processor`到图像和问题，并格式化了标签，如上所述：

```py
>>> import torch

>>> def preprocess_data(examples):
...     image_paths = examples['image_id']
...     images = [Image.open(image_path) for image_path in image_paths]
...     texts = examples['question']    

...     encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

...     for k, v in encoding.items():
...           encoding[k] = v.squeeze()
    
...     targets = []

...     for labels, scores in zip(examples['label.ids'], examples['label.weights']):
...         target = torch.zeros(len(id2label))

...         for label, score in zip(labels, scores):
...             target[label] = score
      
...         targets.append(target)

...     encoding["labels"] = targets
    
...     return encoding
```

为了在整个数据集上应用预处理函数，使用🤗 Datasets的[`~datasets.map`]函数。您可以通过将`batched=True`设置为一次处理数据集的多个元素来加速`map`。此时，可以删除您不需要的列。

```py
>>> processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
>>> processed_dataset
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask', 'labels'],
    num_rows: 200
})
```

作为最后一步，使用[`DefaultDataCollator`]创建一个批处理的示例：

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## 训练模型

您现在已经准备好开始训练模型了！使用[`ViltForQuestionAnswering`]加载ViLT模型。指定标签的数量以及标签映射：

```py
>>> from transformers import ViltForQuestionAnswering

>>> model = ViltForQuestionAnswering.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
```

此时，只需要三个步骤即可完成：

1. 在[`TrainingArguments`]中定义您的训练超参数：

```py
>>> from transformers import TrainingArguments

>>> repo_id = "MariaK/vilt_finetuned_200"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

2. 将训练参数与模型、数据集、处理器和数据collator一起传递给[`Trainer`]。

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=processed_dataset,
...     tokenizer=processor,
... )
```

3. 调用[`~Trainer.train`]来微调您的模型。

```py
>>> trainer.train() 
```

一旦训练完成，使用[`~Trainer.push_to_hub`]方法将您的模型共享到Hub上：

```py
>>> trainer.push_to_hub()
```

## 推理

既然您已经微调了一个ViLT模型，并将其上传到了🤗 Hub上，您可以使用它进行推理。尝试将您微调的模型用于推理的最简单方式是在[`Pipeline`]中使用它。

```py
>>> from transformers import pipeline

>>> pipe = pipeline("visual-question-answering", model="MariaK/vilt_finetuned_200")
```

此指南中的模型仅在200个示例上进行了训练，因此请不要对它期望太高。让我们看看它是否至少学到了一些东西，并使用VQA数据集中的第一个示例来说明推理过程：

```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
>>> print(question)
>>> pipe(image, question)
"Where is he looking?"
[{'score': 0.5498199462890625, 'answer': 'down'}]
```

尽管置信度不高，但该模型确实学到了一些东西。通过更多的示例和更长时间的训练，您将获得更好的结果！

如果您愿意，您也可以手动复制Pipeline的结果：
1. 获取一个图像和一个问题，使用您模型的处理器对其进行准备。
2. 通过模型转发预处理结果。
3. 从logits中获取最可能的答案id，并在`id2label`中找到实际答案。

```py
>>> processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")

>>> image = Image.open(example['image_id'])
>>> question = example['question']

>>> # prepare inputs
>>> inputs = processor(image, question, return_tensors="pt")

>>> model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits
>>> idx = logits.argmax(-1).item()
>>> print("Predicted answer:", model.config.id2label[idx])
Predicted answer: down
```

## 零-shot VQA

前一模型将VQA视为分类任务进行处理。近期的一些模型，如BLIP、BLIP-2和InstructBLIP，则将VQA视为生成任务。让我们以[BLIP-2](../model_doc/blip-2)为例。它引入了一种新的视觉-语言预训练范式，其中可以使用任意预训练视觉编码器和LLM的组合（在[BLIP-2博文](https://huggingface.co/blog/blip-2)中了解更多）。这使得在多个视觉-语言任务中实现了最先进的结果，包括视觉问答。

让我们看看您如何在VQA任务上使用这个模型。首先，让我们加载模型。这里我们将模型明确发送到GPU（如果有的话），而之前在训练时不需要这样做，因为[`Trainer`]会自动处理： 

```py
>>> from transformers import AutoProcessor, Blip2ForConditionalGeneration
>>> import torch

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)
```

这个模型接受图像和文本作为输入，所以让我们使用VQA数据集中第一个示例的完全相同的图像/问题对： 

```py 
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
```

要使用BLIP-2进行视觉问答任务，文本提示必须遵循特定的格式：`Question: {} Answer:`.

```py
>>> prompt = f"Question: {question} Answer:" 
```

现在我们需要使用模型的processor预处理图像/提示，通过模型传递处理过的输入，然后解码输出：

```py
>>> inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

>>> generated_ids = model.generate(**inputs, max_new_tokens=10)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
"He is looking at the crowd"
```

正如您所见，模型识别了人群和面部的朝向（向下看），但它似乎忽略了人群在滑冰者后面的事实。尽管如此，在无法获取人工注释数据集的情况下，这种方法可以快速产生有用的结果。