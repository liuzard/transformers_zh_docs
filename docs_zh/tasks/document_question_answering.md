# 文档问答

[[在 Colab 中打开]]

文档问答，也称为文档可视化问答，是一项涉及为关于文档图像的问题提供答案的任务。支持此任务的模型的输入通常是图像和问题的组合，输出是以自然语言表达的答案。这些模型利用了多种模态，包括文本、单词位置（边界框）和图像本身。

本指南说明了如何：

- 在[DocVQA 数据集](https://huggingface.co/datasets/nielsr/docvqa_1200_examples_donut)上对 [LayoutLMv2](../model_doc/layoutlmv2) 进行微调。
- 使用微调的模型进行推理。

<Tip>

本教程中所示任务受以下模型架构支持：

<!--此提示是由 `make fix-copies` 自动生成的，请勿手动填写！-->

[LayoutLM](../model_doc/layoutlm)、[LayoutLMv2](../model_doc/layoutlmv2)、[LayoutLMv3](../model_doc/layoutlmv3)

<!--生成提示结束-->

</Tip>

LayoutLMv2 通过在标记的最终隐藏状态上添加一个问答头，来解决文档问答任务，以预测答案的起始和结束标记的位置。换句话说，该问题被视为抽取问题回答：给定上下文，提取哪个信息片段是答案。上下文来自 OCR 引擎的输出，此处使用的是 Google 的 Tesseract。

开始之前，请确保已安装所有必要的库。LayoutLMv2 依赖于 detectron2、torchvision 和 tesseract。

```bash
pip install -q transformers datasets
```

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install torchvision
```

```bash
sudo apt install tesseract-ocr
pip install -q pytesseract
```

安装完所有依赖项后，请重新启动运行时。

我们鼓励您与社区共享您的模型。登录到您的 Hugging Face 帐户以将其上传到 🤗 Hub。当提示时，输入您的令牌以登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

让我们定义一些全局变量。

```py
>>> model_checkpoint = "microsoft/layoutlmv2-base-uncased"
>>> batch_size = 4
```

## 加载数据

在本指南中，我们使用了一个预处理的 DocVQA 的小样本，您可以在 🤗 Hub 上找到。如果您希望使用完整的 DocVQA 数据集，可以在 [DocVQA 主页](https://rrc.cvc.uab.es/?ch=17) 上注册并下载。如果这样做，请按照 [如何加载本地和远程文件到 🤗 数据集](https://huggingface.co/docs/datasets/loading#local-and-remote-files) 的指南操作。

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("nielsr/docvqa_1200_examples")
>>> dataset
DatasetDict({
    train: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 1000
    })
    test: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 200
    })
})
```

如您所见，数据集已经划分为训练集和测试集。随机查看一个示例，以熟悉特征。

```py
>>> dataset["train"].features
```

这里是各个字段的含义：
* `id`：示例的 ID
* `image`：包含文档图像的 PIL.Image.Image 对象
* `query`：问题字符串 - 自然语言问句，有多种语言
* `answers`：由人工注释者提供的正确答案列表
* `words` 和 `bounding_boxes`：OCR 的结果，本教程不使用
* `answer`：此处不使用的另一个模型匹配的答案

让我们保留仅含有英文问题的数据，并删除包含预测（在 `answer` 字段中）的特征。我们还将从提供的答案集合中选择第一个答案。或者，您也可以随机采样。

```py
>>> updated_dataset = dataset.map(lambda example: {"question": example["query"]["en"]}, remove_columns=["query"])
>>> updated_dataset = updated_dataset.map(
...     lambda example: {"answer": example["answers"][0]}, remove_columns=["answer", "answers"]
... )
```

请注意，此指南中使用的 LayoutLMv2 检查点已经训练过 `max_position_embeddings = 512`（您可以在[检查点的 `config.json` 文件](https://huggingface.co/microsoft/layoutlmv2-base-uncased/blob/main/config.json#L18)中找到此信息）。我们可以截断示例，以避免嵌入可能超过 512 的情况，这样就不会出现答案位于大型文档的末尾且可能被截断的情况。在这里，我们将删除那些嵌入的长度可能超过 512 的少数示例。如果数据集中的大多数文档都很长，您可以实施一个滑动窗口策略——参见[此笔记本](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb)以了解详细信息。

```py
>>> updated_dataset = updated_dataset.filter(lambda x: len(x["words"]) + len(x["question"].split()) < 512)
```

此时我们还可以从数据集中删除 OCR 特性。这些是通过 OCR 得到的结果，用于对不同的模型进行微调。如果要使用它们，仍需要进行一些处理，因为它们与本指南中使用的模型的输入要求不匹配。相反，我们可以使用 [`LayoutLMv2Processor`]在原始数据上同时进行 OCR 和标记化的处理。这样我们就可以得到与模型期望输入相匹配的输入。如果您想手动处理图像，请查看 [`LayoutLMv2` 模型文档](../model_doc/layoutlmv2) 以了解模型期望的输入格式。

```py
>>> updated_dataset = updated_dataset.remove_columns("words")
>>> updated_dataset = updated_dataset.remove_columns("bounding_boxes")
```

最后，如果我们没有查看图像示例，数据探索就不够完整。

```py
>>> updated_dataset["train"][11]["image"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/docvqa_example.jpg" alt="DocVQA Image Example"/>
 </div>

## 预处理数据

文档问答任务是一个多模态任务，您需要确保每个模态的输入按照模型的期望进行预处理。让我们首先加载 [`LayoutLMv2Processor`]，该处理器在内部结合了可以处理图像数据的图像处理器和可以编码文本数据的标记器。

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
```

### 预处理文档图像

首先，让我们通过处理器的 `image_processor` 准备文档图像，以使其适合模型。通过默认设置，图像处理器将图像调整为 224x224，确保其具有正确的通道顺序，使用 tesseract 进行 OCR 来获取单词和归一化的边界框。在本教程中，所有这些默认设置都是我们所需的。编写一个函数，将默认的图像处理应用于一批图像，并返回 OCR 的结果。

```py
>>> image_processor = processor.image_processor


>>> def get_ocr_words_and_boxes(examples):
...     images = [image.convert("RGB") for image in examples["image"]]
...     encoded_inputs = image_processor(images)

...     examples["image"] = encoded_inputs.pixel_values
...     examples["words"] = encoded_inputs.words
...     examples["boxes"] = encoded_inputs.boxes

...     return examples
```

为了以快速的方式将此预处理应用于整个数据集，请使用 [`~datasets.Dataset.map`]。

```py
>>> dataset_with_ocr = updated_dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=2)
```

### 预处理文本数据

一旦我们对图像应用了 OCR，我们还需要编码数据集的文本部分，以使其准备好输入模型。这包括将我们在上一步中获取的单词和边界框转换为标记级别的 `input_ids`、`attention_mask`、`token_type_ids` 和 `bbox`。对于文本预处理，我们将需要处理器的 `tokenizer`。

```py
>>> tokenizer = processor.tokenizer
```

在上述预处理之外，我们还需要添加模型的标签。对于 🤗 Transformers 中的 `xxxForQuestionAnswering` 模型，标签由 `start_positions` 和 `end_positions` 组成，指示答案的开始和结束的位置。

让我们从这一点开始。定义一个辅助函数，可以在一个较大的列表（单词列表）中找到一个子列表（答案分割成了单词列表）。

该函数将接受两个列表作为输入，`words_list` 和 `answer_list`。它将遍历 `words_list`，检查 `words_list` 中的当前单词（words_list[i]) 是否等于 `answer_list` 的第一个单词（answer_list[0])，并且检查从当前单词开始且与 `answer_list` 长度相同的 `words_list` 的子列表是否等于 `answer_list`。如果这个条件为真，就意味着已找到一个匹配，函数将记录此匹配及其开始索引（idx）和结束索引（idx + len(answer_list) - 1）。如果找到超过一个匹配，函数将仅返回第一个匹配。如果未找到匹配，函数将返回 (`None`、0 和 0)。

```py
>>> def subfinder(words_list, answer_list):
...     matches = []
...     start_indices = []
...     end_indices = []
...     for idx, i in enumerate(range(len(words_list))):
...         if words_list[i] == answer_list[0] and words_list[i : i + len(answer_list)] == answer_list:
...             matches.append(answer_list)
...             start_indices.append(idx)
...             end_indices.append(idx + len(answer_list) - 1)
...     if matches:
...         return matches[0], start_indices[0], end_indices[0]
...     else:
...         return None, 0, 0
```

为说明此函数如何找到答案的位置，让我们在一个例子上使用它：

```py
>>> example = dataset_with_ocr["train"][1]
>>> words = [word.lower() for word in example["words"]]
>>> match, word_idx_start, word_idx_end = subfinder(words, example["answer"].lower().split())
>>> print("Question: ", example["question"])
>>> print("Words:", words)
>>> print("Answer: ", example["answer"])
>>> print("start_index", word_idx_start)
>>> print("end_index", word_idx_end)
Question:  Who is in  cc in this letter?
Words: ['wie', 'baw', 'brown', '&', 'williamson', 'tobacco', 'corporation', 'research', '&', 'development', 'internal', 'correspondence', 'to:', 'r.', 'h.', 'honeycutt', 'ce:', 't.f.', 'riehl', 'from:', '.', 'c.j.', 'cook', 'date:', 'may', '8,', '1995', 'subject:', 'review', 'of', 'existing', 'brainstorming', 'ideas/483', 'the', 'major', 'function', 'of', 'the', 'product', 'innovation', 'graup', 'is', 'to', 'develop', 'marketable', 'nove!', 'products', 'that', 'would', 'be', 'profitable', 'to', 'manufacture', 'and', 'sell.', 'novel', 'is', 'defined', 'as:', 'of', 'a', 'new', 'kind,', 'or', 'different', 'from', 'anything', 'seen', 'or', 'known', 'before.', 'innovation', 'is', 'defined', 'as:', 'something', 'new', 'or', 'different', 'introduced;', 'act', 'of', 'innovating;', 'introduction', 'of', 'new', 'things', 'or', 'methods.', 'the', 'products', 'may', 'incorporate', 'the', 'latest', 'technologies,', 'materials', 'and', 'know-how', 'available', 'to', 'give', 'then', 'a', 'unique', 'taste', 'or', 'look.', 'the', 'first', 'task', 'of', 'the', 'product', 'innovation', 'group', 'was', 'to', 'assemble,', 'review', 'and', 'categorize', 'a', 'list', 'of', 'existing', 'brainstorming', 'ideas.', 'ideas', 'were', 'grouped', 'into', 'two', 'major', 'categories', 'labeled', 'appearance', 'and', 'taste/aroma.', 'these', 'categories', 'are', 'used', 'for', 'novel', 'products', 'that', 'may', 'differ', 'from', 'a', 'visual', 'and/or', 'taste/aroma', 'point', 'of', 'view', 'compared', 'to', 'canventional', 'cigarettes.', 'other', 'categories', 'include', 'a', 'combination', 'of', 'the', 'above,', 'filters,', 'packaging', 'and', 'brand', 'extensions.', 'appearance', 'this', 'category', 'is', 'used', 'for', 'novel', 'cigarette', 'constructions', 'that', 'yield', 'visually', 'different', 'products', 'with', 'minimal', 'changes', 'in', 'smoke', 'chemistry', 'two', 'cigarettes', 'in', 'cne.', 'emulti-plug', 'te', 'build', 'yaur', 'awn', 'cigarette.', 'eswitchable', 'menthol', 'or', 'non', 'menthol', 'cigarette.', '*cigarettes', 'with', 'interspaced', 'perforations', 'to', 'enable', 'smoker', 'to', 'separate', 'unburned', 'section', 'for', 'future', 'smoking.', '«short', 'cigarette,', 'tobacco', 'section', '30', 'mm.', '«extremely', 'fast', 'buming', 'cigarette.', '«novel', 'cigarette', 'constructions', 'that', 'permit', 'a', 'significant', 'reduction', 'iretobacco', 'weight', 'while', 'maintaining', 'smoking', 'mechanics', 'and', 'visual', 'characteristics.', 'higher', 'basis', 'weight', 'paper:', 'potential', 'reduction', 'in', 'tobacco', 'weight.', '«more', 'rigid', 'tobacco', 'column;', 'stiffing', 'agent', 'for', 'tobacco;', 'e.g.', 'starch', '*colored', 'tow', 'and', 'cigarette', 'papers;', 'seasonal', 'promotions,', 'e.g.', 'pastel', 'colored', 'cigarettes', 'for', 'easter', 'or', 'in', 'an', 'ebony', 'and', 'ivory', 'brand', 'containing', 'a', 'mixture', 'of', 'all', 'black', '(black', 'paper', 'and', 'tow)', 'and', 'ail', 'white', 'cigarettes.', '499150498']
Answer:  T.F. Riehl
start_index 17
end_index 18
```

然而，一旦示例被编码，它们会变成这样：

```py
>>> encoding = tokenizer(example["question"], example["words"], example["boxes"])
>>> tokenizer.decode(encoding["input_ids"])
[CLS] who is in cc in this letter? [SEP] wie baw brown & williamson tobacco corporation research & development ...
```

我们需要找到编码输入中答案的位置。
* `token_type_ids`告诉我们哪些标记属于问题，哪些属于文档的单词。
* `tokenizer.cls_token_id`将帮助我们找到输入开头的特殊标记。
* `word_ids`将帮助匹配原始`words`中找到的答案与完整编码输入中的相同答案，并确定答案在编码输入中的起始/结束位置。

考虑到这一点，让我们创建一个编码数据集批处理的函数：

```py
>>> def encode_dataset(examples, max_length=512):
...     questions = examples["question"]
...     words = examples["words"]
...     boxes = examples["boxes"]
...     answers = examples["answer"]

...     # encode the batch of examples and initialize the start_positions and end_positions
...     encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)
...     start_positions = []
...     end_positions = []

...     # loop through the examples in the batch
...     for i in range(len(questions)):
...         cls_index = encoding["input_ids"][i].index(tokenizer.cls_token_id)

...         # find the position of the answer in example's words
...         words_example = [word.lower() for word in words[i]]
...         answer = answers[i]
...         match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())

...         if match:
...             # if match is found, use `token_type_ids` to find where words start in the encoding
...             token_type_ids = encoding["token_type_ids"][i]
...             token_start_index = 0
...             while token_type_ids[token_start_index] != 1:
...                 token_start_index += 1

...             token_end_index = len(encoding["input_ids"][i]) - 1
...             while token_type_ids[token_end_index] != 1:
...                 token_end_index -= 1

...             word_ids = encoding.word_ids(i)[token_start_index : token_end_index + 1]
...             start_position = cls_index
...             end_position = cls_index

...             # loop over word_ids and increase `token_start_index` until it matches the answer position in words
...             # once it matches, save the `token_start_index` as the `start_position` of the answer in the encoding
...             for id in word_ids:
...                 if id == word_idx_start:
...                     start_position = token_start_index
...                 else:
...                     token_start_index += 1

...             # similarly loop over `word_ids` starting from the end to find the `end_position` of the answer
...             for id in word_ids[::-1]:
...                 if id == word_idx_end:
...                     end_position = token_end_index
...                 else:
...                     token_end_index -= 1

...             start_positions.append(start_position)
...             end_positions.append(end_position)

...         else:
...             start_positions.append(cls_index)
...             end_positions.append(cls_index)

...     encoding["image"] = examples["image"]
...     encoding["start_positions"] = start_positions
...     encoding["end_positions"] = end_positions

...     return encoding
```

现在我们有了这个预处理函数，我们可以对整个数据集进行编码：

```py
>>> encoded_train_dataset = dataset_with_ocr["train"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["train"].column_names
... )
>>> encoded_test_dataset = dataset_with_ocr["test"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["test"].column_names
... )
```

让我们看一下编码数据集的特征长什么样子：

```py
>>> encoded_train_dataset.features
{'image': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='uint8', id=None), length=-1, id=None), length=-1, id=None), length=-1, id=None),
 'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
 'token_type_ids': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'bbox': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
 'start_positions': Value(dtype='int64', id=None),
 'end_positions': Value(dtype='int64', id=None)}
```

## 评估

文档问答的评估需要大量的后处理。为了不花费太多时间，本指南跳过了评估步骤。[`Trainer`]仍然在训练过程中计算评估损失，因此您对模型的性能并不完全不知情。最常用的抽取式问答评估方法是使用F1得分/完全匹配度。如果您想自己实现它，请参考Hugging Face课程的[问答章节](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing)。

## 训练

恭喜！您已成功完成本指南最困难的部分，现在可以开始训练自己的模型了。训练包括以下步骤：
* 使用与预处理中相同的检查点使用[`AutoModelForDocumentQuestionAnswering`]加载模型。
* 在[`TrainingArguments`]中定义训练超参数。
* 定义一个将示例批处理在一起的函数，这里[`DefaultDataCollator`]很合适。
* 将训练参数与模型、数据集和数据整合器一起传递给[`Trainer`]。
* 调用[`~Trainer.train`]来微调模型。

```py
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
```

在[`TrainingArguments`]中使用`output_dir`指定保存模型的位置，并根据需要配置超参数。
如果您希望与社区共享您的模型，请将`push_to_hub`设置为`True`（您必须登录Hugging Face才能上传您的模型）。
在这种情况下，`output_dir`也将是存储模型检查点的仓库的名称。

```py
>>> from transformers import TrainingArguments

>>> # REPLACE THIS WITH YOUR REPO ID
>>> repo_id = "MariaK/layoutlmv2-base-uncased_finetuned_docvqa"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     evaluation_strategy="steps",
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

定义一个简单的数据整合器将示例批处理在一起。

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

最后，将所有内容整合在一起，并调用[`~Trainer.train`]：

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=encoded_train_dataset,
...     eval_dataset=encoded_test_dataset,
...     tokenizer=processor,
... )

>>> trainer.train()
```

要将最终模型添加到🤗 Hub中，请创建一个模型卡并调用`push_to_hub`：

```py
>>> trainer.create_model_card()
>>> trainer.push_to_hub()
```

## 推理

现在您已经微调了一个LayoutLMv2模型，并将其上传到了🤗 Hub中，您可以用来进行推理。尝试使用[`Pipeline`]最简单的方法来尝试使用微调模型进行推理。

这里有一个示例：
```py
>>> example = dataset["test"][2]
>>> question = example["query"]["en"]
>>> image = example["image"]
>>> print(question)
>>> print(example["answers"])
'Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?'
['TRRF Vice President', 'lee a. waller']
```

接下来，用您的模型实例化一个用于文档问答的pipeline，并将图像+问题组合传递给它。

```py
>>> from transformers import pipeline

>>> qa_pipeline = pipeline("document-question-answering", model="MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> qa_pipeline(image, question)
[{'score': 0.9949808120727539,
  'answer': 'Lee A. Waller',
  'start': 55,
  'end': 57}]
```

您也可以手动复制pipeline的结果，如果愿意的话：
1. 获取一张图片和一个问题，使用来自您的模型的processor将其准备好。
2. 通过模型前向传播预处理结果。
3. 模型返回`start_logits`和`end_logits`，这些指示出答案的起始位置和结束位置。两者的形状均为（batch_size，sequence_length）。
4. 在`start_logits`和`end_logits`的最后一个维度上进行argmax，以获得预测的`start_idx`和`end_idx`。
5. 使用tokenizer解码答案。

```py
>>> import torch
>>> from transformers import AutoProcessor
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")

>>> with torch.no_grad():
...     encoding = processor(image.convert("RGB"), question, return_tensors="pt")
...     outputs = model(**encoding)
...     start_logits = outputs.start_logits
...     end_logits = outputs.end_logits
...     predicted_start_idx = start_logits.argmax(-1).item()
...     predicted_end_idx = end_logits.argmax(-1).item()

>>> processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1])
'lee a. waller'
```