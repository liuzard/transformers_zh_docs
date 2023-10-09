```py
# Step 1: Define training hyperparameters
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[100, 300, 500, 1000, 3000],
    values=[1e-4, 5e-5, 3e-5, 1e-5, 1e-6, 1e-7])


# Step 2: Instantiate a pretrained model
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)


# Step 3: Convert the 🤗 Dataset to a `tf.data.Dataset`
train_dataset = train_ds.to_tf_dataset(with_transform=train_transforms)


# Step 4: Compile your model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer)


# Step 5: Add callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoint.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
)


# Step 6: Use `fit()` to run training
model.fit(
    train_dataset,
    validation_data=test_ds,
    epochs=50,
    callbacks=[checkpoint_callback],
)


# Step 7: Save the model to Hub
model.save_pretrained("segformer-b0-scene-parse-150-tf")
```
</tf>
</frameworkcontent>

这是关于语义分割的指南，它将每个像素分配给一个标签或类别。语义分割有几种类型，在语义分割的情况下，不区分相同对象的唯一实例。两个对象被赋予相同的标签（例如，“car”而不是“car-1”和“car-2”）。实际应用包括培训自动驾驶汽车识别行人和重要交通信息，识别医学图像中的细胞和异常，以及监测卫星图像中的环境变化。

此指南将展示如何进行以下操作：

1. 在SceneParse150数据集上微调SegFormer。
2. 使用微调的模型进行推理。

在开始之前，请确保已安装所有必要的库：

```bash
pip install -q datasets transformers evaluate
```

我们鼓励您登录您的Hugging Face帐户，以便您可以将模型上传和共享给社区。在提示时，输入您的令牌进行登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载SceneParse150数据集

首先从🤗数据集库中加载SceneParse150数据集的较小子集。在使用完整数据集进行更长时间的训练之前，这将为您提供实验和确保一切正常的机会。

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
```

使用[`~datasets.Dataset.train_test_split`]方法将数据集的`train`拆分为训练集和测试集：

```py
>>> ds = ds.train_test_split(test_size=0.2)
>>> train_ds = ds["train"]
>>> test_ds = ds["test"]
```

然后查看一个示例：

```py
>>> train_ds[0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>,
 'annotation': <PIL.PngImagePlugin.PngImageFile image mode=L size=512x683 at 0x7F9B0C201DD0>,
 'scene_category': 368}
```

- `image`：场景的PIL图像。
- `annotation`：分割地图的PIL图像，也是模型的目标。
- `scene_category`：描述图像场景的类别ID，如“kitchen”或“office”。在本指南中，您只需要`image`和`annotation`，它们都是PIL图像。

您还需要创建一个字典，将标签id映射到标签类别，这在设置模型时将很有用。从Hub下载映射并创建`id2label`和`label2id`字典：

```py
>>> import json
>>> from huggingface_hub import cached_download, hf_hub_url

>>> repo_id = "huggingface/label-files"
>>> filename = "ade20k-id2label.json"
>>> id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
>>> id2label = {int(k): v for k, v in id2label.items()}
>>> label2id = {v: k for k, v in id2label.items()}
>>> num_labels = len(id2label)
```

## 预处理

下一步是加载SegFormer图像处理器，以准备图像和注释供模型使用。一些数据集（例如此数据集）使用零索引作为背景类别。但是，实际上背景类别不包括在这150个类别中，因此您需要设置`reduce_labels=True`，将所有标签减1。零索引由`255`替换，因此SegFormer的损失函数会忽略它：

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "nvidia/mit-b0"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
```

<frameworkcontent>
<pt>

通常在图像数据集上应用一些数据增强方法，以使模型对过拟合更具鲁棒性。在本指南中，您将使用[`ColorJitter`](https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html)函数，随机更改图像的颜色属性，但您也可以使用您喜欢的任何图像库。

```py
>>> from torchvision.transforms import ColorJitter

>>> jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
```

现在创建两个预处理函数，以准备图像和注释供模型使用。这些函数将图像转换为`pixel_values`，将注释转换为`labels`。对于训练集，在将图像提供给图像处理器之前，应用`jitter`。对于测试集，图像处理器对`images`进行裁剪和归一化，只裁剪`labels`，因为在测试期间不应用数据增强。

```py
>>> def train_transforms(example_batch):
...     images = [jitter(x) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [x for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

使用🤗数据集[`~datasets.Dataset.set_transform`]函数在整个数据集上应用`jitter`。变换是即时应用的，速度更快，占用的磁盘空间更少：

```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

通常，在图像数据集上应用一些数据增强方法可以提高模型对过拟合的鲁棒性。
在本指南中，您将使用[`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image)随机更改图像的颜色属性，但您也可以使用您喜欢的任何图像库。
请定义两个不同的转换函数：
- 包含图像增强的训练数据转换
- 仅转置图像的验证数据转换，因为🤗 Transformers中的计算机视觉模型需要以通道优先的布局（channels-first layout）

```py
>>> import tensorflow as tf


>>> def aug_transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.image.random_brightness(image, 0.25)
...     image = tf.image.random_contrast(image, 0.5, 2.0)
...     image = tf.image.random_saturation(image, 0.75, 1.25)
...     image = tf.image.random_hue(image, 0.1)
...     image = tf.transpose(image, (2, 0, 1))
...     return image


>>> def transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.transpose(image, (2, 0, 1))
...     return image
```

接下来，创建两个预处理函数，以准备图像和注释的批次供模型使用。这些函数应用图像转换，并使用之前加载的`image_processor`将图像转换为`pixel_values`和注释转换为`labels`。`ImageProcessor`还负责调整大小和归一化图像。

```py
>>> def train_transforms(example_batch):
...     images = [aug_transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

使用🤗数据集[`~datasets.Dataset.set_transform`]函数在整个数据集上应用预处理转换。变换是即时应用的，速度更快，占用的磁盘空间更少：

```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```
</tf>
</frameworkcontent>

## 评估

在训练过程中包含一个度量指标通常有助于评估模型的性能。您可以使用🤗 [Evaluate](https://huggingface.co/docs/evaluate/index)库快速加载一个评估方法。对于此任务，加载[mean Intersection over Union](https://huggingface.co/spaces/evaluate-metric/accuracy)（IoU）度量指标（请参阅🤗 Evaluate [快速入门](https://huggingface.co/docs/evaluate/a_quick_tour)了解如何加载和计算度量指标）：

```py
>>> import evaluate

>>> metric = evaluate.load("mean_iou")
```

然后创建一个函数来[`~evaluate.EvaluationModule.compute`]度量指标。首先需要将预测转换为logits，然后将其重塑为与标签大小相匹配，然后才能调用[`~evaluate.EvaluationModule.compute`]：

<frameworkcontent>
<pt>

```py
>>> import numpy as np
>>> import torch
>>> from torch import nn

>>> def compute_metrics(eval_pred):
...     with torch.no_grad():
...         logits, labels = eval_pred
...         logits_tensor = torch.from_numpy(logits)
...         logits_tensor = nn.functional.interpolate(
...             logits_tensor,
...             size=labels.shape[-2:],
...             mode="bilinear",
...             align_corners=False,
...         ).argmax(dim=1)

...         pred_labels = logits_tensor.detach().cpu().numpy()
...         metrics = metric.compute(
...             predictions=pred_labels,
...             references=labels,
...             num_labels=num_labels,
...             ignore_index=255,
...             reduce_labels=False,
...         )
...         for key, value in metrics.items():
...             if type(value) is np.ndarray:
...                 metrics[key] = value.tolist()
...         return metrics
```

</pt>
</frameworkcontent>


<frameworkcontent>
<tf>

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     logits = tf.transpose(logits, perm=[0, 2, 3, 1])
...     logits_resized = tf.image.resize(
...         logits,
...         size=tf.shape(labels)[1:],
...         method="bilinear",
...     )

...     pred_labels = tf.argmax(logits_resized, axis=-1)
...     metrics = metric.compute(
...         predictions=pred_labels,
...         references=labels,
...         num_labels=num_labels,
...         ignore_index=-1,
...         reduce_labels=image_processor.do_reduce_labels,
...     )

...     per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
...     per_category_iou = metrics.pop("per_category_iou").tolist()

...     metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
...     metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
...     return {"val_" + k: v for k, v in metrics.items()}
```

</tf>
</frameworkcontent>

现在您的`compute_metrics`函数已准备就绪，请在设置训练时返回。

## 训练
<frameworkcontent>
<pt>
<Tip>

如果您对使用[`Trainer`]进行模型微调不熟悉，请先查看基本教程[这里](../training.md#finetune-with-trainer)！

</Tip>

现在可以开始训练模型了！使用[`AutoModelForSemanticSegmentation`]加载SegFormer，并将模型与标签id和标签类别之间的映射传递给模型：

```py
>>> from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

>>> model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
```

此时，只剩下三个步骤：

1. 在[`TrainingArguments`]中定义训练超参数。重要的是不要删除未使用的列，因为这将删除`image`列。没有`image`列，您无法创建`pixel_values`。将`remove_unused_columns=False`设置为防止此行为！仅其他必需的参数是`output_dir`，它指定保存模型的位置。设置`push_to_hub=True`将此模型推送到Hub（需要登录Hugging Face以上传您的模型）。在每个epoch结束时，[`Trainer`]将评估IoU度量并保存训练检查点。
2. 将训练参数以及模型、数据集、tokenizer、数据收集器和`compute_metrics`函数传递给[`Trainer`]。
3. 调用[`~Trainer.train`]开始微调模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="segformer-b0-scene-parse-150",
...     learning_rate=6e-5,
...     num_train_epochs=50,
...     per_device_train_batch_size=2,
...     per_device_eval_batch_size=2,
...     save_total_limit=3,
...     evaluation_strategy="steps",
...     save_strategy="steps",
...     save_steps=20,
...     eval_steps=20,
...     logging_steps=1,
...     eval_accumulation_steps=5,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=test_ds,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

完成训练后，请使用[`~transformers.Trainer.push_to_hub`]方法将模型分享到Hub，以便每个人都可以使用您的模型：

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
<Tip>

如果您熟悉使用Keras进行微调模型，请先参阅[基本教程](./training#train-a-tensorflow-model-with-keras)！

</Tip>

要在TensorFlow中微调模型，请按照以下步骤进行：
1. 定义训练超参数，并设置优化器和学习率计划。
2. 实例化预训练模型。
3. 将🤗数据集转换为`tf.data.Dataset`。
4. 编译模型。
5. 添加回调函数来计算指标和上传模型到🤗 Hub。
6. 使用`fit()`方法来运行训练。

```py
# 步骤1：定义训练超参数
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[100, 300, 500, 1000, 3000],
    values=[1e-4, 5e-5, 3e-5, 1e-5, 1e-6, 1e-7])


# 步骤2：实例化预训练模型
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)


# 步骤3：将🤗 Dataset转换为`tf.data.Dataset`
train_dataset = train_ds.to_tf_dataset(with_transform=train_transforms)


# 步骤4：编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer)


# 步骤5：添加回调函数
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoint.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
)


# 步骤6：使用`fit()`方法进行训练
model.fit(
    train_dataset,
    validation_data=test_ds,
    epochs=50,
    callbacks=[checkpoint_callback],
)


# 步骤7：将模型保存到Hub
model.save_pretrained("segformer-b0-scene-parse-150-tf")
```
</tf>
</frameworkcontent>

```markdown
### 推理

好的，既然您已经微调了模型，那么可以用它进行推理！

加载用于推理的图像：

```py
image = ds[0]["image"]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-image.png" alt="Image of bedroom"/>
</div>

<frameworkcontent>
<pt>
尝试使用模型进行推理的最简单的方法是使用[`pipeline`]。使用模型实例化一个图像分割的`pipeline`，然后将图像传递给它：

```py
from transformers import pipeline

segmenter = pipeline("image-segmentation", model="my_awesome_seg_model")
segmenter(image)
[{'score': None,
  'label': 'wall',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062690>},
 {'score': None,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A50>},
 {'score': None,
  'label': 'floor',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062B50>},
 {'score': None,
  'label': 'ceiling',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A10>},
 {'score': None,
  'label': 'bed ',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E90>},
 {'score': None,
  'label': 'windowpane',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062390>},
 {'score': None,
  'label': 'cabinet',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062550>},
 {'score': None,
  'label': 'chair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062D90>},
 {'score': None,
  'label': 'armchair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E10>}]
```

如果需要，您还可以手动复制`pipeline`的结果。使用图像处理器处理图像，并将`pixel_values`放在GPU上：

```py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果有可用的GPU，则使用GPU，否则使用CPU
encoding = image_processor(image, return_tensors="pt")
pixel_values = encoding.pixel_values.to(device)
```

将输入传递给模型并返回`logits`:

```py
outputs = model(pixel_values=pixel_values)
logits = outputs.logits.cpu()
```

接下来，将`logits`调整到原始图像的大小：

```py
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
加载图像处理器以预处理图像并以TensorFlow张量的形式返回输入：

```py
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("MariaK/scene_segmentation")
inputs = image_processor(image, return_tensors="tf")
```

将输入传递给模型并返回`logits`:

```py
from transformers import TFAutoModelForSemanticSegmentation

model = TFAutoModelForSemanticSegmentation.from_pretrained("MariaK/scene_segmentation")
logits = model(**inputs).logits
```

接下来，将`logits`调整到原始图像的大小，并对类维度应用`argmax`函数：
```py
logits = tf.transpose(logits, [0, 2, 3, 1])

upsampled_logits = tf.image.resize(
    logits,
    # 由于`image.size`返回宽度和高度，所以我们颠倒了`image`的形状。
    image.size[::-1],
)

pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0]
```

</tf>
</frameworkcontent>

要可视化结果，加载[数据集颜色调色板](https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51)作为`ade_palette()`将每个类别映射到RGB值。然后，您可以将图像和预测的分割图组合在一起并绘制出来：

```py
import matplotlib.pyplot as plt
import numpy as np

color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
palette = np.array(ade_palette())
for label, color in enumerate(palette):
    color_seg[pred_seg == label, :] = color
color_seg = color_seg[..., ::-1]  # 转换为BGR

img = np.array(image) * 0.5 + color_seg * 0.5  # 将图像与分割图重叠显示
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-preds.png" alt="Image of bedroom overlaid with segmentation map"/>
</div>
```