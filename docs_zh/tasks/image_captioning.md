<!--版权2023年HuggingFace团队，保留所有权利。

根据Apache许可证，版本2.0（“许可证”）进行许可；除非符合许可证，否则不得使用此文件。您可以在以下网址获取许可证的副本。

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证的“原样”分发，不附带任何担保或条件。请查阅许可证以获取特定语言下免责声明和限制。

⚠️请注意，此文件采用Markdown格式，但包含我们的文档构建器（类似于MDX）的特定语法，可能无法在Markdown查看器中正确呈现。-->

# 图像标注

[[open-in-colab]]

图像标注是预测给定图像的标题的任务。它在现实世界的常见应用包括帮助视障人士通过不同情境进行导航。因此，图像标注通过描述图像来提高人们的内容可访问性。

本指南将介绍以下内容：

* 微调图像标注模型。
* 使用已微调的模型进行推理。

开始之前，请确保已安装所有必要的库：

```bash
pip install transformers datasets evaluate -q
pip install jiwer -q
```

我们鼓励您登录Hugging Face账户，这样您可以与社区上传并分享您的模型。在提示时，输入您的令牌进行登录：

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 加载Pokémon BLIP标题数据集

使用🤗数据集库加载一个由{图像-标题}对组成的数据集。要在PyTorch中创建自己的图像标注数据集，可以参考[此笔记本](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb)。

```python
from datasets import load_dataset

ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds
```
```bash
DatasetDict({
    train: Dataset({
        features: ['image', 'text'],
        num_rows: 833
    })
})
```

数据集有两个特征，`image`和`text`。

<Tip>

许多图像标注数据集包含图像的多个标题。在这些情况下，一种常见的策略是在训练过程中从可用的标题中随机选择一个标题。

</Tip>

使用[~datasets.Dataset.train_test_split]方法将数据集的训练集分割成训练集和测试集：

```python
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]
```

让我们从训练集中可视化几个样本。

```python
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np

def plot_images(images, captions):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        plt.title(caption)
        plt.imshow(images[i])
        plt.axis("off")

sample_images_to_visualize = [np.array(train_ds[i]["image"]) for i in range(5)]
sample_captions = [train_ds[i]["text"] for i in range(5)]
plot_images(sample_images_to_visualize, sample_captions)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_training_images_image_cap.png" alt="Sample training images"/>
</div>

## 预处理数据集

由于数据集有两种模态（图像和文本），预处理流水线将预处理图像和标题。

要做到这一点，加载与您即将微调的模型相关联的处理器类。

```python
from transformers import AutoProcessor

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)
```

处理器将内部预处理图像（包括调整大小和像素缩放）并对标题进行分词。

```python
def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs

train_ds.set_transform(transforms)
test_ds.set_transform(transforms)
```

准备好数据集后，您现在可以为微调设置模型了。

## 加载基础模型

将["microsoft/git-base"](https://huggingface.co/microsoft/git-base)加载到[`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM)对象中。

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

## 评估

图像标注模型通常使用[Rouge Score](https://huggingface.co/spaces/evaluate-metric/rouge)或[Word Error Rate](https://huggingface.co/spaces/evaluate-metric/wer)进行评估。对于本指南，您将使用Word Error Rate (WER)。

我们使用🤗 Evaluate库来实现这一点。关于WER的潜在限制和其他内容，请参考[此指南](https://huggingface.co/spaces/evaluate-metric/wer)。

```python
from evaluate import load
import torch

wer = load("wer")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}
```

## 训练！

现在，您已经准备好开始微调模型了。您将使用🤗[`Trainer`]完成此操作。

首先，使用[`TrainingArguments`]定义训练参数。

```python
from transformers import TrainingArguments, Trainer

model_name = checkpoint.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-pokemon",
    learning_rate=5e-5,
    num_train_epochs=50,
    fp16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    remove_unused_columns=False,
    push_to_hub=True,
    label_names=["labels"],
    load_best_model_at_end=True,
)
```

然后将它们与数据集和模型一起传递给🤗 Trainer。

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
```

要开始训练，只需在[`Trainer`]对象上调用[`~Trainer.train`]即可。

```python
trainer.train()
```

您会看到随着训练的进行，训练损失平稳下降。

训练完成后，使用[`~Trainer.push_to_hub`]方法将您的模型分享到Hub，以便每个人都可以使用您的模型：

```python
trainer.push_to_hub()
```

## 推理

从`test_ds`中取一张样本图像来测试模型。

```python
from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/test_image_image_cap.png" alt="Test image"/>
</div>

为模型准备图像。

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values
```

调用[`generate`]并解码预测结果。

```python
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
```

```bash
a drawing of a pink and blue pokemon
```

看起来微调模型生成了一个很好的标题！