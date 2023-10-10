<!--版权所有2022年的拥抱面阅读组。 版权所有。

根据Apache许可证2.0版（“许可证”），除非符合许可证的要求，否则你不得使用此文件。
你可以在以下位置获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，按“原样”分发的软件在
"AS IS" BASIS，不提供任何明示或暗示的保证或条件。请参阅许可证
特定语言的参数和限制。
⚠️注意，此文件采用Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），你的Markdown查看器可能无法正确呈现。-->

# 分享模型

最后两个教程展示了如何使用PyTorch，Keras和🤗加速在分布式环境中对模型进行微调。下一步是与社区共享你的模型！在Hugging Face，我们相信公开共享知识和资源，以使人工智能民主化。我们鼓励你考虑与社区分享你的模型，以帮助他人节省时间和资源。

在本教程中，你将学习两种在[Model Hub（模型中心）](https://huggingface.co/models)上分享训练或微调的模型的方法：

- 以编程方式将文件推送到Hub。
- 使用Web界面将文件拖放到Hub。

<iframe width="560" height="315" src="https://www.youtube.com/embed/XvSGPZFEjDY" title="YouTube视频播放器"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
picture-in-picture" allowfullscreen></iframe>

<Tip>

要与社区分享模型，你需要在[huggingface.co](https://huggingface.co/join)上拥有帐户。你还可以加入现有组织或创建一个新组织。

</Tip>

## 存储库功能

模型中心（Model Hub）上的每个存储库都像一个典型的GitHub存储库一样。我们的存储库提供版本控制、提交历史记录以及可视化差异的功能。

模型中心（Model Hub）内置的版本控制是基于git和[git-lfs](https://git-lfs.github.com/)的。换句话说，你可以将一个模型视为一个存储库，实现更高级的访问控制和可扩展性。版本控制允许*修订*，即通过提交哈希、标签或分支来固定模型的特定版本。

因此，可以使用“revision”参数加载特定的模型版本：

```py
>>> model = AutoModel.from_pretrained(
...     "julien-c/EsperBERTo-small", revision="v2.0.1"  # 标签名、分支名或提交哈希
... )
```

存储库中的文件也易于编辑，你可以查看提交历史记录以及差异：

![vis_diff](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png)

## 设置

在将模型分享到Hub之前，你需要准备好Hugging Face凭据。如果你可以访问终端，请在安装了🤗 Transformers的虚拟环境中运行以下命令。这将把访问token存储在你的Hugging Face缓存文件夹中（默认为`~/.cache/`）：

```bash
huggingface-cli login
```

如果你使用的是Jupyter或Colaboratory等笔记本，请确保已安装[`huggingface_hub`](https://huggingface.co/docs/hub/adding-a-library)库。此库允许你以编程方式与Hub进行交互。

```bash
pip install huggingface_hub
```

然后使用`notebook_login`登录到Hub，并点击[这里](https://huggingface.co/settings/token)生成一个token以进行登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 将模型转换为所有框架

为了确保其他使用不同框架的人也能使用你的模型，我们建议你将模型转换并上传为PyTorch和TensorFlow的检查点。虽然如果你跳过此步骤，用户仍然可以从其他框架加载你的模型，但速度会较慢，因为🤗 Transformers需要动态转换检查点。

转换另一个框架的检查点非常简单。确保你已安装了PyTorch和TensorFlow（请参见[此处](installation.md)的安装说明），然后在另一个框架中找到你任务的特定模型。

<frameworkcontent>
<pt>
将`from_tf=True`指定为从TensorFlow转换为PyTorch的检查点：

```py
>>> pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
>>> pt_model.save_pretrained("path/to/awesome-name-you-picked")
```
</pt>
<tf>
将`from_pt=True`指定为从PyTorch转换为TensorFlow的检查点：

```py
>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
```

然后，你可以使用新的检查点保存你的新TensorFlow模型：

```py
>>> tf_model.save_pretrained("path/to/awesome-name-you-picked")
```
</tf>
<jax>
如果在Flax中可用一个模型，你还可以将PyTorch的检查点转换为Flax的检查点：

```py
>>> flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
...     "path/to/awesome-name-you-picked", from_pt=True
... )
```
</jax>
</frameworkcontent>

## 在训练过程中推送模型

<frameworkcontent>
<pt>
<Youtube id="Z1-XMy-GNLQ"/>

将模型推送到Hub就像添加一个额外的参数或回调一样简单。在[微调教程](training.md)中，你在`TrainingArguments`(TrainingArguments)类中指定超参数和其他训练选项。其中一个训练选项包括直接将模型推送到Hub的功能。在[`TrainingArguments`]中将`push_to_hub=True`：

```py
>>> training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
```

像往常一样将训练参数传递给[`Trainer`]：

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

在微调模型之后，调用[`Trainer`]上的[`~transformers.Trainer.push_to_hub`]将训练好的模型推送到Hub。🤗 Transformers 甚至会自动将训练超参数、训练结果和框架版本添加到模型卡片中！

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
使用[`PushToHubCallback`]将模型推送到Hub。在[`PushToHubCallback`]函数中添加以下内容：

- 模型的输出目录。
- 分词器。
- `hub_model_id`，即你的Hub用户名和模型名称。

```py
>>> from transformers import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
... )
```

将回调添加到[`fit`](https://keras.io/api/models/model_training_apis/)中，🤗 Transformers 将把训练好的模型推送到Hub：

```py
>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```
</tf>
</frameworkcontent>

## 使用`push_to_hub`功能

你也可以直接在模型上调用`push_to_hub`，将其上传到Hub。

在`push_to_hub`中指定你的模型名称：

```py
>>> pt_model.push_to_hub("my-awesome-model")
```

这将在你的用户名下创建一个模型存储库，模型名称为`my-awesome-model`。现在用户可以使用`from_pretrained`函数加载你的模型：

```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("your_username/my-awesome-model")
```

如果你属于一个组织，并希望将模型推送到组织名称下，只需将其添加到`repo_id`中：

```py
>>> pt_model.push_to_hub("my-awesome-org/my-awesome-model")
```

`push_to_hub`功能还可以用于向模型存储库添加其他文件。例如，将分词器添加到模型存储库：

```py
>>> tokenizer.push_to_hub("my-awesome-model")
```

或者也许你希望添加你经过微调的PyTorch模型的TensorFlow版本：

```py
>>> tf_model.push_to_hub("my-awesome-model")
```

现在，当你导航到你的Hugging Face个人资料时，你应该会看到你新创建的模型存储库。点击**Files**标签将显示你上传到存储库的所有文件。

有关如何创建和上传文件到存储库的详细信息，请参阅[此处](https://huggingface.co/docs/hub/how-to-upstream)的Hub文档。

## 使用Web界面上传

倾向于无代码方法的用户可以通过Hub的Web界面上传模型。访问[huggingface.co/new](https://huggingface.co/new)创建一个新的存储库：

![new_model_repo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png)

从这里，提供关于你的模型的一些信息：

- 选择存储库的**所有者**。这可以是你自己或你所属的任何组织。
- 为你的模型选择一个名称，这也将是存储库的名称。
- 选择模型是公共的还是私有的。
- 指定你的模型的许可使用。

现在，点击**Files**标签，然后点击**Add file**按钮将一个新文件上传到你的存储库。然后将文件拖放到上传区域，并添加提交消息。

![upload_file](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png)

## 添加模型卡片

为了确保用户了解模型的功能、限制、潜在偏差和道德考虑，请在存储库中添加一个模型卡片。模型卡片在`README.md`文件中定义。你可以通过以下方式添加模型卡片：

* 手动创建和上传`README.md`文件。
* 在模型存储库中点击**编辑模型卡片**按钮。

查看DistilBert的[模型卡片](https://huggingface.co/distilbert-base-uncased)以获得模型卡片应包括的信息类型的良好示例。有关如何在`README.md`文件中控制模型的其他选项（例如模型的碳足迹或小部件示例）的更多详细信息，请参阅[此处](https://huggingface.co/docs/hub/models-cards)的文档。