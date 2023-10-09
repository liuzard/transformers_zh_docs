<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 安装

安装适用于您正在使用的深度学习库的🤗 Transformers模块，设置缓存，并可选择配置🤗 Transformers以离线运行。

🤗 Transformers已经在Python 3.6+、PyTorch 1.1.0+、TensorFlow 2.0+和Flax上进行了测试。请按照以下针对您所使用的深度学习库的安装说明进行操作：

* [PyTorch](https://pytorch.org/get-started/locally/) 安装引导.
* [TensorFlow 2.0](https://www.tensorflow.org/install/pip) 按转引导.
* [Flax](https://flax.readthedocs.io/en/latest/) 安装引导.

## 通过pip安装

在您的项目中，您应该在[虚拟环境](https://docs.python.org/3/library/venv.html)中安装🤗 Transformers。如果您对Python虚拟环境不熟悉，请参考这个[指南](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)。虚拟环境可以更轻松地管理不同的项目，并避免依赖项之间的兼容性问题。

首先，在您的项目目录中创建一个虚拟环境：

```bash
python -m venv .env
```

Activate the virtual environment. On Linux and MacOs:

```bash
source .env/bin/activate
```
激活虚拟环境。在Linux和MacOS上：

```bash
.env/Scripts/activate
```

现在您可以使用以下命令安装🤗 Transformers：

```bash
pip install transformers
```

仅限于CPU支持时，你可以方便地使用一行命令安装🤗 Transformers和一个深度学习库。例如，使用以下命令安装🤗 Transformers和PyTorch：

```bash
pip install 'transformers[torch]'
```

🤗 Transformers和TensorFlow 2.0：

```bash
pip install 'transformers[tf-cpu]'
```

<Tip warning={true}>

M1 / ARM 用户
    
在安装TensorFlow 2.0之前，您需要先安装以下内容：

```
brew install cmake
brew install pkg-config
```

</Tip>

🤗 Transformers 和 Flax:

```bash
pip install 'transformers[flax]'
```

最后，通过运行以下命令来检查🤗 Transformers是否已正确安装。该命令将下载一个预训练模型：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

然后打印出标签和分数：

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## 从源代码安装

使用以下命令从源代码安装🤗 Transformers：

```bash
pip install git+https://github.com/huggingface/transformers
```

此命令安装的是最新的`main`版本，而不是最新的`stable`版本。`main`版本非常适合跟进最新的开发进展。例如，如果自上次官方发布以来修复了一个错误但尚未发布新版本，则可以使用`main`版本获取该修复。然而，这也意味着`main`版本不一定始终稳定。我们努力保持`main`版本的可用性，并且大多数问题通常在几个小时或一天内解决。如果遇到问题，请在[Issue](https://github.com/huggingface/transformers/issues)中提出，以便我们可以更快地修复！

通过运行以下命令检查🤗 Transformers是否已正确安装：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## 可编辑安装

如果您想要：

- 使用源代码的`main`版本。
- 对🤗 Transformers进行贡献并需要测试代码更改。

则需要进行可编辑安装。请使用以下命令克隆存储库并安装🤗 Transformers：

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

这些命令将把您克隆存储库的文件夹链接到Python库路径中。Python现在会在正常的库路径之外，也会查找您克隆到的文件夹内的内容。例如，如果您通常的Python包安装在`~/anaconda3/envs/main/lib/python3.7/site-packages/`中，Python也会搜索您克隆到的文件夹：`~/transformers/`。

<Tip warning={true}>

如果您希望继续使用该库，您必须保留`transformers`文件夹。

</Tip>

现在，您可以使用以下命令轻松更新克隆版本到最新的🤗 Transformers：

```bash
cd ~/transformers/
git pull
```

您的Python环境将在下次运行时找到🤗 Transformers的`main`版本。

## 通过conda安装

I从conda频道`huggingface`安装：

```bash
conda install -c huggingface transformers
```

## 缓存设置：

预训练模型会被下载并本地缓存在`~/.cache/huggingface/hub`目录下。这是由shell环境变量`TRANSFORMERS_CACHE`指定的默认目录。在Windows系统上，默认目录为`C:\Users\username\.cache\huggingface\hub`。您可以按以下优先顺序更改下面显示的shell环境变量，以指定不同的缓存目录：

1. Shell环境变量（默认）：`HUGGINGFACE_HUB_CACHE`或`TRANSFORMERS_CACHE`。
2. Shell环境变量：`HF_HOME`。
3. Shell环境变量：`XDG_CACHE_HOME` + `/huggingface`。

<Tip>

🤗 Transformers将使用shell环境变量`PYTORCH_TRANSFORMERS_CACHE`或`PYTORCH_PRETRAINED_BERT_CACHE`，如果您是从此库的早期版本转换过来并设置了这些环境变量，除非您指定了shell环境变量`TRANSFORMERS_CACHE`。

</Tip>

## 离线模式

🤗 Transformers可以在防火墙或离线环境中运行，只使用本地文件。设置环境变量`TRANSFORMERS_OFFLINE=1`以启用此功能。

<Tip>

通过设置环境变量`HF_DATASETS_OFFLINE=1`，将[🤗 Datasets](https://huggingface.co/docs/datasets/)添加到离线训练工作流程中。

</Tip>

例如，您通常会在一个正常的网络防火墙下运行程序，如下所示：

```bash
python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

在离线实例中运行相同的程序：

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

脚本现在应该能够正常运行，而无需等待超时，因为它只会查找本地文件。

### 离线获取模型和分词器

使用🤗 Transformers的另一种离线方式是预先下载文件，然后在需要离线使用时指向其本地路径。有三种方法可以实现这一点：

* 通过在[Model Hub](https://huggingface.co/models)上点击↓图标，在用户界面上下载文件。

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)

* 使用`PreTrainedModel.from_pretrained`和`PreTrainedModel.save_pretrained`的工作流程：

    1. 预先下载文件并使用`PreTrainedModel.from_pretrained`加载：

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```

    2. 使用`PreTrainedModel.save_pretrained`将文件保存到指定目录：

    ```py
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```

    3. 当您处于离线状态时，使用指定目录下的`PreTrainedModel.from_pretrained`重新加载文件：

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```

* 使用[huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub)库以编程方式下载文件：

    1. 在虚拟环境中安装`huggingface_hub`库：

    ```bash
    python -m pip install huggingface_hub
    ```

    2. 使用[`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub)函数将文件下载到特定路径。例如，以下命令会将[T0](https://huggingface.co/bigscience/T0_3B)模型的`config.json`文件下载到您指定的路径：

    ```py
    >>> from huggingface_hub import hf_hub_download
    
    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```

一旦文件被下载并本地缓存，指定其本地路径以加载和使用它：

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

请参阅[如何从Hub下载文件](https://huggingface.co/docs/hub/how-to-downstream)部分，了解有关下载存储在Hub上的文件的更多详细信息。

</Tip>
