<!--版权所有 2022 HuggingFace团队。保留所有权利。

根据Apache许可证第2版 （“许可证”）进行许可;除非符合许可证的要求，否则不得使用此文件。
您可以在

http://www.apache.org/licenses/LICENSE-2.0

处获取许可证的副本。

除非适用法律要求或书面同意，否则根据许可证分发的软件是“按原样”分发的，
没有任何明示或暗示的保证或条件。有关的许可证的特定语言下所载明的权利和限制，
请参阅该许可证。

注意：此文件使用Markdown，但包含我们的doc-builder的特定语法（类似于MDX），可能无法在Markdown查看器中正确渲染。

-->

# 使用脚本进行训练

除了🤗 Transformers [notebooks](./noteboks/README)外，还有一些演示如何使用[PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch)，[TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow)或[JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax)训练模型的示例脚本。

您还会发现我们在[研究项目](https://github.com/huggingface/transformers/tree/main/examples/research_projects)和[旧示例](https://github.com/huggingface/transformers/tree/main/examples/legacy)中使用的脚本，它们主要是由社区贡献的。这些脚本不再得到维护，并且需要特定版本的🤗 Transformers，这可能与库的最新版本不兼容。

这些示例脚本不是预计在每个问题上都可以直接使用，您可能需要根据您尝试解决的问题来调整脚本。为了帮助您进行调整，大多数脚本完全显示了如何预处理数据，允许您根据需要进行编辑以适应您的用例。

如果您想在示例脚本中实现某个功能，请在提交Pull请求之前在[论坛](https://discuss.huggingface.co/)或[问题](https://github.com/huggingface/transformers/issues)中讨论。虽然我们欢迎修复错误，但我们不太可能合并一个在可读性方面具有更多功能但牺牲性能的Pull请求。

这个指南将向您展示如何在[PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)和[TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization)中运行一个示例摘要训练脚本。除非另有说明，所有示例都可以在这两个框架上工作。

## 设置

要成功运行示例脚本的最新版本，您必须在新的虚拟环境中**从源代码安装🤗 Transformers**：

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

对于旧版本的示例脚本，请点击下面的切换：

<details>
  <summary>早期版本的🤗 Transformers示例</summary>
	<ul>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples">v4.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples">v4.4.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples">v4.3.3</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples">v4.2.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples">v4.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples">v4.0.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples">v3.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples">v3.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples">v3.3.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples">v3.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples">v3.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples">v3.0.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples">v2.11.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples">v2.10.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples">v2.9.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples">v2.8.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples">v2.7.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples">v2.6.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples">v2.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples">v2.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples">v2.3.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples">v2.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.1.1/examples">v2.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples">v2.0.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples">v1.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples">v1.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples">v1.0.0</a></li>
	</ul>
</details>

然后将您当前的🤗 Transformers克隆切换到特定版本，例如v3.5.1：

```bash
git checkout tags/v3.5.1
```

设置正确的库版本后，切换到所选示例文件夹，并安装示例特定要求：

```bash
pip install -r requirements.txt
```

## 运行脚本

<frameworkcontent>
<pt>
示例脚本下载并预处理了🤗 [Datasets](https://huggingface.co/docs/datasets/)库中的数据集。然后，脚本使用支持摘要生成的架构在[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)上微调了数据集。以下示例演示了如何在[CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)数据集上微调[T5-small](https://huggingface.co/t5-small)模型。由于T5模型的训练方式，需要添加额外的`source_prefix`参数。这个prompt让T5知道这是一个摘要任务。

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>
示例脚本下载并预处理了🤗 [Datasets](https://huggingface.co/docs/datasets/)库中的数据集。然后，脚本使用Keras在支持摘要生成的架构上微调了数据集。以下示例演示了如何在[CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)数据集上微调[T5-small](https://huggingface.co/t5-small)模型。由于T5模型的训练方式，需要添加额外的`source_prefix`参数。这个prompt让T5知道这是一个摘要任务。

```bash
python examples/tensorflow/summarization/run_summarization.py  \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## 分布式训练和混合精度

[训练器](https://huggingface.co/docs/transformers/main_classes/trainer)支持分布式训练和混合精度，这意味着您也可以在脚本中使用它们。要启用这两个功能：

- 添加`fp16`参数以启用混合精度。
- 使用`nproc_per_node`参数设置要使用的GPU数量。

```bash
python -m torch.distributed.launch \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

TensorFlow脚本使用[`MirroredStrategy`](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy)进行分布式训练，在训练脚本中不需要添加任何额外参数。如果可用，TensorFlow脚本将默认使用多个GPU。

## 在TPU上运行脚本

<frameworkcontent>
<pt>
提供tensor处理单元（Tensor Processing Units，TPUs）是为了加速性能而专门设计的。PyTorch使用[XLA](https://www.tensorflow.org/xla)深度学习编译器来支持TPU（有关更多详细信息，请参阅[这里](https://github.com/pytorch/xla/blob/master/README.md)）。要使用TPU，请运行`xla_spawn.py`脚本，并使用`num_cores`参数设置要使用的TPU核心数。

```bash
python xla_spawn.py --num_cores 8 \
    summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>
提供tensor处理单元（Tensor Processing Units，TPUs）是为了加速性能而专门设计的。TensorFlow脚本使用[`TPUStrategy`](https://www.tensorflow.org/guide/distributed_training#tpustrategy)进行TPU上的训练。要使用TPU，请将TPU资源的名称传递给`tpu`参数。

```bash
python run_summarization.py  \
    --tpu name_of_tpu_resource \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## 使用🤗 Accelerate运行脚本

🤗 [Accelerate](https://huggingface.co/docs/accelerate)是仅适用于PyTorch的库，它提供了一种统一的方法，在保持对PyTorch训练循环完全可见的同时，在多种设置（仅CPU、多个GPU、TPU）上训练模型。如果您尚未安装🤗 Accelerate，请确保已安装：

> 注意：由于Accelerate的快速开发，必须安装accelerate的git版本来运行脚本
```bash
pip install git+https://github.com/huggingface/accelerate
```

不再使用`run_summarization.py`脚本，而是使用`run_summarization_no_trainer.py`脚本。支持🤗 Accelerate的脚本将在文件夹中具有一个`task_no_trainer.py`文件。首先运行以下命令以创建并保存配置文件：

```bash
accelerate config
```

测试您的设置以确保其正确配置：

```bash
accelerate test
```

现在，可以启动培训：

```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
```

## 使用自定义数据集

摘要脚本支持自定义数据集，只要它们是CSV或JSON Line文件。当使用自己的数据集时，您需要指定一些额外的参数：

- `train_file`和`validation_file`指定训练和验证文件的路径。
- `text_column`是输入要进行摘要的文本。
- `summary_column`是要输出的目标文本。

使用自定义数据集的摘要脚本如下：

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --text_column text_column_name \
    --summary_column summary_column_name \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

## 测试脚本

在提交整个可能需要几小时才能完成的数据集之前，通常最好先在较小数量的数据集示例上运行脚本以确保一切正常。使用以下参数将数据集截断为最大样本数：

- `max_train_samples`
- `max_eval_samples`
- `max_predict_samples`

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

并不是所有示例脚本都支持`max_predict_samples`参数。如果不确定脚本是否支持该参数，请添加`-h`参数检查：

```bash
examples/pytorch/summarization/run_summarization.py -h
```

## 从检查点恢复训练

在训练中断时，从先前的检查点恢复训练是一个有用的选项，这样可以确保您可以继续之前的工作，而不是从头开始。从检查点恢复训练有两种方法。

第一种方法使用`output_dir previous_output_dir`参数从存储在`output_dir`中的最新检查点恢复训练。在这种情况下，您应该删除`overwrite_output_dir`：

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --output_dir previous_output_dir \
    --predict_with_generate
```

第二种方法使用 `resume_from_checkpoint path_to_specific_checkpoint` 参数从特定的检查点文件夹恢复训练。

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --resume_from_checkpoint path_to_specific_checkpoint \
    --predict_with_generate
```

## 共享你的模型

所有的脚本都可以将你的最终模型上传到 [Model Hub](https://huggingface.co/models)。在开始之前，确保你已登录到 Hugging Face：

```bash
huggingface-cli login
```

然后将 `push_to_hub` 参数添加到脚本中。该参数将在 `output_dir` 中创建一个包含你的 Hugging Face 用户名和文件夹名的仓库。

如果要为你的仓库指定一个特定的名称，请使用 `push_to_hub_model_id` 参数添加它。该仓库将自动列在你的命名空间下。

以下示例展示了如何上传具有特定仓库名称的模型：

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --push_to_hub \
    --push_to_hub_model_id finetuned-t5-cnn_dailymail \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```