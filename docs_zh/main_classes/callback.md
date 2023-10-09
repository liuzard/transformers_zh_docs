<!--
版权所有2020年The HuggingFace团队。保留所有权利。

根据Apache 2.0许可，您可能不会使用此文件，除非您遵守许可。您可以在以下链接处获得许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件以"原样"分发，不附带任何明示或暗示的保证或条件。请参阅许可证以获取特定语言的权限和限制。

⚠️ 请注意，该文件采用Markdown编写，但包含我们的文档生成器（类似于MDX）的特定语法，可能在您的Markdown查看器中无法正确显示。

-->

# 回调函数

回调函数是可以自定义PyTorch训练循环行为的对象[`Trainer`]（此功能尚未在TensorFlow中实现），可以检查训练循环状态（用于进度报告、在TensorBoard或其他ML平台上记录日志等）并做出决策（如提前停止）。

回调函数是只读代码片段，除了返回的[`TrainerControl`]对象外，它们不能更改训练循环中的任何内容。对于需要更改训练循环的自定义内容，您应该子类化[`Trainer`]并覆盖所需的方法（请参阅[trainer](trainer)了解示例）。

默认情况下，[`Trainer`]将使用以下回调函数：

- [`DefaultFlowCallback`]处理日志记录、保存和评估的默认行为。
- [`PrinterCallback`]或[`ProgressCallback`]显示进度并打印日志（如果通过[`TrainingArguments`]停用tqdm，则使用第一个，否则使用第二个）。
- [`~integrations.TensorBoardCallback`]如果可以访问tensorboard（通过PyTorch >= 1.4或tensorboardX）。
- [`~integrations.WandbCallback`]如果安装了[wandb](https://www.wandb.com/)。
- [`~integrations.CometCallback`]如果安装了[comet_ml](https://www.comet.ml/site/)。
- [`~integrations.MLflowCallback`]如果安装了[mlflow](https://www.mlflow.org/)。
- [`~integrations.NeptuneCallback`]如果安装了[neptune](https://neptune.ai/)。
- [`~integrations.AzureMLCallback`]如果安装了[azureml-sdk](https://pypi.org/project/azureml-sdk/)。
- [`~integrations.CodeCarbonCallback`]如果安装了[codecarbon](https://pypi.org/project/codecarbon/)。
- [`~integrations.ClearMLCallback`]如果安装了[clearml](https://github.com/allegroai/clearml)。
- [`~integrations.DagsHubCallback`]如果安装了[dagshub](https://dagshub.com/)。
- [`~integrations.FlyteCallback`]如果安装了[flyte](https://flyte.org/)。

实现回调函数的主要类是[`TrainerCallback`]。它获取用于实例化[`Trainer`]的[`TrainingArguments`]，可以通过[`TrainerState`]访问该Trainer的内部状态，并可以通过[`TrainerControl`]对训练循环进行一些操作。

## 可用的回调函数

以下是库中可用的[`TrainerCallback`]列表：

[[autodoc]] integrations.CometCallback
    - 设置

[[autodoc]] DefaultFlowCallback

[[autodoc]] PrinterCallback

[[autodoc]] ProgressCallback

[[autodoc]] EarlyStoppingCallback

[[autodoc]] integrations.TensorBoardCallback

[[autodoc]] integrations.WandbCallback
    - 设置

[[autodoc]] integrations.MLflowCallback
    - 设置

[[autodoc]] integrations.AzureMLCallback

[[autodoc]] integrations.CodeCarbonCallback

[[autodoc]] integrations.NeptuneCallback

[[autodoc]] integrations.ClearMLCallback

[[autodoc]] integrations.DagsHubCallback

[[autodoc]] integrations.FlyteCallback

## TrainerCallback

[[autodoc]] TrainerCallback

以下是使用PyTorch [`Trainer`]注册自定义回调函数的示例：

```python
class MyCallback(TrainerCallback):
    "在训练开始时打印消息的回调函数"

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MyCallback],  # 我们可以通过这种方式传递回调函数类，也可以传递其实例（MyCallback()）
)
```

注册回调函数的另一种方法是通过调用`trainer.add_callback()`，如下所示：

```python
trainer = Trainer(...)
trainer.add_callback(MyCallback)
# 或者，我们可以传递回调函数类的实例
trainer.add_callback(MyCallback())
```

## TrainerState

[[autodoc]] TrainerState

## TrainerControl

[[autodoc]] TrainerControl