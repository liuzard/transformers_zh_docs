<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 使用🤗 Accelerate进行分布式训练

随着模型越来越大，使用并行计算已成为在有限硬件上训练更大模型和加速训练速度的策略。在Hugging Face，我们创建了[🤗 Accelerate](https://huggingface.co/docs/accelerate)库，帮助用户轻松地在任何类型的分布式设置中训练🤗 Transformers模型，无论是在一台机器上的多个GPU还是跨多台机器的多个GPU。在本教程中，您将学习如何自定义原生PyTorch训练循环以在分布式环境中进行训练。

## 设置

首先，安装🤗 Accelerate：

```bash
pip install accelerate
```

然后导入并创建一个[`~accelerate.Accelerator`]对象。[`~accelerate.Accelerator`]会自动检测您的分布式设置类型，并初始化所有必要的组件进行训练。您不需要显式地将模型放置在设备上。

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## 准备加速

下一步是将所有相关的训练对象传递给[`~accelerate.Accelerator.prepare`]方法。这包括您的训练和评估DataLoader、一个模型和一个优化器：

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## 反向传播

最后一个修改是将训练循环中典型的`loss.backward()`替换为🤗 Accelerate的[`~accelerate.Accelerator.backward`]方法：

```py
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         outputs = model(**batch)
...         loss = outputs.loss
...         accelerator.backward(loss)

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

正如您在下面的代码中所看到的，您只需要添加四行额外的代码到您的训练循环中就可以启用分布式训练！

```diff
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## 训练

一旦您添加了相关的代码行，就可以在脚本或笔记本（例如Colaboratory）中启动训练。

### 使用脚本进行训练

如果您从脚本中运行训练，请运行以下命令创建并保存一个配置文件：

```bash
accelerate config
```

然后使用以下命令启动训练：

```bash
accelerate launch train.py
```

### 使用笔记本进行训练

🤗 Accelerate也可以在笔记本中运行，如果您计划使用Colaboratory的TPU。将负责训练的所有代码封装在一个函数中，并将其传递给[`~accelerate.notebook_launcher`]：

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

有关🤗 Accelerate及其丰富功能的更多信息，请参阅[文档](https://huggingface.co/docs/accelerate)。
