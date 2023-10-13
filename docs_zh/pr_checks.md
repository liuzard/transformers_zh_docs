<!-- ---
版权所有2020年“拥抱面官团”。 版权所有。

根据Apache许可证2.0版（“许可证”）许可;
你不得使用此文件，除非符合许可证的规定。
你可以在以下网址获取许可证的副本

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则该软件将按“原样”分发，
没有明示或暗示的任何保证或条件。
请参阅许可证了解特定语言的管理权限和限制。

⚠️注意，该文件是使用Markdown格式的文件，
但包含特定语法以便用于我们的文档构建器（类似于MDX），
这可能不会在Markdown查看器中正常显示。

-->

# 在拉取请求上进行检查

当你在🤗Transformers上打开拉取请求时，将运行相当数量的检查，
以确保你添加的补丁不会破坏任何现有的内容。这些检查可以分为四类：
- 常规测试
- 文档构建
- 代码和文档格式
- 一般存储库一致性

在本文档中，我们将尝试解释这些不同的检查是什么以及背后的原因，
以及如果其中一个检查在PR中失败了，如何在本地调试它们。

请注意，理想情况下，它们需要你进行开发安装:

```bash
pip install transformers[dev]
```

或者进行可编辑安装：

```bash
pip install -e .[dev]
```

位于Transformers存储库中。由于Transformers的可选依赖项数量增加了很多，
你可能无法获得所有这些依赖项。如果开发安装失败，请确保安装了
你使用的深度学习框架（PyTorch，TensorFlow和/或Flax），然后执行以下操作

```bash
pip install transformers[quality]
```

或者进行可编辑安装：

```bash
pip install -e .[quality]
```

## 测试

所有以`ci/circleci: run_tests_`开头的作业都运行Transformers测试套件的部分内容。这些作业针对特定环境中库的特定部分进行测试：例如，`ci/circleci: run_tests_pipelines_tf`在仅安装了TensorFlow的环境中运行pipelines测试。

请注意，为了避免在测试模块中没有真正更改时运行测试，每次只运行一部分测试套件：运行一个实用工具来确定库在PR之前和之后的差异（GitHub在“文件更改”选项卡中显示的内容），并选择受到该差异影响的测试。可以在本地运行此实用工具：

```bash
python utils/tests_fetcher.py
```

从Transformers存储库的根目录。它将：

1. 对不同的文件检查修改的内容是否是代码还是注释或文档字符串中的更改。仅保留具有实际代码更改的文件。
2. 构建一个内部映射，为库的源代码的每个文件提供递归影响它的所有文件。如果模块B导入模块A，则称模块A影响模块B。对于递归影响，我们需要一个从模块A到模块B的模块链，其中每个模块都导入前一个模块。
3. 在步骤1中获取的文件上应用此映射，从而得到PR影响的模型文件列表。
4. 将每个文件映射到其相应的测试文件，并获取要运行的测试列表。

在本地执行脚本时，你应该能够获得步骤1、3和4的结果，并且知道运行了哪些测试。该脚本还将创建一个名为`test_list.txt`的文件，其中包含要运行的测试列表，你可以使用以下命令在本地运行它们：

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

以防漏掉什么，每日还会运行完整的测试套件。

## 文档构建

`build_pr_documentation`作业将构建和生成一个文档的预览，
以确保在合并你的PR后一切正常。一个机器人将在你的PR中添加一个链接来预览文档。
你对PR所做的任何更改都会自动更新到预览中。如果构建文档失败，
单击失败作业旁边的**Details**以查看出错的位置。通常，错误可能只是`toctree`中缺少文件而已。

如果你对在本地构建或预览文档感兴趣，请查看文档文件夹中的[`README.md`](https://github.com/huggingface/transformers/tree/main/docs)。

## 代码和文档格式

我们使用`black`和`ruff`对所有源代码文件、示例和测试文件应用代码格式。我们还有一个自定义工具负责格式化文档字符串和`rst`文件（`utils/style_doc.py`），以及Transformers `__init__.py`文件中执行的延迟导入的顺序(`utils/custom_init_isort.py`)。
你可以通过运行以下命令来启动这些工具

```bash
make style
```

CI在`ci/circleci: check_code_quality`检查中检查这些是否已应用。它还运行`ruff`，它会基于你的代码进行基本检查，如果找到未定义的变量或未使用的变量，则会发出警告。要在本地运行此检查，请使用以下命令

```bash
make quality
```

这可能需要很长时间，因此，如果要仅对当前分支中修改的文件运行相同的测试，请运行

```bash
make fixup
```

此最后一个命令还会运行存储库一致性的所有其他检查。让我们看一下它们。

## 存储库一致性

这涵盖了所有的测试，以确保你的PR将存储库保持在良好状态，并由`ci/circleci: check_repository_consistency`检查执行。你可以通过执行以下命令在本地运行此检查：

```bash
make repo-consistency
```

这检查以下内容：

- 在init中添加的所有对象都有文档(由`utils/check_repo.py`执行)
- 所有的`__init__.py`文件的两个部分的内容相同（由`utils/check_inits.py`执行）
- 所有被标识为从另一个模块中复制的代码与原始代码保持一致（由`utils/check_copies.py`执行）
- 所有配置类在其文档字符串中至少提及一个有效的检查点（由`utils/check_config_docstrings.py`执行）
- 所有配置类中仅包含在相应的建模文件中使用的属性（由`utils/check_config_attributes.py`执行）
- README和doc中的索引的翻译具有与主README相同的模型列表（由`utils/check_copies.py`执行）
- 文档中的自动生成的表是最新的（由`utils/check_table.py`执行）
- 即使没有安装所有可选依赖项，库也具有所有对象可用（由`utils/check_dummies.py`执行）

如果此检查失败，则前两个项目需要手动修正，后四个项目可以通过运行以下命令自动修复：

```bash
make fix-copies
```

其他检查涵盖了添加新模型的PR，主要包括：

- 所有添加的模型都在Auto-mapping中（由`utils/check_repo.py`执行）
<!-- TODO Sylvain，添加一个检查，确保已实现常见的测试。-->
- 所有模型都经过了适当的测试（由`utils/check_repo.py`执行）

<!-- TODO Sylvain，添加以下内容
- 所有添加的模型都已添加到主README和主文档中
- 所有使用的检查点在Hub上实际存在

-->

### 检查复制

由于Transformers库在模型代码方面非常具体化，每个模型都应完全在一个文件中实现，
而不依赖于其他模型，因此我们添加了一种机制，检查给定模型的特定层的代码的副本是否与原始代码保持一致。这样，当有错误修复时，我们可以看到所有受影响的模型，并选择传递修改或中断副本。

<Tip>

如果一个文件是另一个文件的完全副本，你应该将其注册在`utils/check_copies.py`的常量`FULL_COPIES`中。

</Tip>

此机制依赖于类似于`# Copied from xxx`的注释形式。`xxx`应包含要复制下来的类或函数的完整路径。例如，`RobertaSelfOutput`是`BertSelfOutput`类的直接副本，因此你可以在此[处](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L289)看到一个注释：

```py
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
```

请注意，你可以将其应用于整个类而不是整个类时，也可以应用于从中复制的相关方法。例如，[此处](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L598)可以看到`RobertaPreTrainedModel._init_weights`是从`BertPreTrainedModel`中相同方法的副本，其中也有一个注释：

```py
# Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
```

有时，即使仅有名称不同，复制的代码与原始代码完全相同：例如，在`RobertaAttention`中，我们使用`RobertaSelfAttention`而不是`BertSelfAttention`，但除此之外，代码完全相同。这就是为什么`# Copied from`支持简单的字符串替换，语法如下：`Copied from xxx with foo->bar`。这意味着复制了代码，并将所有的`foo`替换为`bar`。可以在[此处](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L304C1-L304C86)看到`RobertaAttention`中的使用场景：

```py
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
```

请注意，箭头周围不应该有任何空格（除非该空格当然是要替换的一部分）。

你可以用逗号分隔多个模式。例如，`CamemberForMaskedLM`直接从`RobertaForMaskedLM`复制，具有两个替换：`Roberta`到`Camembert`和 `ROBERTA` 到 `CAMEMBERT`。可以在[此处](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/camembert/modeling_camembert.py#L929)看到一个示例，在`RobertaForMaskedLM`中是`RobertaForMaskedLM`，对应的注释是：

```py
# Copied from transformers.models.roberta.modeling_roberta.RobertaForMaskedLM with Roberta->Camembert, ROBERTA->CAMEMBERT
```

如果顺序很重要（因为替换之一可能与之前的替换冲突），则从左到右执行替换。

<Tip>

如果替换更改了格式（例如，将简短名称替换为非常长的名称），则在应用自动格式化程序后检查副本。

</Tip>

当模式只是相同替换的不同大小写变体时（具有一个大写和一个小写变体），最佳选择是添加`all-casing`选项。[这里](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/mobilebert/modeling_mobilebert.py#L1237)是在`MobileBertForSequenceClassification`中的一个示例，它有以下注释：

```py
# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
```

在这种情况下，代码是从`BertForSequenceClassification`复制过来，替换：
- `Bert` 到 `MobileBert`（例如，在init中使用 `MobileBertModel`）
- `bert` 到 `mobilebert´（例如，定义`self.mobilebert`）
- `BERT` 到 `MOBILEBERT`（在`MOBILEBERT_INPUTS_DOCSTRING`常量中）
