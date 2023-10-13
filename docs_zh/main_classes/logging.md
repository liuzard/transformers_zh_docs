<!--
版权所有2020年The HuggingFace团队。

根据Apache许可证2.0版（“许可证”），你除非符合许可证中的规定，否则不得使用此文件。
你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律或书面同意，根据许可证分发的软件以“原样基础”分发，
不附带任何明示或暗示的保证或条件。请查看许可证了解许可中的具体语言和限制。

⚠️请注意，此文件是使用Markdown编写的，但包含了我们doc-builder的特定语法（类似于MDX），
所以在你的Markdown查看器中可能无法正确显示。

-->

# 日志记录

🤗Transformers拥有一个集中式的日志记录系统，以便你可以轻松设置库的详细程度。

当前库的默认详细程度为`WARNING`。

要更改详细程度，请使用以下直接设置器之一。例如，以下是如何将详细程度更改为INFO级别。

```python
import transformers

transformers.logging.set_verbosity_info()
```

你还可以使用环境变量`TRANSFORMERS_VERBOSITY`来覆盖默认的详细程度。你可以将其设置为以下值之一：`debug`、`info`、`warning`、`error`、`critical`。例如：

```bash
TRANSFORMERS_VERBOSITY=error ./myprogram.py
```

另外，可以通过将环境变量`TRANSFORMERS_NO_ADVISORY_WARNINGS`设置为`true`值来禁用某些`warnings`，例如*1*。这将禁用使用[`logger.warning_advice`]记录的任何警告。例如：

```bash
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
```

下面是一个在你自己的模块或脚本中使用与库相同的记录器的示例：

```python
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
```


此日志记录模块的所有方法都在下面的文档中进行了说明，其中主要的方法为
[`logging.get_verbosity`]，用于获取记录器中当前的详细程度，
以及[`logging.set_verbosity`]，用于将详细程度设置为你选择的级别。
按顺序（从最不详细到最详细），这些级别（及其相应的整数值）为：

- `transformers.logging.CRITICAL`或`transformers.logging.FATAL`（整数值为50）：仅报告最重要的错误。
- `transformers.logging.ERROR`（整数值为40）：仅报告错误。
- `transformers.logging.WARNING`或`transformers.logging.WARN`（整数值为30）：仅报告错误和警告。这是库使用的默认级别。
- `transformers.logging.INFO`（整数值为20）：报告错误、警告和基本信息。
- `transformers.logging.DEBUG`（整数值为10）：报告所有信息。

默认情况下，在模型下载期间将显示`tqdm`进度条。[`logging.disable_progress_bar`]和[`logging.enable_progress_bar`]可用于取消显示或取消取消此行为。

## 基础设置器

[[autodoc]] logging.set_verbosity_error

[[autodoc]] logging.set_verbosity_warning

[[autodoc]] logging.set_verbosity_info

[[autodoc]] logging.set_verbosity_debug

## 其他函数

[[autodoc]] logging.get_verbosity

[[autodoc]] logging.set_verbosity

[[autodoc]] logging.get_logger

[[autodoc]] logging.enable_default_handler

[[autodoc]] logging.disable_default_handler

[[autodoc]] logging.enable_explicit_format

[[autodoc]] logging.reset_format

[[autodoc]] logging.enable_progress_bar

[[autodoc]] logging.disable_progress_bar
