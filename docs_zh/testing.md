<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；你除非符合许可证，否则不得使用此文件。你可以在以下位置获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”提供的，不附带任何明示或默示的保证或条件。请参阅许可证以获取特定语言的权限和限制。

⚠️请注意，此文件采用Markdown格式，但包含特定用于我们doc-builder（类似于MDX）的语法，可能在你的Markdown查看器中无法正确显示。

-->

# 测试


让我们来看看🤗Transformers模型如何进行测试，以及如何编写新的测试和改进现有测试。

存储库中有两个测试套件：

1. `tests` -- 用于通用API的测试
2. `examples` -- 主要用于各种不属于API的应用的测试

## 如何测试transformers

1. 一旦提交PR，它就会在9个CircleCi作业中进行测试。对该PR进行的每次新提交都会重新进行测试。这些作业
   在这个 [config文件]中定义（https://github.com/huggingface/transformers/tree/main/.circleci/config.yml），所以如果需要，你可以在你的机器上重现相同的
   环境。

   这些CI作业不会运行“@slow”测试。

2. 有3个由 [github actions]（https://github.com/huggingface/transformers/actions）运行的任务：

   - [torch hub集成]（https://github.com/huggingface/transformers/tree/main/.github/workflows/github-torch-hub.yml）: 检查torch hub
     集成是否正常工作。

   - [自己托管（推送）]（https://github.com/huggingface/transformers/tree/main/.github/workflows/self-push.yml）：仅在`main`上进行GPU上的快速测试
     提交。只有在`main`的提交中更新了以下文件夹中的代码时，它才会运行：`src`，
     `tests`，`.github`（以防止在添加的模型卡片、笔记本等上运行）。

   - [自己托管运行器]（https://github.com/huggingface/transformers/tree/main/.github/workflows/self-scheduled.yml）：在GPU上运行普通和慢速测试
     `tests`和`examples`中：

```bash
RUN_SLOW=1 pytest tests/
RUN_SLOW=1 pytest examples/
```

   结果可在[此处]观察到（https://github.com/huggingface/transformers/actions）。



## 运行测试





### 选择要运行的测试

本文档详细介绍了如何运行测试的许多细节。如果阅读完所有内容后，你仍需要更多详细信息
你将在这里找到它们[这里]（https://docs.pytest.org/en/latest/usage.html）。

下面是一些运行测试的最有用方法。

运行全部：

```console
pytest
```

或：

```bash
make test
```

需要注意的是，后者定义为：

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

它告诉pytest：

- 运行与CPU内核数相同的测试进程（如果RAM不多，则可能过多！）
- 确保由同一测试进程运行来自同一文件的所有测试
- 不使用捕获输出
- 以详细模式运行



### 获取所有测试的列表

测试套件的所有测试：

```bash
pytest --collect-only -q
```

给定测试文件的所有测试：

```bash
pytest tests/test_optimization.py --collect-only -q
```

### 运行特定的测试模块

要运行单个测试模块：

```bash
pytest tests/utils/test_logging.py
```

### 运行特定的测试

由于大多数测试中使用了unittest，要运行特定的子测试，你需要知道包含这些测试的unittest类的名称。例如，它可能是：

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

其中：

- `tests/test_optimization.py` - 包含测试的文件
- `OptimizationTest` - 类的名称
- `test_adam_w` - 特定测试函数的名称

如果文件包含多个类，可以选择运行给定类的所有测试。例如：

```bash
pytest tests/test_optimization.py::OptimizationTest
```

将运行该类中的所有测试。

正如前面提到的，你可以通过运行以下命令查看`OptimizationTest`类中包含的所有测试：

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

你可以通过关键字表达式来运行测试。

要仅运行名称中包含“adam”的测试：

```bash
pytest -k adam tests/test_optimization.py
```

可以使用逻辑`and`和`or`来指示是否应匹配所有关键字或任何一个。可以使用`not`来进行否定。

要运行除了名称中包含“adam”的所有测试：

```bash
pytest -k "not adam" tests/test_optimization.py
```

你可以在一个模式中组合两个模式：

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

例如，要同时运行`test_adafactor`和`test_adam_w`，可以使用：

```bash
pytest -k "test_adam_w or test_adam_w" tests/test_optimization.py
```

请注意，这里我们使用`or`，因为我们希望任一关键字都匹配以包含两者。

如果只想包含同时包含两个模式的测试，应使用`and`：

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### 运行“accelerate”测试

有时你需要在模型上运行“accelerate”测试。为此，只需将`-m accelerate_tests`添加到你的命令中，例如，如果你要在`OPT`上运行这些测试，则运行：
```bash
RUN_SLOW=1 pytest -m accelerate_tests tests/models/opt/test_modeling_opt.py 
```


### 运行文档测试 

为了测试文档示例是否正确，你应该检查`doctests`是否通过。 
例如，让我们使用[`WhisperModel.forward`的docstring](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py#L1017-L1035)： 

```python 
r"""
Returns:

Example:
    ```python
    >>> import torch
    >>> from transformers import WhisperModel, WhisperFeatureExtractor
    >>> from datasets import load_dataset

    >>> model = WhisperModel.from_pretrained("openai/whisper-base")
    >>> feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features
    >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 2, 512]
    ```"""

```

只需运行以下命令即可自动测试所需文件中的每个docstring示例： 
```bash 
pytest --doctest-modules <path_to_file_or_dir>
```
如果文件扩展名为markdown，则应添加`--doctest-glob="*.md"`参数。

### 仅运行已修改的测试

你可以通过使用 [pytest-picked](https://github.com/anapaulagomes/pytest-picked) 来运行与未暂存文件或当前分支（根据 Git）相关的测试。这是一种在快速测试你的更改没有破坏时的快速测试的好方法。
任何东西，因为它不会运行与你没有触及的文件相关的测试。

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

将从未提交但已修改的文件和文件夹运行所有测试。

### 在源代码修改时自动重试失败的测试

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist)提供了一项非常有用的功能，可以检测到所有失败
测试，然后等待你修改文件并在修复期间连续重新运行那些失败的测试，直到它们通过为止。这样就不需要在修复后重新启动pytest了。直到所有测试通过后再执行完整的运行。

```bash
pip install pytest-xdist
```

进入模式：`pytest -f`或`pytest --looponfail`

通过查看`looponfailroots`根目录及其所有内容（递归地）来检测文件更改。如果默认值对你不起作用，可以在`setup.cfg`中设置配置选项来更改项目设置：

```ini
[tool:pytest]
looponfailroots = transformers tests
```

或`pytest.ini`/``tox.ini``文件：

```ini
[pytest]
looponfailroots = transformers tests
```

这将只在相应的目录中查找文件更改，相对于ini文件的目录而指定。

[pytest-watch](https://github.com/joeyespo/pytest-watch) 是这个功能的另一种实现方式.


### 跳过测试模块

如果要运行所有测试模块，除了一些例外，你可以通过指定要运行的测试的显式列表来排除它们。例如，要运行除了`test_modeling_*.py`测试之外的全部测试：

```bash
pytest *ls -1 tests/*py | grep -v test_modeling*
```

### 清除状态

应清除CI构建和需要隔离性（针对速度）的缓存：

```bash
pytest --cache-clear tests
```

### 并行运行测试

如前所述，`make test`通过`pytest-xdist`插件（`-n X`参数，例如`-n 2`
以运行2个并行作业）并行运行测试。

`pytest-xdist`的`--dist=`选项允许控制如何对测试进行分组。`--dist=loadfile`将
位于一个文件中的测试放入同一进程中。

由于执行的测试的顺序是不同且不可预测的，如果使用`pytest-xdist`运行测试套件产生失败（意味着我们有一些未检测到的耦合测试），可以使用 [pytest-replay](https://github.com/ESSS/pytest-replay) 在相同的顺序中重播测试，这样能帮助减少那种失败序列的数量。

### 测试顺序和重复

最好多次重复运行测试，按顺序、随机或集合，以检测潜在的，与状态相关的错误（清除状态）。直接多次重复测试只是用来检测一些由DL随机性暴露出来的问题。

#### 重复测试

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder)：

```bash
pip install pytest-flakefinder
```

然后运行每个测试多次（默认为50次）：

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

<Tip>

此插件与`pytest-xdist`的`-n`标志不兼容。

</Tip>

<Tip>

还有另一个插件`pytest-repeat`，但它与`unittest`不兼容。

</Tip>

#### 以随机顺序运行测试

```bash
pip install pytest-random-order
```

重要：只要存在`pytest-random-order`，测试将自动随机化，无需更改配置或命令行选项。

正如前面解释的，这可以检测耦合测试——其中一个测试的状态会影响另一个测试的状态。当
安装`pytest-random-order`时，它会打印它用于该会话的随机种子，例如：

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

因此，如果给定特定序列失败，你可以通过添加完全相同的种子来重现它，例如：

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

仅当你使用完全相同的测试列表（或没有测试列表）时，它才会重现确切的顺序。一旦你开始手动缩小列表，你就不能再依赖种子，而必须手动以失败的确切顺序列出它们，并告诉pytest不要再对它们进行随机排序，使用`--random-order-bucket=none`，例如：

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

要为所有测试禁用随机排序：

```bash
pytest --random-order-bucket=none
```

默认情况下，隐含`--random-order-bucket=module`，这将在模块级别上对文件进行随机排序。它也可以
按照`class`，`package`，`global`和`none`级别进行洗牌。有关完整详细信息，请参阅其
[文档]（https://github.com/jbasko/pytest-random-order）。

另一个随机化的替代方案是：[`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly)。这个
模块具有非常相似的功能/接口，但它没有 `pytest-random-order` 中可用的 桶模式。它具有与 `pytest-random-order` 相同的问题，一经安装即会强制实施。

### 查看和感觉变化

#### pytest-sugar

[pytest-sugar](https://github.com/Frozenball/pytest-sugar)是一个改善外观和感觉、添加进度条并立即显示失败测试和assert的插件。安装后会自动激活。

```bash
pip install pytest-sugar
```

要在没有它的情况下运行测试，请运行：

```bash
pytest -p no:sugar
```

或卸载它。



#### 显示每个子测试名称及其进度

通过`pytest`为单个或一组测试（在`pip install pytest-pspec`之后）：

```bash
pytest --pspec tests/test_optimization.py
```

#### 立即显示失败的测试

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail)会立即显示失败和错误，而不是等到测试会话结束。

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### GPU还是非GPU

在启用GPU的设置上，要以仅CPU模式进行测试，请添加`CUDA_VISIBLE_DEVICES=""`：

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

或者如果你有多个GPU，请指定要由`pytest`使用的GPU。例如，如果你有GPU `0` 和 `1`，则只使用第二个GPU：

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

当你希望在不同的GPU上运行不同的任务时，这很方便。

某些测试必须在仅CPU上运行，其他测试必须在CPU或GPU或TPU上运行，而其他测试必须在多个GPU上运行。以下跳过
装饰器用于根据CPU/GPU/TPU需求设置测试：


- `require_torch` - 该测试仅在torch下运行
- `require_torch_gpu` - 使用`require_torch`，还需要至少1个GPU
- `require_torch_multi_gpu` - 使用`require_torch`，还需要至少2个GPU
- `require_torch_non_multi_gpu` - 使用`require_torch`，还需要0个或1个GPU
- `require_torch_up_to_2_gpus` - 使用`require_torch`，还需要0个或1个或2个GPU
- `require_torch_tpu` - 使用`require_torch`，还需要至少1个TPU

让我们在下面看一个测试，在有2个或更多个GPU可用且已安装pytorch的情况下才能运行：

```python no-style
@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

如果一个测试需要`tensorflow`，请使用`require_tf`装饰器。例如：

```python no-style
@require_tf
def test_tf_thing_with_tensorflow():
```

这些装饰符可以堆叠使用。例如，如果一个测试较慢并且需要至少一个pytorch下的GPU，下面是如何设置：

```python no-style
@require_torch_gpu
@slow
def test_example_slow_on_gpu():
```

有些装饰符（如`@parametrized`）会重写测试的名称，因此`@require_*`跳过装饰符必须在它们之后列出，以便正确工作。下面是正确使用的示例：

```python no-style
@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

这个顺序问题在`@pytest.mark.parametrize`中是不存在的，你可以把它放在首位或末尾都可以，它仍然可以正常工作。但是它仅适用于非单元测试代码。

测试中：

- 可用的GPU数量是多少：

```python
from transformers.testing_utils import get_gpu_count

n_gpu = get_gpu_count()  # works with torch and tf
```

### 使用特定的PyTorch后端或设备进行测试

要在特定的torch设备上运行测试套件，请添加`TRANSFORMERS_TEST_DEVICE="$device"`，其中`$device`是目标后端。例如，要仅在CPU上运行测试：
```bash
TRANSFORMERS_TEST_DEVICE="cpu" pytest tests/utils/test_logging.py
```

该变量对于测试自定义或不常见的PyTorch后端（如`mps`）非常有用。它还可以用于通过定位特定的GPU或在仅CPU模式下进行测试来实现与`CUDA_VISIBLE_DEVICES`相同的效果。

在导入`torch`后第一次使用后，某些设备将需要额外的导入。这可以使用环境变量`TRANSFORMERS_TEST_BACKEND`进行指定：
```bash
TRANSFORMERS_TEST_BACKEND="torch_npu" pytest tests/utils/test_logging.py
```


### 分布式训练

`pytest`无法直接处理分布式训练。如果尝试这样做，子进程不会正确地执行测试套件，而是认为它们是`pytest`，开始在循环中运行测试。但如果生成一个正常的进程，然后从中生成多个工作进程并管理IOpipeline，它会正常工作。

这是一些使用它的测试的例子：

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/tree/main/tests/trainer/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/tree/main/tests/deepspeed/test_deepspeed.py)

要直接跳到执行点，搜索这些测试中的`execute_subprocess_async`调用。

你将至少需要2个GPU才能看到这些测试的效果：

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

### 输出捕获

在测试执行期间，任何发送到`stdout`和`stderr`的输出都会被捕获。如果一个测试或设置方法失败，其相应的捕获输出通常会与失败的回溯一起显示。

要禁用输出捕获并正常获取`stdout`和`stderr`，使用`-s`或`--capture=no`：

```bash
pytest -s tests/utils/test_logging.py
```

将测试结果发送到Junit格式的输出：

```bash
py.test tests --junitxml=result.xml
```

### 颜色控制

如果不想要颜色（例如，白色背景上的黄色不可读）：

```bash
pytest --color=no tests/utils/test_logging.py
```

### 将测试报告发送到在线粘贴服务

为每个测试失败创建一个URL：

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

这将将测试运行信息提交到远程粘贴服务，并为每个失败提供一个URL。你可以像往常一样选择测试，或者添加`-x`（如果只想发送一个特定的失败）。

为整个测试会话日志创建一个URL：

```bash
pytest --pastebin=all tests/utils/test_logging.py
```

## 编写测试

🤗transformers 测试基于`unittest`，但由`pytest`运行，因此大多数情况下可以使用这两个系统的功能。

你可以在[这里](https://docs.pytest.org/en/stable/unittest.html)阅读其支持的功能，但重要的是要记住，大多数`pytest`修饰器不起作用。参数化也不起作用，但是我们使用`parameterized`模块，它的工作方式类似。

### 参数化

通常，需要使用不同的参数多次运行相同的测试。可以在测试内部完成，但是这样无法仅对一组参数运行测试。

```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ]
    )
    def test_floor(self, name, input, expected):
        assert_equal(math.floor(input), expected)
```

默认情况下，此测试将运行3次，每次将 `test_floor` 的最后3个参数分配给参数列表中对应的参数。

并且可以使用以下命令仅运行 `negative` 和 `integer` 子测试：

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

或使用以下命令运行除了 `negative` 子测试的所有子测试：

```bash
pytest -k "not negative" tests/test_mytest.py
```

除了使用 `-k` 过滤器之外，还可以通过使用名称来找出每个子测试，并根据需要运行任何一个或所有子测试。

```bash
pytest test_this1.py --collect-only -q
```

它会列出：

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

所以现在可以仅运行2个特定的子测试：

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

[parameterized](https://pypi.org/project/parameterized/)模块已经在`transformers`的开发者依赖中，可以在`unittest`和`pytest`测试中使用它。

不过，如果测试不是`unittest`，可以使用`pytest.mark.parametrize`（或者你可能会在一些现有的测试中看到）。

以下是相同的示例，这次使用`pytest`的`parametrize`修饰符：

```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
    "name, input, expected",
    [
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ],
)
def test_floor(name, input, expected):
    assert_equal(math.floor(input), expected)
```

与`parameterized`一样，`pytest.mark.parametrize`可以对要运行的子测试进行精细的控制，如果`-k`过滤器无法满足需求。除此之外，这种参数化函数会为子测试创建稍微不同的名称集。以下是它们的形式：

```bash
pytest test_this2.py --collect-only -q
```

它将列出：

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

因此，现在可以仅运行特定的测试：

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

与先前的示例一样。


### 文件和目录

在测试中，通常需要相对于当前测试文件的位置，这并不容易，因为测试可以从多个目录中调用，或者可能位于具有不同深度的子目录中。辅助类`transformers.test_utils.TestCasePlus`通过整理所有基本路径并提供易于访问的方式来解决此问题：

- `pathlib`对象（全部已解析）：

  - `test_file_path` - 当前测试文件的路径，即 `__file__`
  - `test_file_dir` - 包含当前测试文件的目录
  - `tests_dir` - `tests` 测试套件的目录
  - `examples_dir` - `examples` 测试套件的目录
  - `repo_root_dir` - 仓库的根目录
  - `src_dir` - `src`的目录（即 `transformers` 子目录所在的位置）

- 字符串路径：与上面相同，但这些路径以字符串形式返回，而不是 `pathlib` 对象：

  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`

只需确保测试位于`transformers.test_utils.TestCasePlus`的子类中，即可开始使用它们。例如：

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

如果不需要使用 `pathlib` 操作路径，或者只需要将路径作为字符串，可以在 `pathlib` 对象上调用 `str()` 或使用以 `_str` 结尾的访问器。例如：

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

### 临时文件和目录

对于并行运行的测试，使用唯一的临时文件和目录非常重要，以便测试不会相互覆盖彼此的数据。此外，我们希望在每个创建它们的测试结束时删除临时文件和目录。因此，使用像 `tempfile` 这样的包以满足这些需求是至关重要的。

但是，在调试测试时，你需要能够查看临时文件或目录中的内容，并且希望知道其确切路径，而不是在每次测试重新运行时进行随机化。

辅助类`transformers.test_utils.TestCasePlus`最适用于此类目的。它是`unittest.TestCase`的子类，因此我们可以轻松地从测试模块继承它。

下面是其用法示例：

```python
from transformers.testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

此代码将创建一个唯一的临时目录，并将 `tmp_dir` 设置为其位置。

- 创建唯一的临时目录：

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir` 将包含创建的临时目录的路径。它将在测试结束时自动删除。

- 创建我选择的临时目录，在测试开始之前确保为空，并在测试结束后不清空。

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

这在调试时非常有用，当你想要监视特定目录并确保以前的测试没有在那里留下任何数据时。

- 你可以通过直接覆盖 `before` 和 `after` 参数来覆盖默认行为，从而得到以下行为之一：

  - `before=True`：临时目录将始终在测试开始时被清空。
  - `before=False`：如果临时目录已经存在，则任何现有文件将保留在其中。
  - `after=True`：临时目录始终会在测试结束时被删除。
  - `after=False`：临时目录将始终保留在测试结束时。

<Tip>

为了安全地运行相当于`rm -r`，只允许在项目仓库检出子目录下注册临时目录，以免错误地删除`/tmp`或类似的重要文件系统部分。即请始终传递以 `./` 开头的路径。

</Tip>

<Tip>

每个测试可以注册多个临时目录，除非另有要求，它们都将被自动删除。

</Tip>

### 临时 sys.path 改写

如果需要临时改写`sys.path`以从另一个测试中进行导入，可以使用`ExtendSysPath`上下文管理器。示例：

```python
import os
from transformers.testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### 跳过测试

当发现错误并编写了新的测试但尚未修复错误时，跳过测试是很有用的。为了能够将其提交到主存储库，我们需要确保在`make test`期间跳过它。

方法：

- **跳过**意味着只有满足某些条件时才期望测试通过，否则`pytest`应该完全跳过运行测试。常见的例子是仅在非Windows平台上跳过仅适用于Windows的测试，或者跳过依赖于目前不可用的外部资源（例如数据库）的测试。

- **xfail**意味着你预计测试会失败出现某些问题。常见的例子是尚未实现的功能或尚未修复的错误的测试。当标记为`pytest.mark.xfail`的测试尽管预期失败但实际上通过时，它们将被报告为`xpass`。

两者之间的一个重要区别是`skip`不会运行测试，而`xfail`会。因此，如果引起错误的代码会导致一些会影响其他测试的坏状态，请不要使用`xfail`。

#### 实现

- 以下是如何无条件跳过整个测试：

```python no-style
@unittest.skip("this bug needs to be fixed")
def test_feature_x():
```

或通过`pytest`：

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

或通过`xfail`方式：

```python no-style
@pytest.mark.xfail
def test_feature_x():
```

- 根据测试内部的一些内部检查跳过测试：

```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

或整个模块的跳过：

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

或`xfail`方式：

```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- 如果缺少某个导入，则跳过模块中的所有测试：

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

- 根据条件跳过测试：

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

或：

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

或跳过整个模块：

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

更多详情、示例和方法在[这里](https://docs.pytest.org/en/latest/skipping.html)。

一些装饰器，如`@parameterized`，会重写测试名称，因此必须最后列出`@slow`和其他跳过装饰器`@require_*`，以便它们能够正常工作。以下是正确使用的示例：

```python no-style
@parameteriz ed.expand(...)
@slow
def test_integration_foo():
```

正如本文档开头所解释的，慢速测试会定期运行，而不是在Pull Request（PR）的CI检查中运行。因此，在提交PR之前在你的计算机上运行慢速测试非常重要，以确保不会漏掉任何问题。

以下是选择标记为慢速测试的大致决策机制：

- 如果测试侧重于库的一个内部组件（例如，建模文件、分词文件、流水线），则应该在非慢速测试套件中运行该测试。如果侧重于库的其他方面，例如文档或示例，则应该在慢速测试套件中运行这些测试。然后，我们还可以有一些例外情况：
- 所有需要下载大量权重或大于~50MB的数据集（例如，模型或分词器集成测试，流水线集成测试）的测试都应该设置为慢速测试。如果要添加新模型，你应该创建并上传到hub的模型的微小版本（带有随机权重）用于集成测试。在接下来的几段中将对此进行讨论。
- 所有需要进行特定优化以提高速度的训练的测试都应该被设置为慢速测试。
- 如果其中一些本应为非慢速测试的测试运行非常慢，则可以纳入例外情况，并将它们设置为`@slow`。自动建模测试会将大型文件保存到磁盘并加载，它们是标记为`@slow`的测试的良好示例。
- 如果测试在CI上完成时间小于1秒（包括下载时间），则应将其视为正常测试。

总体而言，所有非慢速测试需要完全涵盖不同的内部组成部分，同时保持快速。例如，我们可以通过使用具有随机权重的特殊创建的微小模型进行测试来实现重要的覆盖范围。此类模型仅具有最小数量的层（例如2层）、词汇量（例如1000）等。然后，`@slow`测试可以使用大型慢速模型进行定性测试。要查看对应的使用情况，只需搜索带有“tiny”的*小型*模型：

```bash
grep tiny tests examples
```

下面是一个创建小型模型的[脚本示例](https://github.com/huggingface/transformers/tree/main/scripts/fsmt/fsmt-make-tiny-model.py)，它创建了名为[stas/tiny-wmt19-en-de](https://huggingface.co/stas/tiny-wmt19-en-de)的小型模型。你可以根据自己的具体模型架构轻松调整该脚本。

如果执行过程中出现了下载巨大模型的性能问题，正确测量运行时间可能会变得困难。但是，如果你在本地运行测试，下载的文件将被缓存，因此不会计算下载时间。因此，在CI日志中检查执行速度报告是非常重要的（使用`pytest --durations=0 tests`命令的输出）。

该报告还有助于查找未标记为慢速测试或需要重写为快速测试的慢速异常值。如果你注意到CI上的测试套件开始变慢，则该报告的顶部列表将显示最慢的测试项。