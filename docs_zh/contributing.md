<!---
版权所有 2020 年 HuggingFace 团队。保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”）获得许可;
你不得除符合许可证的条款外使用此文件。
你可以通过以下方式获取许可证的副本

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律有要求或书面同意，否则以"按原样"方式分发软件，
无论是明示的或暗示的，也没有任何保证或条件。
请参阅许可证以了解特定语言的管理权限和
许可证下的限制。

-->
# 贡献到 🤗Transformers

欢迎所有人一起贡献，我们珍视每个人的贡献。代码
提供的帮助。

如果你不知道从哪里开始，有一个特殊的[Good First
问题](https://github.com/huggingface/transformers/contribute)清单。它会为你提供一个列表
开放问题，适合初学者，并帮助你开始贡献开源。只需在问题中发表评论，表示你想要在该问题上工作。

如果你愿意面临一些更有挑战性的事情，你也可以查看[Good Second Issue](https://github.com/huggingface/transformers/labels/Good%20Second%20Issue)列表。通常情况下，如果你认为你知道自己在做什么，那就去做吧，我们会帮助你取得成就！🚀

> 所有贡献对社区同等有价值。🥰

## 解决未解决的问题

如果你注意到现有代码的问题，并想法来解决，可以自由地[开始贡献](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md/#create-a-pull-request)并打开拉取请求！

## 提交与错误相关的问题或功能请求

在提交与错误相关的问题或功能请求时，请尽力遵循以下准则。这将使我们更容易迅速地给你提供反馈。

### 你发现了错误吗？

感谢使用者报告他们遇到的问题，🤗Transformers 库非常健壮可靠。

在报告问题之前，我们非常希望你 **确保未曾报告过该错误**
（在 GitHub 的 Issues 下使用搜索栏）。你的问题应与库本身的问题相关，并且不应与你的代码相关。如果你不确定错误是否在你的代码或库中，请先在[论坛](https://discuss.huggingface.co/)上询问。这样，我们将能够更快地回应与库相关的问题而不是一般问题。

确认错误尚未报告过后，请在你的问题中包含以下信息，以便我们能够快速解决它：

* 当适用时，请提供你的 **操作系统类型和版本**、**Python 版本**、**PyTorch 版本** 和 **TensorFlow 版本**。
* 提供一个简短的自包含的代码片段，以便我们能够在不到 30 秒的时间内重现错误。
* 如果发生了异常，请提供*完整的*回溯信息。
* 附加任何其他可能有助于解决问题的信息，比如截图。

为了自动获取操作系统和软件版本，请在 shell 中运行以下命令：

```bash
transformers-cli env
```

你也可以在仓库的根目录中运行相同的命令：

```bash
python src/transformers/commands/transformers_cli.py env
```

### 你想要新功能吗？

如果你希望在 🤗Transformers 中看到新功能，请打开一个问题并描述：

1. 此功能背后的**动机**是什么？它是否与库的问题或困扰有关？它是否与你为某个项目所需的功能有关？是否与你工作过的功能有关，你认为它可能对社区有所帮助？

   无论是什么，我们都很乐意听听你的想法！

2. 尽可能详细地描述你请求的功能。你提供的信息越详细，我们就能够提供更好的帮助。
3. 提供演示功能用法的 *代码片段*。
4. 如果该功能与一篇论文有关，请包含链接。

如果你写得足够好，我们在创建它时就已经完成了 80%。

我们添加了[模板](https://github.com/huggingface/transformers/tree/main/templates)来帮助你开始处理问题。

## 你想要实现一个新模型吗？

不断有新模型发布，如果你想要实现一个新模型，请提供以下信息

* 模型的简要描述和论文链接。
* 如果模型是开源的，请提供实现链接。
* 如果模型权重可用，请提供权重链接。

如果你愿意贡献模型，请告诉我们，好让我们能协助你将其添加到 🤗Transformers！

我们已经添加了一个[详细指南和模板](https://github.com/huggingface/transformers/tree/main/templates)，以帮助你开始添加新模型，我们还有一份更详细的指南，[如何将模型添加到🤗Transformers](https://huggingface.co/docs/transformers/add_new_model)。

## 你想要添加文档吗？

我们始终在寻求对文档进行改进，使其更加清晰和准确。请告诉我们文档如何改进，例如拼写错误和缺少的内容，以及不清楚或不准确的内容。如果你有兴趣，我们很乐意进行更改或协助你进行贡献！

有关如何生成、构建和编写文档的详细信息，请参阅文档[README](https://github.com/huggingface/transformers/tree/main/docs)。

## 创建拉取请求

在编写任何代码之前，我们强烈建议你搜索现有的 PR 或问题，
以确保没有其他人正在处理同样的事情。如果你不确定，最好是打开一个问题以获取一些反馈。

要贡献到
🤗Transformers，你需要基本的 `git` 熟练技能。
虽然 `git` 不是最容易使用的工具，但它有最好的
帮助手册。在 shell 中输入 `git --help` 并享受吧！如果你更喜欢阅读书籍，那么 [Pro
Git](https://git-scm.com/book/en/v2) 是一个非常好的参考。

你需要 **[Python 3.8]((https://github.com/huggingface/transformers/blob/main/setup.py#L426))** 或更高的版本来贡献到 🤗Transformers。请按照以下步骤开始贡献：

1. 通过单击存储库页面上的**[Fork](https://github.com/huggingface/transformers/fork)**按钮，对[存储库](https://github.com/huggingface/transformers)进行分支。这将在你的 GitHub 用户帐户下创建代码的副本。

2. 将分支克隆到本地磁盘，并将基本存储库添加为远程存储库：

   ```bash
   git clone git@github.com:<your Github handle>/transformers.git
   cd transformers
   git remote add upstream https://github.com/huggingface/transformers.git
   ```

3. 创建新分支以保存你的开发更改：

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **不要**使用 `main` 分支进行工作！

4. 在虚拟环境中运行以下命令，设置开发环境：

   ```bash
   pip install -e ".[dev]"
   ```

   如果🤗Transformers 已在虚拟环境中安装，请使用 `pip uninstall transformers` 将其删除，然后再使用 `-e` 标志重新安装。
   
   根据你的操作系统，并且由于 Transformers 的可选依赖项数量在不断增加，你可能会遇到此命令的失败。如果是这种情况，请确保你安装了你正在使用的深度学习框架（PyTorch、TensorFlow 和/或 Flax），然后执行以下操作：

   ```bash
   pip install -e ".[quality]"
   ```

   这对大多数用例应该足够。

5. 在你的分支上开发功能。

   在编写代码时，你应确保测试套件
   通过。运行受你更改影响的测试，例如：

   ```bash
   pytest tests/<TEST_TO_RUN>.py
   ```

   有关测试的更多信息，请查看
   [Testing](https://huggingface.co/docs/transformers/testing) 指南。

   🤗Transformers 依赖于 `black` 和 `ruff` 来一致地格式化其源代码。
   在进行更改后，应用自动样式更正和代码验证
   无法以一次性自动完成的：

   ```bash
   make fixup
   ```

   该目标还经过优化，只与你正在处理的 PR 所修改的文件一起工作。

   如果你愿意依次运行各个检查，请使用以下命令应用
   样式更正：

   ```bash
   make style
   ```

   🤗Transformers 还使用 `ruff` 和一些自定义脚本来检查代码错误。
   质量控制是由 CI 运行的，但是你可以使用相同的检查运行：

   ```bash
   make quality
   ```

   最后，我们有很多脚本，用于确保在添加新模型时不会忘记更新某些文件。你可以使用以下命令运行这些脚本：

   ```bash
   make repo-consistency
   ```

   要了解有关这些检查的详细信息以及如何解决其中的任何问题，请查看
   [Checks on a Pull Request](https://huggingface.co/docs/transformers/pr_checks) 指南。

   如果你正在修改 `docs/source` 目录下的文档，请确保文档仍然可以构建。当你打开拉取请求时，CI 也将运行此检查。要运行本地检查，请确保安装了文档生成器：

   ```bash
   pip install ".[docs]"
   ```

   从仓库的根目录运行以下命令：

   ```bash
   doc-builder build transformers docs/source/en --build_dir ~/tmp/test-build
   ```

   这将在 `~/tmp/test-build` 文件夹中构建文档，你可以使用喜欢的编辑器查看生成的 Markdown 文件。你还可以在打开拉取请求时在 GitHub 上预览文档。

   一旦你对更改感到满意，请使用 `git add` 添加已更改的文件，并在本地记录更改：

   ```bash
   git add modified_file.py
   git commit
   ```

   请记住编写[良好的提交
   消息](https://chris.beams.io/posts/git-commit/)以清楚表达你所做的更改！

   在将你的代码副本与原始存储库保持同步之前，请基于 `upstream/branch` 重派基于你的分支属于该分支的情况下的拉取请求。或者，在维护者的要求下切换分支：

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   将你的更改推送到你的分支：

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   如果你已经打开了拉取请求，则需要使用 `--force` 标志强制推送。否则，如果尚未打开拉取请求，则可以像平常一样推送你的更改。

6. 现在，你可以前往你在 GitHub 上的存储库分支，并单击**拉取请求**以打开拉取请求。请确保你在下面的[清单](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md/#pull-request-checklist)中选中所有框。准备好后，你可以将更改发送给项目维护者进行审查。

7. 如果维护者请求更改，没关系，我们的核心贡献者也会这样！为了让所有人能够看到拉取请求中的更改，在本地分支上工作并将更改推送到你的存储库。它们将自动出现在
利用 [Changes](https://github.com/huggingface/transformers/blob/main/templates/changes.md) 和相关文件。

### 拉取请求清单

☐ 拉取请求标题应概括你的贡献。<br>
☐ 如果你的拉取请求与问题有关，请在拉取请求说明中提及问题编号，以确保它们链接在一起（查看问题的人将知道你正在处理此问题）。<br>
☐ 要指示进展中的工作，请在标题前缀中添加 `[WIP]`。这些很有用，以避免重复工作，并将其与准备合并的 PR 区分开。<br>
☐ 确保现有测试通过。<br>
☐ 如果添加新功能，请为其添加测试。<br>
   - 如果你要添加新模型，请确保使用
     `ModelTester.all_model_classes = (MyModel, MyModelWithLMHead,...)` 来触发常见测试。
   - 如果你添加了新的 `@slow` 测试，请确保使用
     `RUN_SLOW=1 python -m pytest tests/models/my_new_model/test_my_new_model.py` 运行以进行运行。
   - 如果你要添加新的分词器，请编写测试，并确保
     `RUN_SLOW=1 python -m pytest tests/models/{your_model_name}/test_tokenization_{your_model_name}.py` 通过。
   - CircleCI 不会运行慢速测试，但是 GitHub Actions 会每天运行！<br>

☐ 所有公共方法必须具有有关的文档字符串（参见
[`modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
的示例）。<br>
☐ 由于库迅速增长，请不要添加重量级的非文本文件，如图像、视频和其他非文本文件。而是使用 Hub
仓库，例如 [`hf-internal-testing`](https://huggingface.co/hf-internal-testing)
来托管这些文件，并通过 URL 引用它们。我们建议将与文档相关的图像放在以下仓库中：
[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)。
你可以在该数据集仓库上打开 PR，并请 Hugging Face 成员合并它。

有关在拉取请求上运行的检查的更多信息，请查看我们的[在拉取请求上运行的检查](https://huggingface.co/docs/transformers/pr_checks) 指南。

### 测试

为了测试库的行为和一些示例，包含了一个广泛的测试套件。库测试位于
[tests](https://github.com/huggingface/transformers/tree/main/tests) 文件夹中，示例测试位于
[examples](https://github.com/huggingface/transformers/tree/main/examples) 文件夹中。

我们喜欢使用 `pytest` 和 `pytest-xdist`，因为速度更快。从存储库的根目录，指定要运行的测试的*子文件夹或测试文件*。

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
```

同样，对于 `examples` 目录，请指定要运行的测试的*子文件夹或测试文件*。例如，以下命令测试 PyTorch `examples` 目录中的文本分类子文件夹:

```bash
pip install -r examples/xxx/requirements.txt  # 只在第一次需要
python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

事实上，我们的 `make test` 和 `make test-examples` 命令就是这样实现的（不包括 `pip install`）！

你也可以指定一个较小的测试集，以仅测试你正在开发的功能。

默认情况下，会跳过慢速测试，但是你可以将 `RUN_SLOW` 环境变量设置为 `yes` 以运行它们。这将会下载很多 GB 的模型数据，所以请确保你拥有足够的磁盘空间、良好的网络连接或者足够的耐心！

<Tip warning={true}>

记得要指定 *要运行测试的子文件夹或测试文件的路径*。否则，你将会运行 `tests` 或 `examples` 文件夹中的所有测试，这将需要很长时间！

</Tip>

```bash
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

和慢速测试一样，还有其他一些在测试过程中默认未启用的环境变量可用：
- `RUN_CUSTOM_TOKENIZERS`：启用自定义分词器的测试。
- `RUN_PT_FLAX_CROSS_TESTS`：启用 PyTorch + Flax 整合的测试。
- `RUN_PT_TF_CROSS_TESTS`：启用 TensorFlow + PyTorch 整合的测试。

可以在 [testing_utils.py](src/transformers/testing_utils.py) 中找到更多环境变量和详细信息。

🤗Transformers 仅使用 `pytest` 作为测试运行器，不会在测试套件本身中使用任何 `pytest` 特定的功能。

这意味着 `unittest` 得到了完全支持。以下是如何使用 `unittest` 运行测试的方法：

```bash
python -m unittest discover -s tests -t . -v
python -m unittest discover -s examples -t examples -v
```

### 风格指南

对于文档字符串，🤗Transformers 遵循 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html)。
请查看我们的[文档编写指南](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification)以获取更多信息。

### 在 Windows 上开发

在 Windows 上（除非你在 [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) 或 WSL 中工作），你需要配置 git 将 Windows 的 `CRLF` 行尾转换为 Linux 的 `LF` 行尾：

```bash
git config core.autocrlf input
```

在 Windows 上运行 `make` 命令的一种方法是使用 MSYS2：

1. [下载 MSYS2](https://www.msys2.org/)，并假设已安装在 `C:\msys64`。
2. 打开命令行 `C:\msys64\msys2.exe`（可以从 **开始** 菜单中找到）。
3. 在 shell 中运行：`pacman -Syu`，然后使用 `pacman -S make` 安装 `make`。
4. 将 `C:\msys64\usr\bin` 添加到你的 PATH 环境变量中。

现在你可以在任何终端（如 Powershell、cmd.exe 等）中使用 `make` 命令了！🎉

### 将 fork 的仓库与上游主仓库（Hugging Face 仓库）进行同步

在更新 fork 的仓库的主分支时，请按照以下步骤操作，以避免向上游仓库发送通知（它会为每个上游 PR 添加参考注释）并向涉及这些 PR 的开发人员发送不必要的通知。

1. 在可能的情况下，避免使用分支和 PR 在 fork 的仓库中与上游仓库同步。而是直接合并到 fork 的主分支中。
2. 如果确实需要 PR，在检出你的分支后，请按照以下步骤操作：

```bash
git checkout -b your-branch-for-syncing
git pull --squash --no-commit upstream main
git commit -m '<your message without GitHub references>'
git push --set-upstream origin your-branch-for-syncing
```
