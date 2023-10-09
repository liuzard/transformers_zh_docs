<!--版权所有2020年HuggingFace团队保留。

根据Apache许可证，版本2.0 (the "License"); only可以在符合许可证条件的情况下使用
这个文件。你可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

软件在许可证规定的情况下分发
本来是基于“按现状”分布的 BASIS，没有任何明示或暗示的保证或条件。有关这个
表示或暗示，包括但不限于，适销性、针对特定目的适用性和不侵犯知识产权的
保证和条件。-->

# 如何将模型添加到🤗 Transformers？

通过社区贡献者的帮助，**🤗 Transformers**库通常能够提供新的模型。但这可能是一个具有挑战性的项目，并且需要对**🤗 Transformers**库和要实现的模型有深入的了解。在Hugging Face，我们正在努力赋予更多的社区成员能力来积极地添加模型，因此我们撰写了本指南，以引导您完成添加PyTorch模型的过程（请确保已经[安装了PyTorch](https://pytorch.org/get-started/locally/)）。

<Tips>

如果您有意于实现一个TensorFlow模型，请参考[如何将🤗 Transformers模型转换为TensorFlow](add_tensorflow_model.md)指南！

</Tips>

在这个过程中，您将会：

- 提供开源最佳实践的见解
- 理解[最受欢迎的深度学习库](https://github.com/huggingface/transformers/blob/master/README.md#philosophy)的设计原理
- 学会如何高效测试大模型
- 学会如何集成诸如`black`，`ruff`和`make fix-copies`等Python工具，以确保代码的规范和可读性

Hugging Face团队的成员将随时为您提供帮助，您永远不会孤单。🤗 ❤️

要开始，请在🤗 Transformers中打开一个[New model addition](https://github.com/huggingface/transformers/issues/new?assignees=&labels=New+model&template=new-model-addition.yml)问题，以添加您要在🤗 Transformers中看到的模型。如果您对贡献特定模型不太感兴趣，您可以按照[New model label](https://github.com/huggingface/transformers/labels/New%20model)进行筛选，看看是否有未认领的模型请求并开始工作。

打开一个新的模型请求后，第一步是熟悉🤗 Transformers，如果您还没有熟悉。

## 🤗 Transformers的概述

首先，您应该对🤗 Transformers有一个概览。🤗 Transformers是一个非常明确的库，因此您可能不同意其中的一些哲学观点或设计选择。然而，从我们的经验来看，库的基本设计选择和哲学原则对于高效扩展🤗 Transformers并保持合理的维护成本是至关重要的。

更好地理解这个库的一个好的起点是阅读我们的[哲学文档](https://github.com/huggingface/transformers/blob/master/README.md#philosophy)。作为我们工作方式的结果，有一些选择我们尝试应用到所有模型上：

- 通常倾向于组合而不是抽象
- 重复代码并不总是坏事，如果它能够极大地提高模型的可读性或可访问性
- 尽量使模型文件尽可能自包含，这样当您阅读特定模型的代码时，您理想情况下只需要查看相应的`modeling_....py`文件即可

在我们看来，库的代码不仅仅是提供一个产品的手段，例如使用BERT进行推理的能力，而且还是我们想改进的真正产品。因此，当添加一个模型时，用户不仅仅是指将使用您的模型的人，还是将阅读、尝试理解和可能调整您的代码的每个人。

记住这一点，让我们深入了解一下库的常规设计。

### 模型概述

要成功添加一个模型，重要的是要理解您的模型与其配置、[`PreTrainedModel`]和[`PretrainedConfig`]之间的交互。为了示例目的，我们将将要添加到🤗 Transformers的模型称为`BrandNewBert`。

让我们来看一下：

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_overview.png"/>

正如您所看到的，我们在🤗 Transformers中确实使用了继承，但我们将抽象级别保持在绝对最低限度。库中任何模型都没有超过两个层次的抽象。`BrandNewBertModel`继承自`BrandNewBertPreTrainedModel`，后者再继承自[`PreTrainedModel`]，而这就是全部。作为一项一般性原则，我们希望确保新模型仅依赖于[`PreTrainedModel`]。自动提供给每个新模型的重要功能包括[`~PreTrainedModel.from_pretrained`]和[`~PreTrainedModel.save_pretrained`]，用于序列化和反序列化。其他重要功能，例如`BrandNewBertModel.forward`，应该在新的`modeling_brand_new_bert.py`脚本中完全定义。接下来，我们要确保具有特定头层的模型，例如`BrandNewBertForMaskedLM`，不继承自`BrandNewBertModel`，而是将`BrandNewBertModel`作为一个可以在其前向传递中调用的组件，以保持抽象级别低。每个新模型都需要一个配置类，称为`BrandNewBertConfig`。此配置始终作为属性存储在[`PreTrainedModel`]中，因此可以通过`config`属性访问所有继承自`BrandNewBertPreTrainedModel`的类：

```python
model = BrandNewBertModel.from_pretrained("brandy/brand_new_bert")
model.config  # model可以访问其配置
```

与模型相似，配置从[`PretrainedConfig`]继承基本的序列化和反序列化功能。请注意，配置和模型始终以两种不同的格式进行序列化-模型以*pytorch_model.bin*文件格式，配置以*config.json*文件格式。调用[`~PreTrainedModel.save_pretrained`]将自动调用[`~PretrainedConfig.save_pretrained`]，因此模型和配置都将保存起来。


### 代码风格

当编写新模型时，请记住🤗 Transformers是一个有个人特点的库，我们在编写代码方面有一些小技巧:-)

1. 您的模型的前向传递应该完全在建模文件中编写，并且与库中的其他模型完全独立。如果您想重用来自其他模型的块，请复制代码，并在顶部添加一个`# Copied from`的注释（参见[此处](https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/roberta/modeling_roberta.py#L160)以获得良好的示例和[此处](pr_checks.md#check-copies)以获得有关复制注释更多的文档）。
2. 代码应该可以被任何非英语为母语的人完全理解。这意味着您应该选择具有描述性的变量名，并避免使用缩写词。例如，“activation”比“act”更好。强烈不建议使用单个字母的变量名，除非它是for循环中的索引。
3. 通常我们更喜欢较长和显式的代码，而不是短小神奇的代码。
4. 在PyTorch中，避免以`nn.Sequential`为基类进行子类化，而是以`nn.Module`为基类并编写前向传递，这样使用您的代码的任何人都可以通过添加打印语句或断点来快速调试它。
5. 您的函数签名应该有类型注释。对于其他内容，良好的变量名比类型注释更易读和易懂。

### 分词器的概述

你还没准备好:-（很快会添加这个部分！

## 将模型添加到🤗 Transformers的分步骤介绍

每个人在端口模型时都有不同的喜好，因此您可以参考其他贡献者将模型端口到Hugging Face的简要摘要，这对您可能非常有帮助。以下是关于如何端口模型的社区博客列表：

1. [导入GPT2模型](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28)，by [Thomas](https://huggingface.co/thomwolf)
2. [导入WMT19 MT模型](https://huggingface.co/blog/porting-fsmt)，by [Stas](https://huggingface.co/stas)

凭借我们的经验，当添加模型时需要牢记的最重要的事情是：

- 不要重复造轮子!您将为新🤗 Transformers模型添加的代码的大部分可能已经存在于🤗 Transformers的某个地方。花一些时间找到可以复制的类似的已存在的模型和分词器。[grep](https://www.gnu.org/software/grep/) 和 [rg](https://github.com/BurntSushi/ripgrep) 是你的好朋友。请注意，您的模型的分词器很有可能是基于一种模型实现的，而您的模型的建模代码则基于另一种实现。例如，FSMT的建模代码基于BART，而FSMT的分词器代码基于XLM。
- 这更多是一项工程挑战而不是科学挑战。您应该在创建一个高效的调试环境上花更多时间，而不是试图理解论文中的所有理论方面。
- 当您遇到困难时，寻求帮助！模型是🤗 Transformers的核心组成部分，因此Hugging Face非常乐意在每个步骤上帮助您添加您的模型。如果您发现自己没有取得进展，请不要犹豫提问。

接下来，我们试图提供一个我们发现在将模型迁移到🤗 Transformers时最有用的一般性的步骤指南。

以下列表概述了必须完成的添加模型的所有工作，并可以作为*待办事项清单*使用：

☐ （可选）理解BrandNewBert的理论方面<br>
☐ 准备好了🤗 Transformers的开发环境<br>
☐ 在原始存储库中配置调试环境<br>
☐ 编写可以使用原始存储库和检查点成功运行`forward()`传递的脚本<br>
☐ 成功将模型框架添加到🤗 Transformers<br>
☐ 成功将原始检查点转换为🤗 Transformers检查点<br>
☐ 成功在🤗 Transformers中运行`forward()`传递，输出与原始检查点相同<br>
☐ 完成了🤗 Transformers中的模型测试<br>
☐ 成功将分词器添加到🤗 Transformers<br>
☐ 运行端到端集成测试<br>
☐ 完成了文档<br>
☐ 将模型权重上传到Hub<br>
☐ 提交拉取请求<br>
☐ （可选）添加演示笔记本


首先，我们通常建议您先对 BrandNewBert 进行一下理论上的了解。然而，如果您更喜欢在实践中了解模型的理论方面，则可以直接深入到 BrandNewBert 的代码库中。选择这个选项可能更适合于您，如果您的工程技巧胜过理论技巧，如果您在理解 BrandNewBert 的论文中遇到困难，或者如果您更喜欢编程而不是阅读科学论文。

### 1. （可选）BrandNewBert 的理论方面

如果 BrandNewBert 的论文存在，您应该花一些时间阅读它。有可能有很多难以理解的部分。如果是这种情况，不要担心 - 不用担心！目标不是对论文进行深入的理论了解，而是从中提取在有效地在🤗 Transformers 中重新实现模型所需的必要信息。话虽如此，您不需要花太多时间在理论方面上，而是应该专注于实践方面，即：

- *brand_new_bert* 是什么类型的模型？类似BERT的仅编码器模型？类似GPT-2的仅解码器模型？类似BART的编码器-解码器模型？如果不熟悉这些差异，请参阅[model_summary](https://github.com/huggingface/transformers/blob/master/README.md#model-summary)。
- *brand_new_bert* 有哪些应用？文本分类？文本生成？Seq2Seq任务，例如摘要？
- 这个模型的何种功能使其与BERT/GPT-2/BART不同？
- 与 *brand_new_bert* 最相似的[🤗 Transformers 模型](https://huggingface.co/transformers/#contents)是哪一个？
- 使用了什么类型的分词器？是 sentencepiece 分词器？还是 word piece 分词器？它与BERT或BART使用的分词器是相同的吗？

当您觉得对模型的体系结构有了良好的概览后，您可能希望向Hugging Face团队提问您可能有的任何问题。这可能包括有关模型的体系结构、注意层等方面的问题。我们将非常乐意为您提供帮助。

### 2. 接下来准备您的环境

1. 点击存储库页面上的“Fork”按钮，将[库](https://github.com/huggingface/transformers)fork到您的GitHub用户账户下。这将在您的GitHub用户账户下创建该代码的一个副本。

2. 将您的 `transformers` fork 克隆到本地磁盘，并添加base存储库作为远程存储库：

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

3. 设置开发环境，例如运行以下命令设置虚拟环境：

```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```

根据您的操作系统，由于`🤗 Transformers`的可选依赖项数量不断增加，您可能会遇到此命令失败的情况。如果是这样，请确保已安装您要使用的深度学习框架（PyTorch，TensorFlow和/或Flax），然后执行：

```bash
pip install -e ".[quality]"
```

对于大多数使用情况来说，这应该足够了。然后可以返回到上一级目录：

```bash
cd ..
```

4. 我们建议您将 PyTorch 版本的 *brand_new_bert* 添加到 Transformers。要安装 PyTorch，请按照 https://pytorch.org/get-started/locally/ 上的说明进行操作。

**注意：**您不需要安装CUDA。使新模型在CPU上运行就足够了。

5. 要导入 *brand_new_bert*，您还需要访问其原始存储库：

```bash
git clone https://github.com/org_that_created_brand_new_bert_org/brand_new_bert.git
cd brand_new_bert
pip install -e .
```

现在您已经设置了一个开发环境，可以将 *brand_new_bert* 导入到 🤗 Transformers 中。

### 3.-4. 使用原始存储库运行预训练的检查点

在最初阶段，您将在原始的 *brand_new_bert* 存储库上工作。通常，原始实现非常“研究型”。意思是文档可能不全，代码可能难以理解。但这正是您重新实现 *brand_new_bert* 的原因。在Hugging Face，我们的主要目标之一是让人们站在巨人的肩膀上，这在这里非常明显，因为我们带来了一个可以工作的模型，并重新编写它，使其尽可能“易于访问、用户友好和美观”。这是将模型重拾到🤗 Transformers中的最重要的动力 - 试图将复杂的新NLP技术提供给**每个人**。



你应该首先通过进入原始仓库来开始。

在原始仓库中成功运行官方预训练模型通常是**最困难的**一步。根据我们的经验，花一些时间熟悉原始代码非常重要。你需要弄清楚以下几点：

- 在哪里找到预训练权重？
- 如何将预训练权重加载到相应的模型中？
- 如何独立运行分词器而不依赖模型？
- 追踪一次前向传递，以了解哪些类和函数需要一个简单的前向传递。通常，你只需要重新实现这些函数。
- 能够定位模型的重要组件：模型的类在哪里？是否有模型的子类，例如EncoderModel，DecoderModel？注意力层在哪里？是否有多个不同的注意力层，例如*self-attention*、*cross-attention*...？
- 如何在原始环境中调试模型？你需要添加`print`语句吗？你可以使用像*ipdb*这样的交互式调试器吗？还是应该使用像PyCharm这样高效的IDE来调试模型？

在开始移植过程之前，能够**高效地**调试原始仓库中的代码非常重要！此外，记住你是在使用开源库进行工作，因此不要犹豫在原始仓库中打开问题，甚至提交拉取请求。这个仓库的维护者很可能非常乐意有人研究他们的代码！

此时，你可以根据自己的偏好选择调试环境和策略来调试原始模型。我们强烈建议使用Jupyter笔记本进行工作，如果你熟悉Jupyter笔记本的使用。

Jupyter笔记本的优点在于它们允许逐个单元格执行，这有助于更好地将逻辑组件进行分割，并且可以更快地进行调试循环，因为可以存储中间结果。此外，笔记本通常更容易与其他贡献者共享，如果你想向Hugging Face团队寻求帮助，这可能非常有用。如果您熟悉Jupyter笔记本，我们强烈建议您使用它们。

Jupyter笔记本的明显缺点是，如果您不习惯使用它们，您将不得不花一些时间适应新的编程环境，并且可能无法再使用您已知的调试工具，例如`ipdb`。

对于运行原始模型的调试环境，通常有两种选择：

- [Jupyter笔记本](https://jupyter.org/) / [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)
- 本地Python脚本。

Jupyter笔记本的优点是它们可以按单元格逐个执行，这对于将逻辑组件彼此分离和更快速地进行调试循环很有帮助，因为可以存储中间结果。此外，相对于本地脚本，笔记本通常更容易与其他贡献者共享。

Jupyter笔记本的明显缺点是，如果您不习惯使用它们，您将不得不花一些时间适应新的编程环境，并且可能无法再使用您已知的调试工具，例如`ipdb`。

无论您选择哪种策略，推荐的过程通常是相同的，即首先从调试起始层开始，然后再调试最后层。

建议按照以下顺序检索输出，无论您选择的是哪种策略：

1. 检索传递给模型的输入ID
2. 检索词嵌入
3. 检索第一个Transformer层的输入
4. 检索第一个Transformer层的输出
5. 检索后续n - 1个Transformer层的输出
6. 检索整个BrandNewBert模型的输出

其中，输入ID应为整数数组，例如 `input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]`

以下各层的输出通常由多维浮点数组组成，可能如下所示：

```
[[
 [-0.1465, -0.6501,  0.1993,  ...,  0.1451,  0.3430,  0.6024],
 [-0.4417, -0.5920,  0.3450,  ..., -0.3062,  0.6182,  0.7132],
 [-0.5009, -0.7122,  0.4548,  ..., -0.3662,  0.6091,  0.7648],
 ...,
 [-0.5613, -0.6332,  0.4324,  ..., -0.3792,  0.7372,  0.9288],
 [-0.5416, -0.6345,  0.4180,  ..., -0.3564,  0.6992,  0.9191],
 [-0.5334, -0.6403,  0.4271,  ..., -0.3339,  0.6533,  0.8694]]],
```

我们希望每个添加到🤗Transformers的模型都通过一些集成测试，这意味着原始模型和🤗Transformers中的重新实现版本必须给出准确相同的输出，精度为0.001！由于在不同的库框架中，相同的模型可能会出现略微不同的输出，这是正常的，我们接受1e-3（0.001）的误差容限。如果模型输出几乎相同是不够的，它们必须几乎相同。因此，您将多次将🤗Transformers版本的中间输出与旧实现版本的中间输出进行对比，因此，原始仓库中的**高效**调试环境绝对至关重要。以下是使您的调试环境尽可能高效的一些建议。

- 找到调试中间结果的最佳方法。原始仓库使用PyTorch编写？那你可能要花时间编写一个更长的脚本，将原始模型分解成较小的子组件，以检索中间值。原始仓库使用TensorFlow 1？那么你可能需要依赖于TensorFlow的打印操作，如[tf.print](https://www.tensorflow.org/api_docs/python/tf/print) 来输出中间值。原始仓库使用Jax？那么请确保在运行前向传递时模型**未进行编译**，例如，请查看[此链接](https://github.com/google/jax/issues/196)。
- 使用尽可能小的预训练检查点。检查点越小，你的调试循环就越快。如果你的预训练模型太大，以至于前向传递需要超过10秒的时间，那就不是高效的了。如果只能使用非常大的检查点，则更有意义的做法可能是在新环境中创建一个带有随机初始化权重的虚拟模型，并将这些权重与🤗Transformers版本的模型进行比较。
- 确保使用最简单的方式调用原始仓库中的前向传递。 ideally，你希望找到原始仓库中**只**调用一次前向传递的函数，即通常称为`predict`、`evaluate`、`forward`或`__call__`的函数。你不希望调试一个多次调用`forward`函数的函数，例如用于生成文本的`autoregressive_sample`、`generate`。
- 尝试将分词过程与模型的*forward*传递分离开来。如果原始仓库中显示了一些需要输入字符串的示例，那么尝试找出在前向调用中字符串输入在哪一步被转换为输入ID，并从这一步开始。这可能意味着您可能需要自己编写一个小脚本或更改原始代码，以便直接输入ID而不是输入字符串。
- 确保调试环境中的模型**不处于训练模式**，这通常会导致模型产生随机输出，因为模型中有多个dropout层。确保调试环境中的前向传递是**确定性的**，以便不使用dropout层。如果旧实现和新实现位于相同的框架中，可以使用*transformers.utils.set_seed*。

下一节将为您提供更具体的详细信息/提示，说明如何对*brand_new_bert*进行调试。

### 5.-14. 将BrandNewBert移植到🤗Transformers

接下来，您可以开始向🤗Transformers添加新代码。进入您的🤗Transformers分支克隆：

```bash
cd transformers
```

在您正在添加的模型的体系结构与现有模型的体系结构完全匹配的特殊情况下，您只需要添加一个转换脚本，如[此部分](#撰写转换脚本)所述。在这种情况下，您可以直接重用已存在模型的整个模型体系结构。

否则，让我们开始生成一个新模型。您有两个选择：

- `transformers-cli add-new-model-like`，以添加类似于现有模型的新模型
- `transformers-cli add-new-model`，以根据我们的模板添加新模型（将看起来像BERT或Bart，具体取决于您选择的模型类型）

在这两种情况下，您都会被要求填写有关您的模型的基本信息的调查问卷。

**在主要的huggingface/transformers仓库上开一个Pull Request**

在开始修改自动生成的代码之前，现在是时候在🤗Transformers上打开一个“正在进行的工作（WIP）”拉取请求（PR）了，例如“[WIP] 添加*brand_new_bert*”，这样您和Hugging Face团队就可以同时在🤗Transformers中集成模型。

您应该执行以下步骤：

1. 从您的主分支创建一个具有描述性名称的分支

```bash
git checkout -b add_brand_new_bert
```

2. 提交自动生成的代码：

```bash
git add .
git commit
```

3. 获取并将基础分支（main）的变更合并到当前分支：

```bash
git fetch upstream
git rebase upstream/main
```

4. 将变更推送到您的账户：

```bash
git push -u origin a-descriptive-name-for-my-changes
```

5. 完成后，前往 GitHub 上您的分支页面。单击“Pull Request”。请确保将Hugging Face团队的成员的GitHub用户名添加为审阅者，以便Hugging Face团队在未来的更改中得到通知。

6. 在 GitHub 上的拉取请求页面上的右侧，点击“Convert to draft”将其更改为草稿。

接下来，每当你取得一些进展时，不要忘记提交你的工作并将其推送到您的账户，以便在拉取请求中显示。此外，您应该定期使用以下命令将您的工作与当前的主分支进行更新：

```bash
git fetch upstream
git merge upstream/main
```

一般来说，关于模型或您的实现的所有问题都应该在您的PR中提出，并在PR中进行讨论/解决。这样，当您提交新代码或遇到问题时，Hugging Face团队将始终收到通知。通常在PR中指出您添加的代码非常有帮助，这样Hugging Face团队就可以高效地理解您的问题或疑问。

为此，您可以转到“Files changed”选项卡，查看您的所有更改，转到相关的行，并单击“+”符号添加评论。每当一个问题或问题得到解决时，您可以点击创建的评论的“Resolve”按钮。

通过相同的方式，Hugging Face团队在代码审核时会提出评论。我们建议在GitHub上通过PR提出大部分问题。对于一些不太适合公开的非常普遍的问题，请随时通过Slack或电子邮件联系Hugging Face团队。

**5. 修改为brand_new_bert生成的模型代码**

首先，我们将只关注模型本身，不考虑分词器。所有相关的代码应该在`src/transformers/models/brand_new_bert/modeling_brand_new_bert.py`和`src/transformers/models/brand_new_bert/configuration_brand_new_bert.py`这两个生成的文件中找到。

现在你终于可以开始编码了 :). 生成的代码位于 `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py`，如果是编码器-解码器模型，则该代码将与 BART 的体系结构相同，如果是仅编码器模型，则与 BERT 的体系结构相同。此时，你应该回顾一下关于模型的理论方面的内容："该模型与 BERT 或 BART 有何不同？" 根据这些差异来实现更改，通常意味着更改 "self-attention" 层、归一化层的顺序等... 另外，通常查看 Transformers 中已存在模型的类似体系结构，以更好地了解应如何实现自己的模型，这通常是有用的。

请注意，此时并不需要确保代码完全正确或干净。相反，建议首先在 `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py` 添加一个未加工的、直接粘贴的原始代码版本，直到你觉得已添加了所有必要的代码。根据我们的经验，使用转换脚本进行快速添加所需代码并增强/更正代码的效率更高，如下一部分所述。这一步的关键是要确保你能够实例化🤗 Transformers 实现的 *brand_new_bert*，即下面的命令应该可以运行成功:

```python
from transformers import BrandNewBertModel, BrandNewBertConfig

model = BrandNewBertModel(BrandNewBertConfig())
```

上述命令将创建一个满足 `BrandNewBertConfig()` 默认参数定义的模型，具有随机权重，从而确保所有组件的 `init()` 方法正常工作。

请注意，所有随机初始化应在 `BrandnewBertPreTrainedModel` 类的 `_init_weights` 方法中进行。它应该根据配置的变量来初始化所有叶子模块。这里是一个使用 BERT 的 `_init_weights` 方法示例:

```py
def _init_weights(self, module):
    """初始化权重"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

如果需要，我们可以根据需要有一些更多自定义方案的初始化。例如，在 `Wav2Vec2ForPreTraining` 中，最后两个线性层需要使用常规的 PyTorch `nn.Linear` 初始化，但所有其他层应使用上面所述的初始化方法。编码如下:

```py
def _init_weights(self, module):
    """初始化权重"""
    if isinstnace(module, Wav2Vec2ForPreTraining):
        module.project_hid.reset_parameters()
        module.project_q.reset_parameters()
        module.project_hid._is_hf_initialized = True
        module.project_q._is_hf_initialized = True
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
```

`_is_hf_initialized` 标志在内部用于确保我们只初始化子模块一次。通过将其设置为 `True`，我们确保自定义的初始化不会在以后被覆盖，在这种情况下，初始化权重的函数不会应用到它们上面。

**6. 编写转换脚本**

接下来，你应该编写一个转换脚本，让你能够将你在原始库中用于调试 *brand_new_bert* 的检查点转换为与你刚刚创建的🤗 Transformers 实现的 *brand_new_bert* 兼容的检查点。建议不要从头开始编写转换脚本，而是查看已存在的🤗 Transformers 转换脚本，找到用于转换与 *brand_new_bert* 相同框架编写的类似模型的脚本，并做出适应你的情况的轻微修改。如果需要，可以向 Hugging Face 团队寻求帮助，让他们为你指出一个用于你的模型的类似的已存在转换脚本。

- 如果你要将模型从 TensorFlow 迁移到 PyTorch，可以从 BERT 的转换脚本 [这里](https://github.com/huggingface/transformers/blob/7acfa95afb8194f8f9c1f4d2c6028224dbed35a2/src/transformers/models/bert/modeling_bert.py#L91) 开始
- 如果你要将模型从 PyTorch 迁移到 PyTorch，可以从 BART 的转换脚本 [这里](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py) 开始

接下来，我们将快速解释 PyTorch 模型如何存储层权重并定义层名称。在 PyTorch 中，层的名称由你给层的类属性的名称定义。让我们定义一个名为 `SimpleModel` 的 PyTorch 模型，如下所示:

```python
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(10, 10)
        self.intermediate = nn.Linear(10, 10)
        self.layer_norm = nn.LayerNorm(10)
```

现在我们可以创建该模型定义的实例，它将使用随机权重填充所有权重: `dense`、`intermediate`、`layer_norm`。我们可以打印该模型以查看其体系结构：

```python
model = SimpleModel()

print(model)
```

这将打印出以下内容:

```
SimpleModel(
  (dense): Linear(in_features=10, out_features=10, bias=True)
  (intermediate): Linear(in_features=10, out_features=10, bias=True)
  (layer_norm): LayerNorm((10,), eps=1e-05, elementwise_affine=True)
)
```

我们可以看到，层的名称由 PyTorch 中的类属性名称定义。你可以打印出特定层的权重值:

```python
print(model.dense.weight.data)
```

可以看到权重已经随机初始化为:

```
tensor([[-0.0818,  0.2207, -0.0749, -0.0030,  0.0045, -0.1569, -0.1598,  0.0212,
         -0.2077,  0.2157],
        [ 0.1044,  0.0201,  0.0990,  0.2482,  0.3116,  0.2509,  0.2866, -0.2190,
          0.2166, -0.0212],
        [-0.2000,  0.1107, -0.1999, -0.3119,  0.1559,  0.0993,  0.1776, -0.1950,
         -0.1023, -0.0447],
        [-0.0888, -0.1092,  0.2281,  0.0336,  0.1817, -0.0115,  0.2096,  0.1415,
         -0.1876, -0.2467],
        [ 0.2208, -0.2352, -0.1426, -0.2636, -0.2889, -0.2061, -0.2849, -0.0465,
          0.2577,  0.0402],
        [ 0.1502,  0.2465,  0.2566,  0.0693,  0.2352, -0.0530,  0.1859, -0.0604,
          0.2132,  0.1680],
        [ 0.1733, -0.2407, -0.1721,  0.1484,  0.0358, -0.0633, -0.0721, -0.0090,
          0.2707, -0.2509],
        [-0.1173,  0.1561,  0.2945,  0.0595, -0.1996,  0.2988, -0.0802,  0.0407,
          0.1829, -0.1568],
        [-0.1164, -0.2228, -0.0403,  0.0428,  0.1339,  0.0047,  0.1967,  0.2923,
          0.0333, -0.0536],
        [-0.1492, -0.1616,  0.1057,  0.1950, -0.2807, -0.2710, -0.1586,  0.0739,
          0.2220,  0.2358]]).
```

在转换脚本中，你应该使用检查点中相应层的准确权重，填充这些随机初始化的权重。例如:

```python
# 检索匹配的层权重，例如通过递归算法
layer_name = "dense"
pretrained_weight = array_of_dense_layer

model_pointer = getattr(model, "dense")

model_pointer.weight.data = torch.from_numpy(pretrained_weight)
```

在这样做之前，必须验证你的 PyTorch 模型的每个随机初始化权重及其对应的预训练检查点权重在**形状和名称**上完全匹配。为此，**必须**为形状添加断言语句，并打印出检查点权重的名称。例如，你应该添加如下语句:

```python
assert (
    model_pointer.weight.shape == pretrained_weight.shape
), f"Pointer shape of random weight {model_pointer.shape} and array shape of checkpoint weight {pretrained_weight.shape} mismatched"
```

此外，你还应该打印出两个权重的名称，以确保它们匹配，例如:

```python
logger.info(f"Initialize PyTorch weight {layer_name} from {pretrained_weight.name}")
```

如果形状或名称不匹配，你可能错误地将检查点权重分配给了随机初始化的🤗 Transformers 实现的层。

形状不正确很可能是由于在 `BrandNewBertConfig()` 中设置了不完全匹配的配置参数，这些参数与要转换的检查点使用的参数不完全匹配。然而，也可能是因为 PyTorch 的层实现要求在进行设置之前先对权重进行转置。

最后，你还应该检查是否已初始化**全部**所需权重，并打印出未用于初始化的所有检查点权重，以确保正确转换模型。在转换所有权重到🤗 Transformers 实现之后，可以将模型保存在你选择的文件夹 `/path/to/converted/checkpoint/folder` 下，其中应该包含一个 `pytorch_model.bin` 文件和一个 `config.json` 文件:

```python
model.save_pretrained("/path/to/converted/checkpoint/folder")
```

**7. 实现前向传递**

在成功加载预训练权重到🤗 Transformers 实现之后，现在你应该确保正确实现了前向传递。在[了解原始库](#34-run-a-pretrained-checkpoint-using-the-original-repository) 中，你已经创建了一个使用原始库运行模型前向传递的脚本。现在，你应该编写一个类似的脚本，使用🤗 Transformers 实现而不是原始实现。它应该如下所示:

```python
model = BrandNewBertModel.from_pretrained("/path/to/converted/checkpoint/folder")
input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]
output = model(input_ids).last_hidden_states
```

很可能🤗 Transformers 实现和原始模型实现的输出并不完全相同，或者前向传递会引发错误。不要灰心，这是正常情况！首先，你应该确保前向传递不会引发任何错误。经常会出现错误的情况是使用了不正确的维度，导致 "维度不匹配" 错误，或者使用了错误的数据类型对象，比如将 `torch.long` 误用为 `torch.float32`。如果无法解决某些错误，请随时向 Hugging Face 团队寻求帮助。

确保两个实现的输出精度达到 `1e-3` 是确保🤗 Transformers 实现正确的最后一部分。首先，你应该确保输出形状完全相同，即 `outputs.shape` 在🤗 Transformers 实现的脚本和原始实现的脚本中产生相同的值。接下来，你应该确保输出值也是相同的。这是增加新模型中最困难的部分之一。导致输出不相同的常见错误包括:

- 没有添加某些层，比如没有添加一个 "activation" 层，或者忘记了残差连接
- 单词嵌入矩阵没有连接
- 使用了错误的位置嵌入，因为原始实现使用了偏移
- 在前向传递期间应用了丢弃。要解决这个问题，请确保设置了 *model.training* 为 False，并且在前向传递过程中没有错误地激活了丢弃层，例如将 *self.training* 传递给 [PyTorch 的 functional dropout](https://pytorch.org/docs/stable/nn.functional.html?highlight=dropout#torch.nn.functional.dropout)

解决问题的最好方式通常是并排查看原始实现和🤗 Transformers 实现的前向传递，并检查是否存在任何差异。理想情况下，应该在前向传递的两个实现中调试/打印中间输出，以找到🤗 Transformers 实现与原始实现输出不同的确切网络位置。首先，确保两个脚本中硬编码的 `input_ids` 相同。接下来，验证 "input_ids" 的第一个转换的输出（通常是词嵌入）是否完全相同。然后，从网络的最后一层开始，逐层检查网络的输出是否相同。在某个点上，你会注意到两个实现之间的差异，这应该指出🤗 Transformers 实现中的错误。根据我们的经验，一种简单而有效的方法是在原始实现和🤗 Transformers 实现中添加许多打印语句，并在网络的相同位置添加这些打印语句，然后逐步删除显示相同中间结果的打印语句。

当你确信两个实现会产生相同的输出时，可以通过使用 `torch.allclose(original_output, output, atol=1e-3)` 来验证输出的相等性，你已经完成了最困难的部分！恭喜你——剩下的工作应该很轻松 😊。

**8. 添加所有必要的模型测试**

此时，你已经成功添加了一个新的模型。然而，很可能该模型还不完全符合所需的设计。为确保实现与🤗 Transformers 完全兼容，应通过运行所有常规测试来验证。Cookiecutter 应该自动为你的模型添加了一个测试文件，可能在相同目录下的 `tests/models/brand_new_bert/test_modeling_brand_new_bert.py`。运行该测试文件以验证所有常规测试是否通过:

```bash
pytest tests/models/brand_new_bert/test_modeling_brand_new_bert.py
```

在修复了所有常规测试之后，现在关键是确保所有你做的工作都得到了很好的测试，以便：

- a) 社区成员可以通过查看 *brand_new_bert* 的特定测试来方便地理解你的工作
- b) 对模型进行的任何未来更改都不会破坏模型的重要功能的测试

首先，应该添加集成测试。这些集成测试本质上与您之前用于实现🤗 Transformers模型的调试脚本相同。已经通过Cookiecutter添加了这些模型测试的模板，名为`BrandNewBertModelIntegrationTests`，只需由您填写。为确保这些测试通过，请运行

```bash
RUN_SLOW=1 pytest -sv tests/models/brand_new_bert/test_modeling_brand_new_bert.py::BrandNewBertModelIntegrationTests
```

<Tip>

如果您使用Windows，应将`RUN_SLOW=1`替换为`SET RUN_SLOW=1`

</Tip>

其次，还应在`BrandNewBertModelTester`/`BrandNewBertModelTest`下单独对与*brand_new_bert*特有的所有特性进行测试。这一部分经常被忽视，但在两个方面非常有用：

- 它有助于通过展示*brand_new_bert*的特殊功能如何工作来将您在模型添加过程中获得的知识传递给社区。
- 未来的贡献者可以通过运行这些特殊测试来快速测试模型的变化。

**9. 实现分词器**

接下来，我们应该添加*brand_new_bert*的分词器。通常，分词器与🤗 Transformers的已有分词器等同或非常相似。

非常重要的是找到/提取原始的分词器文件，并设法将该文件加载到🤗 Transformers的分词器实现中。

为确保分词器可以正常工作，建议先在原始存储库中创建一个脚本，该脚本输入一个字符串并返回`input_ids`。它可能类似于以下伪代码：

```python
input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
model = BrandNewBertModel.load_pretrained_checkpoint("/path/to/checkpoint/")
input_ids = model.tokenize(input_str)
```

您可能需要再次深入研究原始存储库，以找到正确的分词器函数，或者甚至可能需要对原始存储库的克隆进行更改，以仅输出`input_ids`。一旦编写了使用原始存储库的功能性分词脚本，还应创建一个类似于以下内容的🤗 Transformers的脚本：

```python
from transformers import BrandNewBertTokenizer

input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."

tokenizer = BrandNewBertTokenizer.from_pretrained("/path/to/tokenizer/folder/")

input_ids = tokenizer(input_str).input_ids
```

当`input_ids`和这个最终的步骤得出相同的值时，还应添加一个分词器测试文件。

与*brand_new_bert*的模型测试文件类似，*brand_new_bert*的分词器测试文件还应包含一些硬编码的集成测试。

**10. 运行端到端的集成测试**

在添加了分词器之后，您还应在🤗 Transformers中的`tests/models/brand_new_bert/test_modeling_brand_new_bert.py`上添加一些端到端的集成测试，使用模型和分词器。

这样的测试应该通过有意义的文本到文本样本展示🤗 Transformers实现按预期工作的情况。有意义的文本到文本样本可以包括*例如*源到目标翻译对、文章到摘要对、问题到答案对等。如果没有将任何导入的检查点微调到下游任务，只需依赖模型测试即可。

最后，请确保在GPU上运行所有测试，以确保模型完全功能。可能会忘记在模型的内部张量中添加一些`.to(self.device)`语句，此类测试将显示出错误。如果无法访问GPU，Hugging Face团队可以负责为您运行这些测试。

**11. 添加文档注释**

现在，*brand_new_bert*的所有必要功能都已添加完成-您离完成不远了！只剩下添加良好的文档注释和文档页面了。Cookiecutter应该已经添加了一个名为`docs/source/model_doc/brand_new_bert.md`的模板文件，您需要填写该文件。在使用您的模型之前，模型的用户通常会首先查看此页面。因此，文档必须易于理解和简洁。为了展示模型的正确使用方法，添加一些*提示*非常有用。不要犹豫与Hugging Face团队讨论有关文档注释的问题。

接下来，确保添加到`src/transformers/models/brand_new_bert/modeling_brand_new_bert.py`的文档注释是正确的，并包含所有必要的输入和输出。我们有一份详细的关于撰写文档和我们的文档字符串格式的指南。在处理🤗 Transformers的代码之前，提醒自己始终要对待文档至少与对待代码一样的谨慎，因为文档通常是社区与模型的第一个接触点。

**代码重构**

太棒了，您现在已经添加了*brand_new_bert*的所有必要代码。此时，您应该通过运行以下命令来修正一些潜在的不正确的代码风格：

```bash
make style
```

并验证您的代码风格是否通过质量检查：

```bash
make quality
```

🤗 Transformers中可能仍然存在一些非常严格的设计测试可能失败的情况，这会在您的Pull Request的测试中显示出来。这通常是由于文档字符串中缺少的一些信息或一些不正确的命名所致。如果您在这一步遇到问题，Hugging Face团队肯定会帮助您。

最后，在确保代码正常工作之后，对代码进行重构是个好主意。所有测试都通过后，现在是时候再次检查添加的代码，并进行一些重构了。

您现在已经完成了编码部分，恭喜！🎉您真棒！😎

**12. 将模型上传到模型库**

在最后一部分中，您应该将所有检查点转换并上传到模型库，并为每个上传的模型检查点添加一个模型卡片。您可以通过阅读我们的[模型共享和上传页面](https://huggingface.co/transformers/model_sharing.html)来熟悉模型库的功能。您应该在这里与Hugging Face团队一起工作，以为每个检查点选择适当的名称，并获得在*brand_new_bert*的作者组织下上传模型所需的访问权限。`transformers`中的所有模型都具有`push_to_hub`方法，这是将检查点快速、高效地推送到模型库的方法。下面是一个小片段：

```python
brand_new_bert.push_to_hub("brand_new_bert")
# 取消以下行的注释以推送到一个组织。
# brand_new_bert.push_to_hub("<organization>/brand_new_bert")
```

创建适合每个检查点的模型卡片是值得花些时间的。模型卡片应突出显示此特定检查点的特定特征，例如，检查点在哪些数据集上进行了预训练/微调？在哪项下游任务上应该使用该模型？还应包含一些关于如何正确使用模型的代码。

**13. (可选) 添加notebook**

添加一本笔记本，详细介绍了如何在推断和/或下游任务上对*brand_new_bert*进行微调，会非常有帮助。这不是合并您的PR所必需的，但对于社区来说非常有用。

**14. 提交您完成的PR**

您现在已经完成了编程部分，可以进行最后一步，即将您的PR合并到主分支中。通常，在此时，Hugging Face团队应该已经帮助您了，但值得花些时间对您完成的PR进行一个很好的描述，并根据需要向代码添加评论，如果您想要向审阅人员指出某些设计选择。

### 分享您的工作！！

现在，是时候从社区中获得一些对您工作的认可了！完成模型添加对于Transformers和整个NLP社区来说是一个重要的贡献。您的代码和移植的预训练模型肯定会被数百甚至数千名开发人员和研究人员使用。为您的工作感到自豪，并将您的成就与社区分享。

**您又为社区提供了一个非常容易访问的模型！🤯**