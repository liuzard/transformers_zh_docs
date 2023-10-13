<!--版权 2022年HuggingFace团队保留所有权利。

根据Apache许可证，版本2.0（“许可证”）授权你使用此文件，但使用该文件必须符合许可证的规定。
你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本许可证下的软件分发是基于“按原样”提供的，
没有任何明示或暗示的担保或条件。有关许可的更多信息，请参见许可证。

⚠️请注意，此文件以Markdown格式编写，但包含我们的doc-builder（类似MDX）的特定语法，
如果你的Markdown查看器无法正确渲染。

-->

# 如何将🤗Transformers模型转换为TensorFlow？

在设计应用程序时，可以使用多个可用的框架与🤗Transformers，这使你可以充分发挥他们的优势，但这意味着需要根据每个模型添加兼容性。好消息是，将现有模型添加到TensorFlow中比[从头开始添加新模型](add_new_model.md)更简单！无论你是希望更深入地了解大型TensorFlow模型，还是希望进行一次重大的开源贡献，或将TensorFlow应用于你选择的模型，本指南都适用于你。

本指南可以使你，作为我们社区的一员，在获得Hugging Face团队的最低监督下，为使用🤗Transformers的TensorFlow模型权重和/或架构做出贡献。编写新模型并非易事，但希望本指南能使这个过程更加轻松愉快，让其成为一次轻松的散步而不仅仅是过山车之旅。汇集我们的集体经验对于使这个过程变得越来越容易至关重要，因此我们强烈鼓励你对本指南提出改进建议！

在深入探讨之前，如果你对🤗Transformers还不熟悉，建议你查看以下资源：
- [🤗Transformers概览](add_new_model.md#general-overview-of-transformers)
- [Hugging Face的TensorFlow理念](https://huggingface.co/blog/tensorflow-philosophy)

在本指南的其余部分，你将了解如何添加新的TensorFlow模型架构所需的步骤、将PyTorch转换为TensorFlow模型权重的过程以及如何高效地调试跨ML框架的不匹配。让我们开始吧！

<Tip>

你不确定是否已有相应的TensorFlow架构对应你希望使用的模型吗？

&nbsp;

检查你选择的模型的`config.json`文件的`model_type`字段
（[示例](https://huggingface.co/bert-base-uncased/blob/main/config.json#L14)）。如果🤗Transformers中相应的模型文件夹中有以“modeling_tf”开头的文件，这意味着它具有相应的TensorFlow架构
（[示例](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert)）。

</Tip>


## 添加TensorFlow模型架构代码的逐步指南

有多种方法可以设计大型模型架构，以及多种实现该设计的方式。然而，你可能还记得我们的[🤗Transformers概览](add_new_model.md#general-overview-of-transformers)中，
我们是一个有着自己想法的团队 - 使用🤗Transformers的易用性依赖于一致的设计选择。根据经验，我们可以告诉你关于添加TensorFlow模型的一些重要事项：

- 不要重复造轮子！往往至少有两个参考实现值得检查：你正在实现的模型的PyTorch等效实现以及同一类问题的其他TensorFlow模型。
- 优秀的模型实现能经受时间的考验。发生这种情况不是因为代码漂亮，而是因为代码清晰易懂、易于调试和建立。如果你通过在TensorFlow模型中复制其他TensorFlow模型中的模式并将其与PyTorch实现的不匹配最小化来帮助维护人员简化TensorFlow实现，那么你的贡献将会长寿。
- 当卡住时请寻求帮助！🤗Transformers团队在这里为你提供帮助，我们可能已经找到了你面临问题的解决方案。

以下是添加TensorFlow模型架构所需的步骤概述：
1. 选择要转换的模型
2. 准备transformers的开发环境
3. （可选）了解理论方面和现有的实现
4. 实现模型架构
5. 实现模型测试
6. 提交拉取请求
7. （可选）构建演示并与世界分享

### 1.-3. 准备你的模型贡献

**1. 选择要转换的模型**

让我们从基础知识开始：你需要了解的第一件事是你想要转换的架构。如果你还没有决定要使用的特定架构，向🤗Transformers团队寻求建议是最大化你的影响力的好方法 - 我们将指导你选择缺少TensorFlow支持的最重要的架构。如果你想要与TensorFlow一起使用的特定模型已经在🤗Transformers中具有TensorFlow架构实现，但缺少权重，请随时直接转到本页的[权重转换部分](#adding-tensorflow-weights-to-hub)。

为了简单起见，本指南的其余部分假设你已经决定为*BrandNewBert*贡献TensorFlow版本（与[添加新模型的指南](add_new_model.md)中的示例相同）。

<Tip>

在开始TensorFlow模型架构的工作之前，请再次确认当前没有对此进行的尝试。
你可以在[拉取请求GitHub页面](https://github.com/huggingface/transformers/pulls?q=is%3Apr)上搜索“BrandNewBert”以确认没有与TensorFlow相关的拉取请求。

</Tip>


**2. 准备transformers的开发环境**

选择了模型架构后，打开一个草稿PR来表示你打算使用该模型进行工作。按照下面的说明设置你的环境并打开草稿PR。

1. 点击仓库页面上的“Fork”按钮，将[repository](https://github.com/huggingface/transformers)fork到你的GitHub帐户下，这将在你的GitHub用户帐户下创建代码的副本。

2. 将你的`transformers`分支克隆到本地磁盘，并将基本仓库添加为远程分支:

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

3. 设置开发环境，例如运行以下命令:

```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```

根据你的操作系统以及由于Transformers的可选依赖项数量的增加，你可能会在此命令中遇到故障。如果出现这种情况，请确保安装了TensorFlow，然后执行以下操作:

```bash
pip install -e ".[quality]"
```

**注意：**你不需要安装CUDA。使新模型在CPU上工作即可。

4. 从你的主分支分叉一个具有描述性名称的分支

```bash
git checkout -b add_tf_brand_new_bert
```

5. 拉取并合并到当前主分支

```bash
git fetch upstream
git rebase upstream/main
```

6. 在`transformers/src/models/brandnewbert/`目录中添加一个名为`modeling_tf_brandnewbert.py`的空`.py`文件。这将是你的TensorFlow模型文件。

7. 使用以下命令将更改推送到你的帐户:

```bash
git add .
git commit -m "initial commit"
git push -u origin add_tf_brand_new_bert
```

8. 确定满意之后，转到GitHub上你分支的Web页面。单击“Pull request”。确保将Hugging Face团队的GitHub帐户添加为评审人，这样Hugging Face团队就会在今后有变更时收到通知。

9. 通过单击GitHub拉取请求Web页面右侧的“Convert to draft”将PR转换为草稿。

现在，你已经设置了⚒️🏗一个开发环境，以在🤗Transformers中将*BrandNewBert*转换为TensorFlow。


**3. (可选) 理解理论方面和现有的实现**

你应该花一些时间阅读*BrandNewBert*的论文，如果存在这样的描述性工作的话。可能有一些论文中难以理解的大段内容。如果是这样，不要担心，这没关系！目标不是深入理论地理解论文，而是提取在使用TensorFlow有效重新实现🤗Transformers中所需的必要信息。话虽如此，在这个阶段你不需要对模型的所有方面都有深入的理解是完全可以的。无论如何，我们强烈鼓励你在我们的[论坛](https://discuss.huggingface.co/)上解决任何紧迫的问题。


当你理解了要实现的模型的基础知识之后，重要的是要理解现有的实现。这是一个很好的机会来确认工作中的实现与你对该模型的期望是否一致，以及预见TensorFlow方面的技术挑战。

你可能会对自己刚刚吸收的大量信息感到不知所措，这是完全自然的。在这个阶段要求你了解模型的所有方面绝对不是一个要求。尽管如此，我们强烈鼓励你在我们的[论坛](https://discuss.huggingface.co/)上解决任何紧迫的问题。


### 4. 模型实现

现在是时候开始编码了。我们建议从PyTorch文件本身开始：将`modeling_brand_new_bert.py`中的内容复制到`src/transformers/models/brand_new_bert/`中的`modeling_tf_brand_new_bert.py`中。本节的目标是修改文件并更新🤗Transformers的导入结构，以便你可以成功导入`TFBrandNewBert`和`TFBrandNewBert.from_pretrained(model_repo, from_pt=True)`，并加载一个可用的TensorFlow *BrandNewBert*模型。

遗憾的是，将PyTorch模型转换为TensorFlow没有固定的步骤。但是，你可以遵循我们的建议，使此过程尽可能顺利：
- 在所有类的名称前加上“TF”（例如，`BrandNewBert`变为`TFBrandNewBert`）。
- 大多数PyTorch操作都有对应的TensorFlow替代品。例如，`torch.nn.Linear`对应`tf.keras.layers.Dense`，`torch.nn.Dropout`对应`tf.keras.layers.Dropout`，等等。如果你对特定操作不确定，可以使用[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf)或[PyTorch文档](https://pytorch.org/docs/stable/)。
- 查找🤗Transformers代码库中的模式。如果遇到没有直接替代的某个操作，那么有可能其他人已经遇到过同样的问题。
- 默认情况下，保持与PyTorch相同的变量名称和结构。这将使调试、跟踪问题和进行后续修复更加容易。
- 某些层在每个框架中的默认值不同。一个著名的例子是批归一化层的epsilon值（在[PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)中为1e-5，在[TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)中为1e-3）。请仔细检查文档！
- PyTorch的`nn.Parameter`变量通常需要在TensorFlow Layer的`build()`内进行初始化。请参见以下示例：[PyTorch](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_vit_mae.py#L212) / [TensorFlow](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L220)
- 如果PyTorch模型在函数的顶部有一个`#copied from ...`，那么你的TensorFlow模型也可以从其复制的架构中借用该函数，前提是它具有TensorFlow架构。
- 在TensorFlow函数中正确设置`name`属性的逻辑对于执行`from_pt=True`的权重交叉装载非常重要。`name`几乎总是PyTorch代码中相应变量的名称。如果未正确设置`name`，在加载模型权重时可以在错误消息中看到它。
- 实际上，基础模型类`BrandNewBertModel`的逻辑将位于`TFBrandNewBertMainLayer`中，这是一个Keras层子类（[示例](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L719)）。`TFBrandNewBertModel`将仅是此层周围的一个包装器。
- 为了加载预训练的权重，需要构建Keras模型。因此，`TFBrandNewBertPreTrainedModel`将需要保存一个模型输入示例，即`dummy_inputs`（[示例](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L916)）。
- 如果遇到困难，请寻求帮助 - 我们在这里帮助你！🤗

除了模型文件本身之外，你还需要为模型类和相关文档页面添加指针。你可以完全按照其他PR中的模式完成此部分
([示例](https://github.com/huggingface/transformers/pull/18020/files))。以下是所需的手动更改列表：
- 将*BrandNewBert*的所有公共类包含在`src/transformers/__init__.py`中
- 将*BrandNewBert*类添加到`src/transformers/models/auto/modeling_tf_auto.py`中相应的自动类中
- 在`src/transformers/utils/dummy_tf_objects.py`中添加与*BrandNewBert*相关的延迟加载类
- 更新`src/transformers/models/brand_new_bert/__init__.py`中的公共类的导入结构
- 在`docs/source/en/model_doc/brand_new_bert.md`中为*BrandNewBert*的公共方法添加文档指针
- 在`docs/source/en/model_doc/brand_new_bert.md`中将你自己添加到*BrandNewBert*的贡献者列表中
- 最后，在`docs/source/en/index.md`中的*BrandNewBert*的TensorFlow列上添加一个绿色的✅

当你对你的实现感到满意时，请运行以下清单以确认你的模型架构已准备就绪：
1. 所有在训练时行为不同的层（例如，Dropout）都使用`training`参数调用，该参数从顶层类传播
2. 每当可能时，你使用了`#copied from ...`
3. `TFBrandNewBertMainLayer`和所有使用它的类的`call`函数都使用`@unpack_inputs`修饰
4. 使用`@keras_serializable`修饰`TFBrandNewBertMainLayer`
5. 可以使用`TFBrandNewBert.from_pretrained(model_repo, from_pt=True)`从PyTorch权重加载TensorFlow模型
6. 你可以使用预期的输入格式调用TensorFlow模型


### 5. 添加模型测试

太棒了，你已经实现了一个TensorFlow模型！现在是时候添加测试来确保你的模型表现如预期了。和前一节一样，我们建议你先将`tests/models/brand_new_bert/`目录中的`test_modeling_brand_new_bert.py`文件复制到`test_modeling_tf_brand_new_bert.py`中，并继续进行必要的TensorFlow替换。现在，在所有的`.from_pretrained()`调用中，你应该使用`from_pt=True`标志来加载现有的PyTorch权重。

完成后，现在是真相的时刻：运行测试！ 😬

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

最有可能的结果是你会看到一堆错误。别担心，这是预期的！调试机器学习模型是非常困难的，成功的关键在于耐心（和`breakpoint()`）。根据我们的经验，最困难的问题往往是由于机器学习框架之间的细微不匹配造成的，对此我们在本指南的结尾给出了一些指点。在其他情况下，一个常规的测试可能不能直接适用于你的模型，在这种情况下，我们建议在模型测试类级别上覆盖。无论出现什么问题，如果你卡住了，请不要犹豫在你的草稿拉取请求中寻求帮助。

当所有的测试都通过了，恭喜你，你的模型离被添加到🤗Transformers库中已经很接近了！ 🎉

### 6.-7. 确保每个人都能使用你的模型

**6. 提交拉取请求**

当你完成了实现和测试后，是时候提交一个拉取请求了。在推送代码之前，请运行我们的代码格式化工具`make fixup` 🪄。这将自动修复任何格式化问题，否则会导致我们的自动检查失败。

现在是时候将你的草稿拉取请求转换为一个真正的拉取请求了。为了这样做，请点击“准备好审核”按钮，并将Joao (`@gante`)和Matt (`@Rocketknight1`)添加为审阅者。一个模型的拉取请求需要至少3个审阅者，但他们会负责为你的模型找到合适的额外审阅者。

在所有的审阅者对你的PR满意后，最后一个行动点是在`.from_pretrained()`调用中删除`from_pt=True`标志。由于没有TensorFlow的权重，你将需要添加它们！请查看下面的说明以了解如何添加。

最后，当TensorFlow的权重合并后，你获得至少3个审阅者的批准，并且所有的CI检查都通过了，请再次在本地检查测试

```bash
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

然后我们将合并你的PR！祝贺你达到了这个里程碑 🎉

**7.（可选）构建演示并与全世界分享**

开源软件最困难的部分之一是它的发现。其他用户如何了解你出色的TensorFlow贡献的存在呢？当然是通过适当的交流！ 📣

与社区分享模型有两种主要方式：
- 构建演示。这些包括Gradio演示、笔记本和其他有趣的方式来展示你的模型。我们极力鼓励你将笔记本添加到我们的[社区驱动的演示](https://huggingface.co/docs/transformers/community)中。
- 在Twitter和LinkedIn等社交媒体上分享故事。你应该为自己的工作感到自豪，并与社区分享你的成就-你的模型现在可以被全世界数千名工程师和研究人员使用了 🌍！我们将很高兴转发你的帖子，并帮助你与社区分享你的工作。


## 将TensorFlow权重添加到🤗Hub

假设TensorFlow模型架构在🤗Transformers中可用，将PyTorch权重转换为TensorFlow权重非常容易！

做法如下：
1. 确保你已经在终端中登录了你的Hugging Face账户。你可以使用命令`huggingface-cli login`登录（你可以在[这里](https://huggingface.co/settings/tokens)找到你的访问token）
2. 运行`transformers-cli pt-to-tf --model-name foo/bar`，其中`foo/bar`是包含你想要转换的PyTorch权重的模型仓库的名称
3. 在🤗Hub中创建的PR中使用`@joaogante`和`@Rocketknight1`标记命令以上命令

就是这样！ 🎉


## 调试跨机器学习框架的不匹配问题 🐛

在添加新的架构或为现有的架构创建TensorFlow权重时，你可能会遇到关于PyTorch和TensorFlow之间的不匹配的错误。你甚至可能决定打开两个框架的模型架构代码，发现它们看起来是相同的。到底是怎么回事？ 🤔

首先，让我们谈谈为什么理解这些不匹配很重要。很多社区成员将直接使用🤗Transformers模型，并相信我们的模型表现如预期。当两个框架之间存在很大的不匹配时，意味着模型对于至少一个框架来说没有遵循参考实现。这可能导致悄无声息的失败，即模型运行但性能不佳。这可能比根本无法运行的模型还要糟糕！为此，我们的目标是在模型的所有阶段中，框架之间的不匹配小于`1e-5`。

就像其他数值问题一样，细节决定成败。而既然如此注重细节的工艺，秘密的关键就是耐心。以下是我们建议处理此类问题的工作流程：
1. 找出不匹配的源头。你正在转换的模型可能在某个点上有几乎相同的内部变量。在两个框架的架构中放置`breakpoint()`语句，并以自顶向下的方式比较这些数值变量的值，直到找到问题的源头。
2. 现在你已经定位了问题的源头，请与🤗Transformers团队联系。我们可能之前见过类似的问题，并且可以迅速提供解决方案。作为备选方案，浏览像StackOverflow和GitHub问题这样的热门页面。
3. 如果没有任何解决方案，那就意味着你必须更深入地去研究了。好消息是你已经找到了问题的关键，所以你可以专注于有问题的指令，将其抽象出来，忽略掉模型的其他部分！坏消息是你将不得不深入到该指令的源代码中。在某些情况下，你可能会发现一个参考实现存在问题-请勇于在上游存储库中开启一个问题。

在某些情况下，在与🤗Transformers团队讨论后，我们可能会发现修复不匹配是不可行的。当输出层中的不匹配非常小（但在隐藏状态中可能很大）时，我们可能会决定忽略它，以便分发该模型。上面提到的`pt-to-tf`命令行工具有一个`--max-error`标志，在转换权重时可以覆盖错误消息。