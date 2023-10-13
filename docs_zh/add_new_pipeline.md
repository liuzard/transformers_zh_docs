<!--版权所有2020年The HuggingFace团队。 保留所有权利。

根据Apache许可证，版本2.0（“许可”）授权;除非你遵守许可，否则你不得使用此文件
许可证。 你可以在以下位置获取许可的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则在许可下分发的软件是根据
“原样” BASIS，不附带任何形式的保证或条件，无论是明示的还是暗示的。有关许可的条件

⚠️请注意，此文件采用Markdown格式，但包含专用于doc-builder（类似于MDX）的特定语法，可能无法

在你的Markdown查看器中正确渲染。-->

# 如何创建自定义pipeline？

在本指南中，我们将看到如何创建自定义pipeline并将其与[Hub](hf.co/models)共享或添加到
🤗Transform库。

首先，你需要确定流水线能够接受的原始输入。它可以是字符串，原始字节，
字典或其他看起来最可能的所需输入。尽量保持这些输入尽可能简单
因为它使兼容性更容易（甚至通过JSON在其他语言中实现）。这些将是
流水线（`preprocess`）的“inputs”。

然后定义`outputs`。与`inputs`原则相同。简化处理越好。这些将是
`postprocess`方法的输出。

首先从继承基类`Pipeline`开始，该基类包含了实现`preprocess`，`postprocess`和两个辅助方法`_forward`和`_sanitize_parameters`所需的四个方法。

```Python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

标准的分解结构支持相对平滑的CPU / GPU支持，同时支持在不同线程的CPU上进行预处理/后处理

`preprocess`将采用最初定义的输入，并将其转换为可以供模型喂食的内容。它可能
包含更多信息，通常是一个字典。

`_forward`是实现细节，不应直接调用。`forward`是首选
调用方法，因为它包含了确保所有内容都在预期设备上工作的保护措施。如果任何东西是
与真实模型关联的，则属于`_forward`方法中，其他任何内容都在preprocess /postprocess中。

`postprocess`方法将采用`_forward`的输出，并将其转换为之前决定的最终输出。

`_sanitize_parameters`存在的目的是允许用户随时传递任何参数，无论是在初始化时`pipeline(...., maybe_arg=4)`还是在调用时`pipe = pipeline(...); output = pipe(...., maybe_arg=4)`。

`_sanitize_parameters`的返回值是3个kwargs字典，将直接传递给`preprocess`，`_forward`和`postprocess`。如果调用者没有用任何额外参数调用，请不要填写任何内容。这
可以在函数定义中保留默认参数，这总是更加“自然”。

在分类任务的后处理中，典型的示例是`top_k`参数。

```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

为了实现这一点，我们将使用一个默认参数`5`更新我们的`postprocess`方法，并编辑
`_sanitize_parameters`以允许此新参数。

```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # 添加处理top_k的逻辑
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
    
    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

尽量保持输入/输出非常简单，最好是JSON序列化，因为这样可以很容易地使用pipeline而无需用户了解新类型的对象。为了方便使用（音频文件，可以是文件名，URL或纯字节），
支持许多不同类型的参数相对较常见。



## 将其添加到受支持任务的列表中

要将`new-task`注册到受支持任务列表中，你需要将其添加到`PIPELINE_REGISTRY`中：

```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

如果需要，你可以指定默认模型，此时模型应该具有特定的修订版（可以是分支名称或提交哈希），以及类型：

```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # 当前支持类型：text、audio、image、multimodal
)
```

## 在Hub上共享你的pipeline

要在Hub上共享你的自定义pipeline，你只需要将`Pipeline`子类的自定义代码保存在一个
Python文件中。例如，假设我们想为句子对分类使用自定义pipeline，如下所示：

```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

该实现与框架无关，适用于PyTorch和TensorFlow模型。如果我们将其保存在
名为`pair_classification.py`的文件中，然后可以导入并像这样注册它：

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```

注册完成后，我们可以使用预训练模型使用它。例如，`sgugger/finetuned-bert-mrpc`已经
在MRPC数据集上进行了微调，用于将句子对分类为是否是释义复述。

```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

然后，我们可以使用`save_pretrained`方法将其共享到Hub中使用`Repository`：

```py
from huggingface_hub import Repository

repo = Repository("test-dynamic-pipeline", clone_from="{your_username}/test-dynamic-pipeline")
classifier.save_pretrained("test-dynamic-pipeline")
repo.push_to_hub()
```

这将把定义`PairClassificationPipeline`的文件复制到文件夹“test-dynamic-pipeline”中，
并将流水线的模型和分词处理器保存起来，然后将所有内容推送到存储库
`{your_username}/test-dynamic-pipeline`。之后，任何人只要提供选项`trust_remote_code=True`就可以使用它：

```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```

## 将pipeline添加到🤗Transformers

如果想要将pipeline贡献给🤗Transformers，你需要在`pipelines`模块中的`pipelines`子模块中添加一个新模块，其中包含你的流水线代码，并将其添加到`pipelines/__init__`中定义的任务列表中。

然后需要添加测试。创建一个名为`tests/test_pipelines_MY_PIPELINE.py`的新文件，并包含其他测试的示例。

`run_pipeline_test`函数将非常通用，并在`model_mapping`和`tf_model_mapping`定义的每种可能的架构上运行，这是非常重要的。

这对于将来的兼容性测试非常重要，也就是说，如果有人为
`XXXForQuestionAnswering`添加了一个新模型，那么pipeline测试将尝试在新模型上运行。因为模型是随机的，所以无法检查实际值，这就是为什么有一个帮助器`ANY`，它将尝试匹配pipeline类型的输出。

你还*需要*编写2（理想情况下是4）个测试。

- `test_small_model_pt`：为该pipeline定义一个小型模型（结果无所谓），并测试pipeline的输出。结果应与`test_small_model_tf`相同。
- `test_small_model_tf`：为该pipeline定义一个小型模型（结果无所谓），并测试pipeline的输出。结果应与`test_small_model_pt`相同。
- `test_large_model_pt`（`可选`）：在一个真正的流水线上测试pipeline，其中结果应该是有意义的。这些测试很慢，应标记为这样。这里的目标是展示pipeline，并确保在未来的发布中没有漂移。
- `test_large_model_tf`（`可选`）：在一个真正的流水线上测试pipeline，其中结果应该是有意义的。这些测试很慢，应标记为这样。这里的目标是展示pipeline，并确保在未来的发布中没有漂移。
