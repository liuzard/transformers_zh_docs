<!--版权所有2022年致人可拥抱团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），您除非符合许可证的规定否则不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证副本

除非适用法律要求或者书面同意，否则以"AS IS"的基础分发软件，没有任何形式的保证或条件，无论是明示的还是暗示的。详细资料请参阅许可证的规定。

⚠️ 请注意，此文件是Markdown格式的，但包含特定语法（类似于MDX）以供我们的文档构建器使用，可能不会在您的Markdown查看器中正确显示。

-->

# ERNIE

## 概述
ERNIE是百度提出的一系列强大的模型，尤其在中文任务中表现优秀，
包括[ERNIE1.0](https://arxiv.org/abs/1904.09223)，[ERNIE2.0](https://ojs.aaai.org/index.php/AAAI/article/view/6428)，
[ERNIE3.0](https://arxiv.org/abs/2107.02137)，[ERNIE-Gram](https://arxiv.org/abs/2010.12148)，[ERNIE-health](https://arxiv.org/abs/2110.07244)等。

这些模型由[nghuyong](https://huggingface.co/nghuyong)贡献，官方代码可在[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)（在PaddlePaddle中）找到。

### 如何使用
以`ernie-1.0-base-zh`为例：

```Python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
model = AutoModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
```

### 支持的模型

|     模型名称      | 语言 |          描述          |
|:-----------------:|:----:|:----------------------:|
|  ernie-1.0-base-zh | 中文 | 层数：12，头数：12，隐藏层：768 |
|  ernie-2.0-base-en | 英文 | 层数：12，头数：12，隐藏层：768 |
| ernie-2.0-large-en | 英文 | 层数：24，头数：16，隐藏层：1024 |
|  ernie-3.0-base-zh | 中文 | 层数：12，头数：12，隐藏层：768 |
| ernie-3.0-medium-zh | 中文 | 层数：6，头数：12，隐藏层：768  |
|  ernie-3.0-mini-zh  | 中文 | 层数：6，头数：12，隐藏层：384  |
| ernie-3.0-micro-zh  | 中文 | 层数：4，头数：12，隐藏层：384  |
|  ernie-3.0-nano-zh  | 中文 | 层数：4，头数：12，隐藏层：312  |
|   ernie-health-zh   | 中文 | 层数：12，头数：12，隐藏层：768 |
|    ernie-gram-zh    | 中文 | 层数：12，头数：12，隐藏层：768 |

您可以在huggingface的模型中心找到所有支持的模型：[huggingface.co/nghuyong](https://huggingface.co/nghuyong)，模型详细信息可在paddle的官方存储库中找到：
[PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html)
和[ERNIE](https://github.com/PaddlePaddle/ERNIE/blob/repro)。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## Ernie配置

[[autodoc]] ErnieConfig
    - all

## Ernie特定输出

[[autodoc]] models.ernie.modeling_ernie.ErnieForPreTrainingOutput

## ErnieModel

[[autodoc]] ErnieModel
    - forward

## ErnieForPreTraining

[[autodoc]] ErnieForPreTraining
    - forward

## ErnieForCausalLM

[[autodoc]] ErnieForCausalLM
    - forward

## ErnieForMaskedLM

[[autodoc]] ErnieForMaskedLM
    - forward

## ErnieForNextSentencePrediction

[[autodoc]] ErnieForNextSentencePrediction
    - forward

## ErnieForSequenceClassification

[[autodoc]] ErnieForSequenceClassification
    - forward

## ErnieForMultipleChoice

[[autodoc]] ErnieForMultipleChoice
    - forward

## ErnieForTokenClassification

[[autodoc]] ErnieForTokenClassification
    - forward

## ErnieForQuestionAnswering

[[autodoc]] ErnieForQuestionAnswering
    - forward