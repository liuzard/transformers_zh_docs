<!-- 版权 2023 年 HuggingFace 团队保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证，否则不得使用此文件。
你可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律有要求或书面同意，软件根据许可证在“现状”下分发，
无论明示或暗示的，不提供任何形式的保证或条件。
有关许可证下的特定语言的权限和限制，请查看许可证。

⚠️ 请注意，此文件采用 Markdown 格式，但包含我们的 doc-builder（类似 MDX 的语法）的特定语法，可能在你的 Markdown 查看器中无法正确呈现。

-->

# X-MOD

## 概述

Jonas Pfeiffer，Naman Goyal，Xi Lin，Xian Li，James Cross，Sebastian Riedel 和 Mikel Artetxe 在 [Lifting the Curse of Multilinguality by Pre-training Modular Transformers](http://dx.doi.org/10.18653/v1/2022.naacl-main.255) 中提出了 X-MOD 模型。
X-MOD 在预训练过程中扩展了多语言遮蔽语言模型（例如 [XLM-R](xlm-roberta)），以包含语言特定的模块化组件（_语言适配器_）。在微调过程中，每个 Transformer 层中的语言适配器将被冻结。

以下是论文的摘要：

*已知多语言预训练模型遭受多语言诅咒，导致其覆盖更多语言时每个语言的性能下降。我们通过引入语言特定的模块来解决此问题，该模块使我们能够增加模型的总容量，同时保持每个语言的可训练参数总数恒定。与后续学习语言特定组件的先前工作不同，我们从一开始就对我们的交叉语言模块 (X-MOD) 模型的模块进行预训练。我们对自然语言推理、命名实体识别和问题回答的实验表明，我们的方法不仅减轻了语言间的负面干扰，还能实现积极的传输，从而改善了单语和跨语言性能。此外，我们的方法可以在事后添加语言而没有可衡量的性能下降，不再将模型使用限制在预训练语言集之内。*

提示：
- X-MOD 类似于 [XLM-R](xlm-roberta)，但不同之处在于需要指定输入语言，以便激活正确的语言适配器。
- 主要的模型（base 和 large）拥有 81 种语言的适配器。

此模型由 [jvamvas](https://huggingface.co/jvamvas) 贡献。
原始代码可以在 [此处](https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/fairseq/models/xmod) 找到，原始文档在 [此处](https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/examples/xmod) 找到。

## 适配器用法

### 输入语言

有两种指定输入语言的方式：
1. 在使用模型之前设置默认语言：

```python
from transformers import XmodModel

model = XmodModel.from_pretrained("facebook/xmod-base")
model.set_default_language("en_XX")
```

2. 对于每个样本，通过显式传递对应的语言适配器的索引来指定输入语言：

```python
import torch

input_ids = torch.tensor(
    [
        [0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2],
        [0, 1310, 49083, 443, 269, 71, 5486, 165, 60429, 660, 23, 2],
    ]
)
lang_ids = torch.LongTensor(
    [
        0,  # en_XX
        8,  # de_DE
    ]
)
output = model(input_ids, lang_ids=lang_ids)
```

### 微调
论文建议在微调过程中冻结嵌入层和语言适配器。提供了一种进行此操作的方法：

```python
model.freeze_embeddings_and_language_adapters()
# 对模型进行微调...
```

### 跨语言转移
在微调后，可以通过激活目标语言的语言适配器来测试零样本跨语言转移：

```python
model.set_default_language("de_DE")
# 对德语示例进行模型评估...
```

## 资源

- [文本分类任务指南](../tasks/sequence_classification)
- [分词分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [遮蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## XmodConfig

[[autodoc]] XmodConfig

## XmodModel

[[autodoc]] XmodModel
    - forward

## XmodForCausalLM

[[autodoc]] XmodForCausalLM
    - forward

## XmodForMaskedLM

[[autodoc]] XmodForMaskedLM
    - forward

## XmodForSequenceClassification

[[autodoc]] XmodForSequenceClassification
    - forward

## XmodForMultipleChoice

[[autodoc]] XmodForMultipleChoice
    - forward

## XmodForTokenClassification

[[autodoc]] XmodForTokenClassification
    - forward

## XmodForQuestionAnswering

[[autodoc]] XmodForQuestionAnswering
    - forward