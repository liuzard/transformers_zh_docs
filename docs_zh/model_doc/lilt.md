<!--版权所有2022年The HuggingFace团队。

根据Apache许可证第2.0版（“许可证”），除非符合许可证，否则不得使用此文件。您可以在下方链接中获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件基于“按原样”分发，不附带任何明示或暗示的保证或条件。请参阅许可证以获取具体的语言权限和限制。

⚠注意，此文件是Markdown格式的，但包含特定于我们的文档构建器的语法（类似于MDX），您的Markdown查看器可能无法正常显示。-->

# LiLT

## 概述

LiLT模型由Jiapeng Wang、Lianwen Jin和Kai Ding在《LiLT：一种简单而有效的面向结构化文档理解的语言无关布局变换器》（https://arxiv.org/abs/2202.13669）中提出。
LiLT允许将任何预训练的RoBERTa文本编码器与轻量级布局变换器相结合，从而实现类似于[LayoutLM](layoutlm)的多语言文档理解。

论文中的摘要如下：

*近来，结构化文档理解因其在智能文档处理中的关键作用而受到广泛关注并取得了显著进展。然而，大多数现有的相关模型只能处理特定语言（通常是英语）的文档数据，这是极为有限的。为了解决这个问题，我们提出了一种简单而有效的面向语言无关的布局变换器（LiLT）用于结构化文档理解。LiLT可以在单一语言的结构化文档上进行预训练，然后使用相应的现成单语/多语文本预训练模型进行直接微调其他语言。在八种语言上的实验结果表明，LiLT在各种广泛使用的下游基准任务上可以达到具有竞争力甚至更好的性能，这使得可以从文档布局结构的预训练中获得独立于语言的益处。*

提示：

- 要将语言无关布局变换器与[hug](https://huggingface.co/models?search=roberta)中的新RoBERTa检查点相结合，请参考[此指南](https://github.com/jpWang/LiLT#or-generate-your-own-checkpoint-optional)。
该脚本将会将`config.json`和`pytorch_model.bin`文件存储在本地。在完成此操作后，可以执行以下操作（假设您正在使用您的HuggingFace账户登录）：

```
from transformers import LiltModel

model = LiltModel.from_pretrained("path_to_your_files")
model.push_to_hub("name_of_repo_on_the_hub")
```

- 在为模型准备数据时，请确保使用与您所组合的布局变换器相对应的标记词汇。
- 由于[lilt-roberta-zh-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base)使用与[LayoutLMv3](layoutlmv3)相同的词汇表，因此可以使用[`LayoutLMv3TokenizerFast`]来为模型准备数据。
对于[lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base)也是如此：可以使用[`LayoutXLMTokenizerFast`]来为该模型准备数据。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/lilt_architecture.jpg"
alt="drawing" width="600"/>

<small> LiLT架构。来源于<a href="https://arxiv.org/abs/2202.13669">原始论文</a>。 </small>

本模型由[nielsr](https://huggingface.co/nielsr)贡献。
原始代码可在[此处](https://github.com/jpwang/lilt)找到。

## 资源

以下是官方Hugging Face资源和社区（标有🌎）资源的列表，可帮助您开始使用LiLT。

- LiLT的演示笔记本可在[此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LiLT)找到。

**文档资源**
- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)

如果您有兴趣提交资源以包含在这里，请随意打开Pull Request，我们会进行审核！该资源应该展示出新的东西，而不是重复现有资源。

## LiltConfig

[[autodoc]] LiltConfig

## LiltModel

[[autodoc]] LiltModel
    - forward

## LiltForSequenceClassification

[[autodoc]] LiltForSequenceClassification
    - forward

## LiltForTokenClassification

[[autodoc]] LiltForTokenClassification
    - forward

## LiltForQuestionAnswering

[[autodoc]] LiltForQuestionAnswering
    - forward