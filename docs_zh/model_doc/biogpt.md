版权所有2022 HuggingFace团队。

根据Apache许可证第2.0版（"许可证"），你除了遵守许可证之外，不得使用此文件。你可以在以下网址获取许可证副本：

http：//www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于"AS IS" BASIS分发的，不附带任何担保或条件。请参阅许可证以了解特定语言下的权利和限制。

⚠️ 请注意，此文件使用Markdown格式，但包含我们文档生成器的特定语法（类似于MDX），可能无法在Markdown查看器中正确显示。

---

# BioGPT

## 概述

BioGPT模型是由Renqian Luo，Liai Sun，Yingce Xia，Tao Qin，Sheng Zhang，Hoifung Poon和Tie-Yan Liu在[《BioGPT：用于生物医学文本生成和挖掘的生成式预训练转换器》](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9)中提出的。BioGPT是一种领域特定的生成式预训练转换器语言模型，用于生物医学文本生成和挖掘。BioGPT遵循Transformer语言模型骨干结构，并在15M篇PubMed摘要上从头开始进行预训练。

来自论文的摘要如下：

*受到在一般自然语言领域取得巨大成功的启发，预训练语言模型在生物医学领域引起了越来越多的关注。在一般语言领域的两个主要预训练语言模型分支，即BERT（及其变体）和GPT（及其变体）中，第一个在生物医学领域得到了广泛研究，如BioBERT和PubMedBERT。尽管它们在各种鉴别性下游生物医学任务上取得了巨大成功，但生成能力的缺失限制了它们的应用范围。在本文中，我们提出了BioGPT，这是一个在大规模生物医学文献上进行预训练的领域特定生成转换器语言模型。我们评估了BioGPT在六个生物医学自然语言处理任务上的表现，并证明我们的模型在大多数任务上优于之前的模型。特别地，我们在BC5CDR、KD-DTI和DDI端到端关系抽取任务上分别获得了44.98%、38.42%和40.76%的F1分数，以及在PubMedQA上的78.2%的准确率，创造了一个新纪录。我们对文本生成的案例研究进一步证明了BioGPT在生物医学文献中生成流畅的生物医学术语描述的优势。*

提示：

- BioGPT是一个具有绝对位置嵌入的模型，因此通常建议在右侧填充输入而不是左侧。
- BioGPT是使用因果语言建模（CLM）目标进行训练的，因此在预测序列中的下一个标记时非常强大。利用这一特性，BioGPT可以生成句法连贯的文本，可以在run_generation.py示例脚本中观察到。
- 模型可以接受`past_key_values`（对于PyTorch）作为输入，这是先前计算的键/值注意力对。使用此（past_key_values或past）值可防止模型在文本生成的上下文中重新计算预计算的值。对于PyTorch，请参阅BioGptForCausalLM.forward（）方法的past_key_values参数，以获取有关其使用的更多信息。

此模型由[kamalkraj]（https://huggingface.co/kamalkraj）贡献。原始代码可以在[此处](https://github.com/microsoft/BioGPT)找到。

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)

## BioGptConfig

[[autodoc]] BioGptConfig


## BioGptTokenizer

[[autodoc]] BioGptTokenizer
    - save_vocabulary


## BioGptModel

[[autodoc]] BioGptModel
    - forward


## BioGptForCausalLM

[[autodoc]] BioGptForCausalLM
    - forward

    
## BioGptForTokenClassification

[[autodoc]] BioGptForTokenClassification
    - forward


## BioGptForSequenceClassification

[[autodoc]] BioGptForSequenceClassification
    - forward