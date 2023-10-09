<!--
版权所有 (c) 2021 NVIDIA公司和HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”），除非符合许可证的要求，
否则不得使用此文件。你可以在下面网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证发布的软件是基于
“按原样”提供的，“没有任何担保或条件”，不论是明示或暗示的。
有关许可证的详细信息，请参阅许可证下的规定。

⚠️请注意，此文件是Markdown格式，但包含我们文档生成器的特定语法（类似于MDX），可能在你的Markdown查看器中无法正确渲染。

-->

# MegatronBERT

## 概述

Mohammad Shoeybi、Mostofa Patwary、Raul Puri、Patrick LeGresley、Jared Casper和Bryan Catanzaro在论文[使用模型并行训练百亿参数语言模型的Megatron-LM](https://arxiv.org/abs/1909.08053)中提出了MegatronBERT模型。

论文中的摘要如下所示：
*最近在语言建模方面的工作表明，训练大型的Transformer模型推进了自然语言处理应用的发展。然而，由于内存限制，训练非常大的模型可能非常困难。在这项工作中，我们提出了训练非常大的Transformer模型的技术，并实现了一种简单高效的内置模型并行方法，可以训练具有数十亿参数的Transformer模型。我们的方法不需要新的编译器或库更改，与pipeline模型并行性正交和互补，并且可以通过在本机PyTorch中插入少量通信操作来完全实现。我们通过使用512个GPU将基于Transformer的模型融合到83亿参数。与维持39TeraFLOPs的强大单个GPU基线相比，我们在整个应用程序上保持15.1PetaFLOPs的扩展效率为76％，这是峰值FLOPs的30％。为了证明大型语言模型可以进一步推动技术的发展（SOTA），我们训练了一个类似于GPT-2的有83亿参数的Transformer语言模型和一个类似于BERT的有39亿参数的模型。我们表明，在BERT模型中，对于层归一化的放置要仔细注意，因为随着模型大小的增长，这对于实现性能的提高至关重要。使用GPT-2模型，我们在WikiText103（相对于SOTA困惑度的10.8与15.8）和LAMBADA（相对于SOTA精度63.2％的66.5％）数据集上取得了SOTA结果。我们的BERT模型在RACE数据集（相对于SOTA精度89.4％的90.9％）上取得了SOTA结果。*

提示：
我们已提供预训练的[BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m)检查点，用于评估或微调下游任务。

要访问这些检查点，请首先[注册](https://ngc.nvidia.com/signup)并设置NVIDIA GPU云（NGC）注册表CLI。下载模型的更多文档可以在[NGC文档](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1)中找到。

或者，你可以直接使用以下命令下载检查点：

BERT-345M-uncased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip
-O megatron_bert_345m_v0_1_uncased.zip
```

BERT-345M-cased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O
megatron_bert_345m_v0_1_cased.zip
```

一旦你从NVIDIA GPU Cloud（NGC）获得了检查点，你需要将它们转换为Hugging Face Transformers和我们的BERT代码端口可以轻松加载的格式。

以下命令允许你进行转换。我们假设文件夹`models/megatron_bert`包含`megatron_bert_345m_v0_1_{cased, uncased}.zip`，并且这些命令是在该文件夹内运行的：

```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_uncased.zip
```

```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_cased.zip
```

此模型由[jdemouth](https://huggingface.co/jdemouth)贡献。原始代码可以在[这里](https://github.com/NVIDIA/Megatron-LM)找到。该存储库包含了一种"张量并行"和"pipeline并行"技术的混合模型并行方法的多GPU和多节点实现。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## MegatronBertConfig

[[autodoc]] MegatronBertConfig

## MegatronBertModel

[[autodoc]] MegatronBertModel
    - forward

## MegatronBertForMaskedLM

[[autodoc]] MegatronBertForMaskedLM
    - forward

## MegatronBertForCausalLM

[[autodoc]] MegatronBertForCausalLM
    - forward

## MegatronBertForNextSentencePrediction

[[autodoc]] MegatronBertForNextSentencePrediction
    - forward

## MegatronBertForPreTraining

[[autodoc]] MegatronBertForPreTraining
    - forward

## MegatronBertForSequenceClassification

[[autodoc]] MegatronBertForSequenceClassification
    - forward

## MegatronBertForMultipleChoice

[[autodoc]] MegatronBertForMultipleChoice
    - forward

## MegatronBertForTokenClassification

[[autodoc]] MegatronBertForTokenClassification
    - forward

## MegatronBertForQuestionAnswering

[[autodoc]] MegatronBertForQuestionAnswering
    - forward
-->