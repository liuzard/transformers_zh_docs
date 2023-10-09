<!--
版权所有2020年的HuggingFace团队保留。

根据Apache许可证，版本2.0（“许可证”）进行许可; 除非符合许可证的规定，
否则您不能使用此文件。您可以在以下位置获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，
依据许可证分发的软件是基于“原样”分发的，
无任何明示或暗示的担保或条件。有关自明的详细信息，请参阅许可证。

⚠️ 请注意，此文件是Markdown文件，但包含我们doc-builder的特定语法（类似MDX），
可能无法在您的Markdown查看器中正确显示。

-->

# CamemBERT

## 概述

CamemBERT模型是由Louis Martin，Benjamin Muller，Pedro Javier Ortiz Suárez，Yoann Dupont，
Laurent Romary，Éric Villemonte de la Clergerie，Djamé Seddah和Benoît Sagot在论文《CamemBERT: a Tasty French Language Model》中提出的。
它基于Facebook于2019年发布的RoBERTa模型。该模型基于138GB的法语文本进行了训练。

该论文的摘要如下：

*预训练语言模型现在在自然语言处理中普遍存在。尽管它们取得了成功，但大多数可用模型要么是在英语数据上进行训练的，
要么是在多种语言数据的拼接上训练的。这使得除英语以外的其他语言对这些模型的实际应用非常有限。为了解决这个问题，
我们发布了CamemBERT，这是BERT（双向编码器转换器）的法语版。我们通过在多种下游任务中比较CamemBERT与多语言模型的性能，
即词性标注，依赖语法分析，命名实体识别和自然语言推理。CamemBERT的性能在大多数任务中都优于现有的方法。我们希望通过发布CamemBERT的预训练模型，
促进法语自然语言处理的研究和应用。*

提示：

- 此实现与RoBERTa相同。有关用法示例以及输入和输出的相关信息，请参阅[RoBERTa的文档](roberta)。

该模型由[camembert](https://huggingface.co/camembert)贡献。原始代码可以在[此处](https://camembert-model.fr/)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## CamembertConfig

[[autodoc]] CamembertConfig

## CamembertTokenizer

[[autodoc]] CamembertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CamembertTokenizerFast

[[autodoc]] CamembertTokenizerFast

## CamembertModel

[[autodoc]] CamembertModel

## CamembertForCausalLM

[[autodoc]] CamembertForCausalLM

## CamembertForMaskedLM

[[autodoc]] CamembertForMaskedLM

## CamembertForSequenceClassification

[[autodoc]] CamembertForSequenceClassification

## CamembertForMultipleChoice

[[autodoc]] CamembertForMultipleChoice

## CamembertForTokenClassification

[[autodoc]] CamembertForTokenClassification

## CamembertForQuestionAnswering

[[autodoc]] CamembertForQuestionAnswering

## TFCamembertModel

[[autodoc]] TFCamembertModel

## TFCamembertForCasualLM

[[autodoc]] TFCamembertForCausalLM

## TFCamembertForMaskedLM

[[autodoc]] TFCamembertForMaskedLM

## TFCamembertForSequenceClassification

[[autodoc]] TFCamembertForSequenceClassification

## TFCamembertForMultipleChoice

[[autodoc]] TFCamembertForMultipleChoice

## TFCamembertForTokenClassification

[[autodoc]] TFCamembertForTokenClassification

## TFCamembertForQuestionAnswering

[[autodoc]] TFCamembertForQuestionAnswering
