<!--版权所有2021年HuggingFace团队。保留所有权利。
根据Apache许可证第2.0版（“许可证”）进行许可；除非符合
许可证，否则不得使用此文件。你可以在以下地址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证，只能在
“按原样”基础上分发软件，不提供任何形式的明示或暗示保证。
有关特定语法的详细信息，你可以阅读许可证，以了解是否可以
正确渲染在你的Markdown查看器中。-->

# Hubert

## 概述

Hubert是由Wei-Ning Hsu、Benjamin Bolte、Yao-Hung Hubert Tsai、Kushal Lakhotia、Ruslan Salakhutdinov和Abdelrahman Mohamed在[《HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units》](https://arxiv.org/abs/2106.07447)中提出的。

以下是该论文的摘要：

*自我监督的语音表示学习方法面临着三个独特问题的挑战：（1）每个输入话语中存在多个声音单位，（2）在预训练阶段没有输入声音单位的词典，（3）声音单位具有可变长度并且没有明确的分段。为了解决这些问题，我们提出了一种用于自我监督语音表示学习的Hidden-Unit BERT（HuBERT）方法，它利用离线聚类步骤为BERT-like预测损失提供对齐的目标标签。我们方法的一个关键要素是仅在掩蔽区域上应用预测损失，这强制模型在连续输入上学习一个结合声学模型和语言模型。HuBERT主要依赖于无监督聚类步骤的一致性，而不是分配的簇标签的内在质量。从简单的100个簇的k-means教师开始，通过两次迭代聚类，HuBERT模型在Librispeech（960h）和Libri-light（60,000h）基准测试中的10min、1h、10h、100h和960h微调子集上要么与wav2vec 2.0的最新性能相匹配，要么有所改进。使用具有10亿参数的模型，HuBERT在更具挑战性的dev-other和test-other评估子集上实现了高达19%和13%的相对WER降低。*

提示：

- Hubert是一个接受对应于语音信号原始波形的浮点数组的语音模型。
- Hubert模型是使用连续时间分类（CTC）进行微调的，所以模型输出必须使用[`Wav2Vec2CTCTokenizer`]进行解码。

该模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。

## 文档资源

- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)

## HubertConfig

[[autodoc]] HubertConfig

## HubertModel

[[autodoc]] HubertModel
    - forward

## HubertForCTC

[[autodoc]] HubertForCTC
    - forward

## HubertForSequenceClassification

[[autodoc]] HubertForSequenceClassification
    - forward

## TFHubertModel

[[autodoc]] TFHubertModel
    - call

## TFHubertForCTC

[[autodoc]] TFHubertForCTC
    - call