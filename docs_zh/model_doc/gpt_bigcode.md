<!--版权所有2023年HuggingFace团队。 版权所有。

根据Apache许可证2.0版（“许可证”），除非符合许可证的要求, 否则您不得使用此文件。 您可以在以下网址获得许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件基于“原样”分发，没有任何形式的保证或条件。 有关许可证下对特定语言的限制和限制，请参阅许可证的特定语言。-->




# GPTBigCode（大型代码）

## 概述

GPTBigCode模型在BigCode的[论文“SantaCoder：不要舍近求远!”](https://arxiv.org/abs/2301.03988) 中提出。列出的作者是：Loubna Ben Allal，Raymond Li，Denis Kocetkov，Chenghao Mou，Christopher Akiki，Carlos Munoz Ferrandis，Niklas Muennighoff，Mayank Mishra，Alex Gu，Manan Dey，Logesh Kumar Umapathi，Carolyn Jane Anderson ，Yangtian Zi，Joel Lamy Poirier，Hailey Schoelkopf，Sergey Troshin，Dmitry Abulkhanov，Manuel Romero，Michael Lappert，Francesco De Toni，Bernardo García del Río，Qian Liu，Shamik Bose，Urvashi Bhattacharyya，Terry Yue Zhu，Ian Yu，Paulo Villegas，Marco Zocca，Souarab Mangrulkar，David Lansky，Huu Nguyen，Danish Contractor，Luis Villa，Jia Li，Dzmitry Bahdanau，Yacine Jernite，Sean Hughes，Daniel Fried，Arjun Guha，Harm de Vries，Leandro von Werra。

来自论文的摘要如下：

* BigCode项目是一个致力于代码大型语言模型的负责任开放科学协作。该技术报告描述了协作截至2022年12月的进展，概述了个人身份信息（PII）清除流程的当前状态，用于消除模型架构风险的实验，以及用于训练数据的更好预处理方法的实验。我们在The Stack的Java、JavaScript和Python子集上训练了11亿参数模型，并在MultiPL-E文本到代码基准测试中对其进行了评估。我们发现更积极地过滤近似重复值可以进一步提高性能，并且令人惊讶的是，从5个以上GitHub stars的存储库中选择文件会导致性能显着下降。尽管是一个规模更小的模型，但我们的最佳模型在Java、JavaScript和Python部分的MultiPL-E上的从左到右生成和填充中优于先前的开源多语言代码生成模型（InCoder-6.7B和CodeGen-Multi-2.7B）。所有模型都在[此https地址](https://huggingface.co/bigcode)下根据OpenRAIL许可证发布。*

该模型是一个具有多查询注意力支持的优化的[GPT2模型](https://huggingface.co/docs/transformers/model_doc/gpt2)。

## 技术细节

与GPT2相比的主要差异。
- 增加了对多查询注意力的支持。
- 使用`gelu_pytorch_tanh`替代经典的`gelu`。
- 避免不必要的同步（之后已在#20061加入GPT2，但未在参考代码库中加入）。
- 使用线性层替代Conv1D（加快速度但使检查点不兼容）。
- 合并“_attn”和“_upcast_and_reordered_attn”。始终将矩阵乘法与缩放合并。将`reorder_and_upcast_attn`重命名为`attention_softmax_in_fp32`。
- 缓存注意力掩码值，以避免每次重新创建它。
- 使用jit来融合注意力fp32转换、掩码、softmax和缩放。
- 将注意力和因果掩码合并为一个单一的掩码，针对整个模型进行预先计算，而不是每个层都进行预计算。
- 将密钥和值缓存合并为一个（这将更改layer_past/present的格式，会带来风险吗？）。
- 对于具有MHA的QKV张量，使用内存布局（self.num_heads, 3, self.head_dim）而不是`(3, self.num_heads, self.head_dim)`。（防止与原始gpt2模型的检查点产生开销）。

您可以在[原始请求中](https://github.com/huggingface/transformers/pull/22575)详细了解这些优化。

## GPTBigCodeConfig

[[autodoc]] GPTBigCodeConfig

## GPTBigCodeModel

[[autodoc]] GPTBigCodeModel
- forward

## GPTBigCodeForCausalLM

[[autodoc]] GPTBigCodeForCausalLM
- forward

## GPTBigCodeForSequenceClassification

[[autodoc]] GPTBigCodeForSequenceClassification
- forward

## GPTBigCodeForTokenClassification

[[autodoc]] GPTBigCodeForTokenClassification
- forward