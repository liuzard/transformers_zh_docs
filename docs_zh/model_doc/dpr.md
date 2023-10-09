<!--版权所有2020年HuggingFace团队保留。

根据Apache许可证2.0版（“许可证”）的规定，除非符合许可证的规定，否则你不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非依法要求或书面同意，否则根据许可证分发的软件是基于“按原样”（“AS IS"）的基础分发的，不附带任何明示或暗示的保证或条件。请参阅许可证中的详细规定以了解特定的语言权限和限制。

⚠️请注意，此文件采用Markdown格式，并包含我们文档生成器的特定语法（类似MDX），可能无法在你的Markdown查看器中正确渲染。-->

# DPR

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=dpr">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-dpr-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/dpr-question_encoder-bert-base-multilingual">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

DPR（Dense Passage Retrieval）是一组用于最先进开放领域问答研究的工具和模型。它在[Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih的 "Dense Passage Retrieval for Open-Domain Question Answering"](https://arxiv.org/abs/2004.04906)中被提出。

该论文的摘要如下：

*在开放领域问答中，依赖于高效的段落检索以选择候选上下文，其中传统的稀疏向量空间模型（如TF-IDF或BM25）是事实上的方法。在这项工作中，我们展示了检索可以仅使用密集表示来实现，其中通过简单的双编码器框架从少量问题和段落中学习嵌入。在各种开放领域QA数据集上进行评估时，我们的密集检索器在前20个段落检索准确性方面大大超过强大的Lucene-BM25系统，相对提高了9-19％，并帮助我们的端到端QA系统建立起多个开放领域QA基准的最新技术水平。*

本模型由[lhoestq](https://huggingface.co/lhoestq)贡献。原始代码可以在[此处](https://github.com/facebookresearch/DPR)找到。

提示：
- DPR包括三个模型：

     * 问题编码器：将问题编码为向量
     * 上下文编码器：将上下文编码为向量
     * Reader：从检索到的上下文中提取问题的答案，并提供相关度评分（如果推断出的范围实际上回答问题，则得分高）。

## DPRConfig

[[autodoc]] DPRConfig

## DPRContextEncoderTokenizer

[[autodoc]] DPRContextEncoderTokenizer

## DPRContextEncoderTokenizerFast

[[autodoc]] DPRContextEncoderTokenizerFast

## DPRQuestionEncoderTokenizer

[[autodoc]] DPRQuestionEncoderTokenizer

## DPRQuestionEncoderTokenizerFast

[[autodoc]] DPRQuestionEncoderTokenizerFast

## DPRReaderTokenizer

[[autodoc]] DPRReaderTokenizer

## DPRReaderTokenizerFast

[[autodoc]] DPRReaderTokenizerFast

## DPR特定的输出

[[autodoc]] models.dpr.modeling_dpr.DPRContextEncoderOutput

[[autodoc]] models.dpr.modeling_dpr.DPRQuestionEncoderOutput

[[autodoc]] models.dpr.modeling_dpr.DPRReaderOutput

## DPRContextEncoder

[[autodoc]] DPRContextEncoder
    - forward

## DPRQuestionEncoder

[[autodoc]] DPRQuestionEncoder
    - forward

## DPRReader

[[autodoc]] DPRReader
    - forward

## TFDPRContextEncoder

[[autodoc]] TFDPRContextEncoder
    - call

## TFDPRQuestionEncoder

[[autodoc]] TFDPRQuestionEncoder
    - call

## TFDPRReader

[[autodoc]] TFDPRReader
    - call