<!--版权所有2020年HuggingFace团队。版权所有。

根据Apache许可证，版本2.0（“许可证”）的规定，您不得使用此文件，除非符合许可证的规定。
您可以在http://www.apache.org/licenses/LICENSE-2.0获得许可证的副本。

除非适用法律要求或书面同意，根据许可证分发软件是根据“按原样”的基础分发的，不附带任何明示或暗示的保证或条件。有关详细信息，请参见许可证下的限制事项。

⚠️请注意，此文件使用Markdown格式，但包含了特定于我们文档构建工具（类似于MDX）的语法，这在您的Markdown查看器中可能无法正确渲染。

-->

# FSMT

**免责声明：**如果您发现了异常情况，请在[Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)中进行报告，并分派@stas00。

## 概述

FSMT（FairSeq MachineTranslation）模型是由Nathan Ng，Kyra Yee，Alexei Baevski，Myle Ott，Michael Auli和Sergey Edunov在[Facebook FAIR's WMT19 News Translation Task Submission](https://arxiv.org/abs/1907.06616)中首次提出的。

该论文的摘要如下：

*本文介绍了Facebook FAIR在WMT19共享新闻翻译任务中的提交。我们参与了两种语言对和四种语言方向，即英语<->德语和英语<->俄语。与去年的提交一样，我们的基线系统是使用Fairseq序列建模工具训练的基于大型BPE的Transformer模型，这些模型依赖于采样的反向翻译。今年，我们尝试了不同的双语数据过滤方案，并添加了经过过滤的反向翻译数据。我们还对领域特定数据进行了集成和微调，然后使用噪声通道模型重新排序进行解码。我们的提交在人类评估活动的所有四个方向上排名第一。在En->De方向上，我们的系统显著优于其他系统以及人工翻译。该系统相比我们的WMT'18提交提高了4.5 BLEU点。*

该模型由[stas](https://huggingface.co/stas)贡献。原始代码可以在[这里](https://github.com/pytorch/fairseq/tree/master/examples/wmt19)找到。

## 实现说明

- FSMT使用未合并成一个的源词汇和目标词汇对。它也不共享嵌入令牌。其分词器非常类似于[`XLMTokenizer`]，主要模型派生自[`BartModel`]。

## FSMTConfig

[[autodoc]] FSMTConfig

## FSMTTokenizer

[[autodoc]] FSMTTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FSMTModel

[[autodoc]] FSMTModel
    - forward

## FSMTForConditionalGeneration

[[autodoc]] FSMTForConditionalGeneration
    - forward