<!--版权所有2020年The HuggingFace团队，保留所有权利。

根据Apache License，第2.0版（“许可证”），你除非符合许可证的规定，否则不得使用此文件。你可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据许可证分发的软件是"按原样"的方式分发的，不附带任何明示或暗示的保证或条件。请参阅许可证中的特定语言以及许可证下的限制。

⚠️ 请注意，此文件采用Markdown格式，但包含我们的文档生成器（类似于MDX）的特定语法，你在Markdown查看器中渲染可能不正确。-->

# XLNet

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=xlnet">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-xlnet-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/xlnet-base-cased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 概述

XLNet模型是由Zhilin Yang，Zihang Dai，Yiming Yang，Jaime Carbonell，Ruslan Salakhutdinov，Quoc V. Le在[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)中提出的。XLNet是Transformer-XL模型的扩展，使用自回归方法进行预训练，通过最大化输入序列分解顺序的所有排列的期望似然来学习双向上下文。

论文中的摘要如下：

*通过对双向上下文进行建模，基于去噪自编码的预训练（例如BERT）比基于自回归语言建模的预训练方法具有更好的性能。然而，BERT依赖于使用屏蔽掩码破坏输入，并且在屏蔽的位置之间忽略依赖性，从而导致预训练和微调之间的不一致。鉴于这些优点和缺点，我们提出了XLNet，这是一种广义的自回归预训练方法，它(1)通过最大化因子分解顺序的所有排列的期望似然来实现学习双向上下文，(2)通过其自回归公式克服了BERT的限制。此外，XLNet将Transformer-XL（现有最先进的自回归模型）的思想整合到了预训练中。在可比的实验设置下，XLNet在包括问题回答、自然语言推理、情感分析和文档排名在内的20个任务中表现出色，往往相差很大。*

提示:

- 可以使用`perm_mask`输入在训练和测试时控制特定的注意模式。
- 由于在各种分解顺序上训练完全自回归的模型的困难，XLNet只使用选定的目标映射输入作为目标进行预训练。
- 要将XLNet用于顺序解码（即非完全双向设置），请使用`perm_mask`和`target_mapping`输入来控制注意范围和输出（参见*examples/pytorch/text-generation/run_generation.py*中的示例）。
- XLNet是为数不多没有序列长度限制的模型之一。
- XLNet不是传统的自回归模型，而是使用了建立在此基础之上的训练策略。它对句子中的标记进行排列，然后允许模型使用最后n个标记来预测第n+1个标记。由于这一切都是通过掩码完成的，所以句子实际上是按正确顺序输入模型的，但是XLNet使用一个隐藏了前面标记的掩码，顺序是给定的1,…，序列长度的一种置换。
- XLNet还使用与Transformer-XL相同的递归机制来构建长期依赖关系。

此模型由[thomwolf](https://huggingface.co/thomwolf)贡献。原始代码可在[此处](https://github.com/zihangdai/xlnet/)找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [token分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## XLNetConfig

[[autodoc]] XLNetConfig

## XLNetTokenizer

[[autodoc]] XLNetTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLNetTokenizerFast

[[autodoc]] XLNetTokenizerFast

## XLNet特定的输出

[[autodoc]] models.xlnet.modeling_xlnet.XLNetModelOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetModelOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetLMHeadModelOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForSequenceClassificationOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForMultipleChoiceOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForTokenClassificationOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForQuestionAnsweringSimpleOutput

## XLNetModel

[[autodoc]] XLNetModel
    - forward

## XLNetLMHeadModel

[[autodoc]] XLNetLMHeadModel
    - forward

## XLNetForSequenceClassification

[[autodoc]] XLNetForSequenceClassification
    - forward

## XLNetForMultipleChoice

[[autodoc]] XLNetForMultipleChoice
    - forward

## XLNetForTokenClassification

[[autodoc]] XLNetForTokenClassification
    - forward

## XLNetForQuestionAnsweringSimple

[[autodoc]] XLNetForQuestionAnsweringSimple
    - forward

## XLNetForQuestionAnswering

[[autodoc]] XLNetForQuestionAnswering
    - forward

## TFXLNetModel

[[autodoc]] TFXLNetModel
    - call

## TFXLNetLMHeadModel

[[autodoc]] TFXLNetLMHeadModel
    - call

## TFXLNetForSequenceClassification

[[autodoc]] TFXLNetForSequenceClassification
    - call

## TFLNetForMultipleChoice

[[autodoc]] TFXLNetForMultipleChoice
    - call

## TFXLNetForTokenClassification

[[autodoc]] TFXLNetForTokenClassification
    - call

## TFXLNetForQuestionAnsweringSimple

[[autodoc]] TFXLNetForQuestionAnsweringSimple
    - call