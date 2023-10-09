<!--版权所有2020 HuggingFace团队。

根据Apache许可证第2.0版（"许可证"）的规定，你不得使用此文件，除非符合许可证的要求。
你可以在以下网址获得许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用的法律要求或书面同意，根据本许可证分发的软件是基于"按原样"的基础上分发的，
没有任何明示或暗示的保证或条件。请参阅许可证规定，获取相关的语言组合和限制详情。

⚠️请注意，此文件以Markdown格式编写，但包含了与我们的文档生成工具相类似的特定语法（类似MDX），
在Markdown查看器中可能无法正确显示。-->

# LED

## 概述

LED模型由Iz Beltagy、Matthew E. Peters和Arman Cohan在[《Longformer：长文档变换器》](https://arxiv.org/abs/2004.05150)中提出。

论文的摘要如下：

*基于Transformer的模型不能处理长序列，因为其自我注意操作与序列长度成二次关系。为了解决这个限制，我们引入了具有线性序列长度缩放的注意机制的Longformer，使其易于处理成千上万个令牌或更长的文档。Longformer的注意机制是标准自我注意机制的一种即插即用替代品，将本地窗口注意力与任务动机全局注意力相结合。沿用以前关于长序列变换器的工作，我们在字符级语言建模上评估了Longformer，并在text8和enwik8上取得了最先进的结果。与大多数先前的工作相反，我们还对Longformer进行了预训练，并在各种下游任务上进行了微调。我们预训练的Longformer在长文档任务上始终优于RoBERTa，并在WikiHop和TriviaQA上取得了新的最先进结果。最后，我们引入了用于支持长文档生成序列到序列任务的Longformer-Encoder-Decoder（LED），并证明其在arXiv摘要数据集上的有效性。*

提示：

- [`LEDForConditionalGeneration`]是[`BartForConditionalGeneration`]的扩展，用*Longformer*的*分块自我注意力*层替换了传统的*自注意力*层。[`LEDTokenizer`]是[`BartTokenizer`]的别名。
- LED非常适用于`input_ids`远远超过1024个令牌的长距离*序列到序列*任务。
- 如果需要，LED将`input_ids`填充为`config.attention_window`的倍数，这样可以获得一小部分加速，当使用`pad_to_multiple_of`参数与 [`LEDTokenizer`]结合使用时。
- LED使用*全局注意力*通过`global_attention_mask`（参见[`LongformerModel`]）来表示。对于摘要，建议只在第一个`<s>`标记上放置*全局注意力*。对于问答，建议对所有问题的标记放置*全局注意力*。
- 为了在所有16384个序列上对LED进行微调，可以在训练导致内存不足（OOM）错误时启用*梯度检查点*，执行`model.gradient_checkpointing_enable()`即可。此外，可以使用`use_cache=False`标志禁用缓存机制以节省内存。
- [此处](https://colab.research.google.com/drive/12INTTR6n64TzS4RrXZxMSXfrOd9Xzamo?usp=sharing)提供了一个展示如何评估LED的笔记本。
- [此处](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing)提供了一个展示如何对LED进行微调的笔记本。
- LED是一个带有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧填充输入。

此模型由[patrickvonplaten](https://huggingface.co/patrickvonplaten)贡献。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## LEDConfig

[[autodoc]] LEDConfig

## LEDTokenizer

[[autodoc]] LEDTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LEDTokenizerFast

[[autodoc]] LEDTokenizerFast

## LED特定输出

[[autodoc]] models.led.modeling_led.LEDEncoderBaseModelOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqModelOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqLMOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput

[[autodoc]] models.led.modeling_tf_led.TFLEDEncoderBaseModelOutput

[[autodoc]] models.led.modeling_tf_led.TFLEDSeq2SeqModelOutput

[[autodoc]] models.led.modeling_tf_led.TFLEDSeq2SeqLMOutput

## LEDModel

[[autodoc]] LEDModel
    - forward

## LEDForConditionalGeneration

[[autodoc]] LEDForConditionalGeneration
    - forward

## LEDForSequenceClassification

[[autodoc]] LEDForSequenceClassification
    - forward

## LEDForQuestionAnswering

[[autodoc]] LEDForQuestionAnswering
    - forward

## TFLEDModel

[[autodoc]] TFLEDModel
    - call

## TFLEDForConditionalGeneration

[[autodoc]] TFLEDForConditionalGeneration
    - call