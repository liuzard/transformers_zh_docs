<!--版权2022年，HuggingFace团队。版权所有。

根据Apache许可证2.0版（“许可证”），你不得使用此文件，除非符合许可证的规定。
你可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按现状”提供的，
没有任何明示或暗示的担保或条件。有关许可证下特定语言的限制和条件，请参阅许可证。

⚠️请注意，此文件是Markdown格式，但包含我们的文档生成器（类似于MDX）的特定语法，可能在你的Markdown查看器中无法正常渲染。

-->

# 时间序列Transformer

<Tip>

这是一个最近引入的模型，因此尚未对API进行广泛测试。未来可能会有一些错误或轻微的变化来修复它。如果你看到一些奇怪的情况，请[提交一个Github问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

</Tip>

## 概述

时间序列Transformer模型是一种用于时间序列预测的普通编码器-解码器Transformer模型。

提示：

- 类似于库中的其他模型，[`TimeSeriesTransformerModel`] 是不带任何头部的原始Transformer，[`TimeSeriesTransformerForPrediction`]在前者之上添加了一个分布头，可用于时间序列预测。请注意，这是一种所谓的概率预测模型，而不是点预测模型。这意味着模型学习一个分布，可以从中进行采样，而不是直接输出值。
- [`TimeSeriesTransformerForPrediction`]由两个模块组成：编码器和解码器。编码器将时间序列值的`context_length`（称为`past_values`）作为输入，解码器将预测未来`prediction_length`个时间序列值（称为`future_values`）。在训练期间，需要向模型提供（`past_values`和`future_values`）的配对。
- 除了原始的（`past_values`和`future_values`）之外，通常还会向模型提供其他特征。这些特征可以是以下内容：
     - `past_time_features`：模型将这些特征添加到`past_values`中的时间特征。这些特征作为Transformer编码器的“位置编码”使用。例如，“月份的第一天”，“年份的第一个月”等作为标量值（然后作为向量堆叠在一起）。
     例如，如果某个时间序列值是在8月11日获得的，则可以将[11, 8]作为时间特征向量（11是“月份的第一天”，8是“年份的第一个月”）。
     - `future_time_features`：模型将这些特征添加到`future_values`中的时间特征。这些特征作为Transformer解码器的“位置编码”使用。例如，“月份的第一天”，“年份的第一个月”等作为标量值（然后作为向量堆叠在一起）。
     例如，如果某个时间序列值是在8月11日获得的，则可以将[11, 8]作为时间特征向量（11是“月份的第一天”，8是“年份的第一个月”）。
     - `static_categorical_features`：随时间不变的分类特征（即所有`past_values`和`future_values`具有相同值的特征）。这里的一个示例是标识给定时间序列的商店ID或地区ID。请注意，这些特征需要对所有数据点（包括未来的数据点）进行知道。
     - `static_real_features`：随时间不变的实值特征（即所有`past_values`和`future_values`具有相同值的特征）。这里的一个示例是你具有时间序列值的产品的图像表示（例如“鞋子”图片的[ResNet](resnet)嵌入）。请注意，这些特征需要对所有数据点（包括未来的数据点）进行知道。
- 模型使用“teacher-forcing”进行训练，类似于Transformer用于机器翻译的训练方式。这意味着在训练期间，将`future_values`向右移动一个位置作为解码器的输入，由`past_values`的最后一个值预置。在每个时间步，模型需要预测下一个目标。因此，训练的设置类似于语言的GPT模型，只是没有`decoder_start_token_id`的概念（我们只是将上下文的最后一个值作为解码器的初始输入）。
- 在推理时，我们将`past_values`的最后一个值作为解码器的输入。接下来，我们可以从模型中进行采样，以在下一个时间步进行预测，然后将预测结果馈送到解码器，以进行下一个预测（也称为自回归生成）。

这个模型由[kashif](https://huggingface.co/kashif)提供。

## 资源

官方的Hugging Face和社区（由🌎表示）资源列表，以帮助你入门。如果你有兴趣提交资源以包含在这里，请随时提出拉取请求，我们将进行审查！这个资源应该展示一些新东西，而不是重复现有的资源。

- 在HuggingFace博客中查看关于时间序列Transformer的博文：[使用🤗 Transformers进行概率时间序列预测](https://huggingface.co/blog/time-series-transformers)


## TimeSeriesTransformerConfig

[[autodoc]] TimeSeriesTransformerConfig


## TimeSeriesTransformerModel

[[autodoc]] TimeSeriesTransformerModel
    - forward


## TimeSeriesTransformerForPrediction

[[autodoc]] TimeSeriesTransformerForPrediction
    - forward