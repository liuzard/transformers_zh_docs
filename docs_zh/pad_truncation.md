<!--版权所有2022年HuggingFace团队保留。

根据Apache许可证2.0版（"许可证"）许可；除非符合许可证规定，否则不能使用此文件。
你可以在以下网址获得许可证副本

http://www.apache.org/licenses/LICENSE-2.0

请注意，此文件是Markdown格式，但包含我们文档生成器的特定语法（类似于MDX），你的Markdown查看器可能无法正确渲染。

-->

# 填充和截断

批量输入通常具有不同的长度，因此无法转换为固定大小的张量。填充和截断是处理此问题的策略，可以从不同长度的批次创建矩形张量。填充添加一个特殊的**填充标记**，以确保较短的序列与批次中最长的序列或模型接受的最大长度具有相同的长度。截断从另一个方向工作，通过截断长序列来缩短长度。

在大多数情况下，将批次填充到最长序列的长度，并截断到模型能够接受的最大长度通常效果不错。但是，如果你需要，API还支持更多策略。你需要的三个参数是：`padding`，`truncation`和`max_length`。

`padding`参数控制填充。它可以是布尔值或字符串：

  - `True`或`'longest'`：填充到批次中最长的序列长度（如果你只提供单个序列，则不会应用填充）。
  - `'max_length'`：填充到由`max_length`参数指定的长度或模型接受的最大长度（如果未提供`max_length`）（`max_length=None`）。如果只提供单个序列，仍然会应用填充。
  - `False`或`'do_not_pad'`：不应用填充。这是默认行为。

`truncation`参数控制截断。它可以是布尔值或字符串：

  - `True`或`'longest_first'`：截断到由`max_length`参数指定的最大长度或模型接受的最大长度（如果未提供`max_length`）（`max_length=None`）。这将逐个标记进行截断，从对中最长的序列中删除一个标记，直到达到适当的长度。
  - `'only_second'`：截断到由`max_length`参数指定的最大长度或模型接受的最大长度（如果未提供`max_length`）（`max_length=None`）。如果提供了序列对（或序列对的批次），则仅截断第二个句子。
  - `'only_first'`：截断到由`max_length`参数指定的最大长度或模型接受的最大长度（如果未提供`max_length`）（`max_length=None`）。如果提供了序列对（或序列对的批次），则仅截断第一个句子。
  - `False`或`'do_not_truncate'`：不应用截断。这是默认行为。

`max_length`参数控制填充和截断的长度。它可以是整数或`None`，在这种情况下，它将默认为模型可以接受的最大长度。如果模型没有特定的最大输入长度，则截断或填充到`max_length`将被禁用。

下表总结了设置填充和截断的推荐方式。如果在以下示例中使用输入序列对，可以将`truncation=True`替换为在`['only_first', 'only_second', 'longest_first']`中选择的`STRATEGY`，即`truncation='only_second'`或`truncation='longest_first'`，以控制对序列对中的两个序列如前所述进行截断。

| 截断                       | 填充                       | 指令                                                      |
|---------------------------|---------------------------|----------------------------------------------------------|
| 不截断                    | 不填充                    | `tokenizer(batch_sentences)`                            |
|                           | 填充到批次中最长序列长度  | `tokenizer(batch_sentences, padding=True)` 或                    |
|                           |                           | `tokenizer(batch_sentences, padding='longest')`                 |
|                           | 填充到模型最大输入长度    | `tokenizer(batch_sentences, padding='max_length')`              |
|                           | 填充到特定长度            | `tokenizer(batch_sentences, padding='max_length', max_length=42)`|
|                           | 填充到值的倍数            | `tokenizer(batch_sentences, padding=True, pad_to_multiple_of=8)` |
| 截断到模型最大输入长度    | 不填充                    | `tokenizer(batch_sentences, truncation=True)` 或                |
|                           |                           | `tokenizer(batch_sentences, truncation=STRATEGY)`               |
|                           | 填充到批次中最长序列长度  | `tokenizer(batch_sentences, padding=True, truncation=True)` 或   |
|                           |                           | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY)`  |
|                           | 填充到模型最大输入长度    | `tokenizer(batch_sentences, padding='max_length', truncation=True)` 或|
|                           |                           | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY)`|
|                           | 填充到特定长度            | 无法实现                                                    |
| 截断到特定长度            | 不填充                    | `tokenizer(batch_sentences, truncation=True, max_length=42)` 或   |
|                           |                           | `tokenizer(batch_sentences, truncation=STRATEGY, max_length=42)`  |
|                           | 填充到批次中最长序列长度  | `tokenizer(batch_sentences, padding=True, truncation=True, max_length=42)` 或|
|                           |                           | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY, max_length=42)`|
|                           | 填充到模型最大输入长度    | 无法实现                                                    |
|                           | 填充到特定长度            | `tokenizer(batch_sentences, padding='max_length', truncation=True, max_length=42)` 或|
|                           |                           | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY, max_length=42)`|