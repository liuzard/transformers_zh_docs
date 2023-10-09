<!--版权所有2023年HuggingFace团队。保留所有权利。

根据Apache许可证版本2.0（“许可证”），除非符合许可证，
否则不得使用此文件。您可以在下面的链接地址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，基于许可证的软件是按照“原样”分发的，
不附带任何明示或暗示的担保或条件。详细了解许可证中的权限和限制。-->

# BROS

## 概述

BROS模型是由Teakgyu Hong、Donghyun Kim、Mingi Ji、Wonseok Hwang、Daehyun Nam、Sungrae Park在文献[BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents](https://arxiv.org/abs/2108.04539)中提出的。 

BROS代表BERT Relying On Spatiality。它是一个仅编码器的Transformer模型，接收一系列令牌及其边界框作为输入，并输出一系列隐藏状态。BROS使用相对空间信息编码，而不使用绝对空间信息。

它采用了两个预训练目标：词语掩盖语言建模目标（TMLM），用于BERT，以及新的区域掩盖语言建模目标（AMLM）。在TMLM中，令牌被随机掩盖，模型使用空间信息和其他未被掩盖的令牌来预测被掩盖的令牌。AMLM是TMLM的二维版本。它随机掩盖文本令牌，并使用与TMLM相同的信息进行预测，但掩盖的是文本块（区域）。

`BrosForTokenClassification`在BrosModel之上有一个简单的线性层，用于预测每个令牌的标签。
`BrosSpadeEEForTokenClassification`在BrosModel之上有一个`initial_token_classifier`和一个`subsequent_token_classifier`。`initial_token_classifier`用于预测每个实体的第一个令牌，`subsequent_token_classifier`用于预测实体内部的下一个令牌。`BrosSpadeELForTokenClassification`在BrosModel之上有一个`entity_linker`。`entity_linker`用于预测两个实体之间的关系。

`BrosForTokenClassification`和`BrosSpadeEEForTokenClassification`本质上执行相同的任务。然而，`BrosForTokenClassification`假设输入令牌被完美地序列化（这是一个非常具有挑战性的任务，因为它们存在于2D空间中），而`BrosSpadeEEForTokenClassification`则允许在处理序列化错误方面更有灵活性，因为它从一个令牌预测下一个连接令牌。

`BrosSpadeELForTokenClassification`执行内部实体链接任务。如果两个实体之间存在某种关系，则从一个令牌（一个实体的令牌）预测到另一个令牌（另一个实体的令牌）的关系。

BROS在关键信息提取（KIE）基准测试中，如FUNSD、SROIE、CORD和SciTSR等方面达到可比或更好的结果，而无需依赖显式的视觉特征。

该论文的摘要如下：

*从文档图像中提取关键信息（KIE）需要理解二维（2D）空间中文本的上下文语义和空间语义。许多最近的研究试图通过开发专注于将文档图像的视觉特征与文本及其布局相结合的预训练语言模型来解决这一任务。另一方面，本文通过回归基本问题来解决这个问题：文本和布局的有效组合。具体来说，我们提出了一种预训练语言模型，名为BROS（BERT Relying On Spatiality），它对2D空间中的文本的相对位置进行编码，并采用未标记文档进行学习区域掩蔽策略。通过这种优化的训练方案来理解2D空间中的文本，BROS在四个KIE基准测试（FUNSD、SROIE*、CORD和SciTSR）上表现出与以往方法相比可比或更好的性能，而无需依赖于视觉特征。本文还揭示了KIE任务中的两个现实挑战——（1）减少来自错误文本排序的错误和（2）从较少的下游示例中进行高效学习，并展示了BROS相比以往方法的优越性。*

提示：

- [`~transformers.BrosModel.forward`] 需要 `input_ids` 和 `bbox`（边界框）。每个边界框应采用（x0，y0，x1，y1）格式（左上角，右下角）。获取边界框取决于外部OCR系统。`x` 坐标应通过文档图像宽度进行归一化，`y` 坐标应该通过文档图像高度进行归一化。

```python
def expand_and_normalize_bbox(bboxes, doc_width, doc_height):
    # 这里，bboxes 是一个numpy数组

    # 归一化边界框 -> 0 ~ 1
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / height
```

- [`~transformers.BrosForTokenClassification.forward`、`~transformers.BrosSpadeEEForTokenClassification.forward`、`~transformers.BrosSpadeEEForTokenClassification.forward`] 不仅需要 `input_ids` 和 `bbox`，还需要用于损失计算的 `box_first_token_mask`。这是一个掩蔽非第一个令牌的每个框的掩码。您可以通过在创建`input_ids`时保存边界框的起始令牌索引来获取该掩码。您可以使用以下代码生成`box_first_token_mask`，
    
```python
def make_box_first_token_mask(bboxes, words, tokenizer, max_seq_length=512):

    box_first_token_mask = np.zeros(max_seq_length, dtype=np.bool_)

    # 对来自 words（List[str]）的每个词进行编码（词汇化）
    input_ids_list: List[List[int]] = [tokenizer.encode(e, add_special_tokens=False) for e in words]

    # 获取每个框的长度
    tokens_length_list: List[int] = [len(l) for l in input_ids_list]

    box_end_token_indices = np.array(list(itertools.accumulate(tokens_length_list)))
    box_start_token_indices = box_end_token_indices - np.array(tokens_length_list)

    # 过滤出超出 max_seq_length 的索引
    box_end_token_indices = box_end_token_indices[box_end_token_indices < max_seq_length - 1]
    if len(box_start_token_indices) > len(box_end_token_indices):
        box_start_token_indices = box_start_token_indices[: len(box_end_token_indices)]

    # 将 box_start_token_indices 设置为 True
    box_first_token_mask[box_start_token_indices] = True

    return box_first_token_mask

```

- 演示脚本可以在[这里](https://github.com/clovaai/bros)找到。

此模型由[jinho8345](https://huggingface.co/jinho8345)贡献。原始代码可以在[这里](https://github.com/clovaai/bros)找到。

## BrosConfig

[[autodoc]] BrosConfig

## BrosProcessor

[[autodoc]] BrosProcessor
    - __call__

## BrosModel

[[autodoc]] BrosModel
    - forward


## BrosForTokenClassification

[[autodoc]] BrosForTokenClassification
    - forward


## BrosSpadeEEForTokenClassification

[[autodoc]] BrosSpadeEEForTokenClassification
    - forward


## BrosSpadeELForTokenClassification

[[autodoc]] BrosSpadeELForTokenClassification
    - forward