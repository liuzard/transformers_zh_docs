<!--版权 2023 年The HuggingFace团队。保留所有权利。

根据Apache许可证，版本2.0（“许可证”）授权;除非符合许可证，否则你不得使用此文件
。你可以在以下网址获取许可证副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的
,不提供任何明示或暗示的保证或条件。请参阅许可证以了解许可下的具体语言和限制。

⚠️ 请注意，此文件为Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），
这可能在你的Markdown查看器中无法正确呈现。-->

# GPTSAN-japanese

## 概览

GPTSAN-japanese模型由Toshiyuki Sakamoto（tanreinama）在仓库中发布。

GPTSAN是一个使用Switch Transformer的日语语言模型。它与T5论文中介绍的Prefix LM模型具有相同的结构，并支持文本生成和掩码语言建模任务。这些基本任务同样可以用于翻译或摘要的微调。

### 生成

可以使用`generate()`方法使用GPTSAN-Japanese模型生成文本。

```python
>>> from transformers import AutoModel, AutoTokenizer
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").cuda()
>>> x_tok = tokenizer("は、", prefix_text="織田信長", return_tensors="pt")
>>> torch.manual_seed(0)
>>> gen_tok = model.generate(x_tok.input_ids.cuda(), token_type_ids=x_tok.token_type_ids.cuda(), max_new_tokens=20)
>>> tokenizer.decode(gen_tok[0])
'織田信長は、2004年に『戦国BASARA』のために、豊臣秀吉'
```

## GPTSAN特点

GPTSAN具有一些独特的特点。它具有Prefix-LM模型的结构，并可作为移位掩码语言模型用于前缀输入标记。未加前缀的输入表现为常规生成模型。
Spout向量是GPTSAN的特殊输入。Spout是在微调过程中使用随机输入进行预训练的，但你可以在微调过程中指定文本类别或任意向量。这使得你可以表示所生成文本的趋势。
GPTSAN具有基于Switch-Transformer的稀疏前馈。你还可以添加其他层并对其进行部分训练。有关详细信息，请参阅原始的GPTSAN仓库。

### Prefix-LM模型

GPTSAN具有`T5`论文中称为Prefix-LM的模型结构。（原始的GPTSAN仓库称其为`hybrid`）
在GPTSAN中，Prefix-LM的`Prefix`部分，即可以被两个标记引用的输入位置，可以指定为任意长度。
对于每个批次，也可以单独指定不同的长度。
这个长度应用于Tokenizer中输入的文本的`prefix_text`。
Tokenizer将Prefix-LM的`Prefix`部分的掩码表示为`token_type_ids`。
模型将`token_type_ids`为1的部分视为`Prefix`部分，即，输入可以引用前后两个标记的部分。

提示：
指定前缀部分是通过传递给自注意力机制的掩码来完成的。
当token_type_ids=None或全部为零时，相当于常规因果屏蔽。

例如：
>>> x_token = tokenizer("ｱｲｳｴ")
input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |
prefix_lm_mask:
SOT | 1 0 0 0 0 0 |
SEG | 1 1 0 0 0 0 |
ｱ   | 1 1 1 0 0 0 |
ｲ   | 1 1 1 1 0 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 1 |

>>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |
prefix_lm_mask:
SOT | 1 1 1 1 1 0 |
ｱ   | 1 1 1 1 1 0 |
ｲ   | 1 1 1 1 1 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 0 |
SEG | 1 1 1 1 1 1 |

>>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
prefix_lm_mask:
SOT | 1 1 1 0 0 0 |
ｱ   | 1 1 1 0 0 0 |
ｲ   | 1 1 1 0 0 0 |
SEG | 1 1 1 1 0 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 1 |

### Spout向量

Spout向量是用于控制文本生成的特殊向量。
这个向量在自注意力机制中被视为第一个嵌入向量，以对生成的标记进行外部关注。
在从`Tanrei/GPTSAN-japanese`发布的预训练模型中，Spout向量是一个128维的向量，通过模型中的8个全连接层，并被投影到作为外部注意力的空间中。
由全连接层投影的Spout向量被分割为传递给所有自注意机制的部分。

## GPTSanJapaneseConfig

[[autodoc]] GPTSanJapaneseConfig

## GPTSanJapaneseTokenizer

[[autodoc]] GPTSanJapaneseTokenizer

## GPTSanJapaneseModel

[[autodoc]] GPTSanJapaneseModel

## GPTSanJapaneseForConditionalGeneration

[[autodoc]] GPTSanJapaneseForConditionalGeneration
    - forward