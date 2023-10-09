<!--版权保留-->
版权所有2020 HuggingFace团队。已经获得许可根据Apache许可证第2版（“许可证”）进行使用此文件，除非符合许可证中的要求，否则你不得使用此文件。你可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件在许可证下分发时是按“原样”的基础分发的，不附带任何明示或暗示的条件或保证。详细了解许可证的特定语言，请参阅许可证。

⚠️请注意，此文件以Markdown格式编写，但包含特定于我们doc-builder的语法（类似于MDX），可能无法在你的Markdown查看器中正确渲染。

# Pegasus

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=pegasus">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-pegasus-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/pegasus_paraphrase">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**免责声明：**如果你发现有什么奇怪的地方，请[提交Github问题](https://github.com/huggingface/transformers/issues/new?assignees=sshleifer&labels=&template=bug-report.md&title)并@patrickvonplaten。

## 概览

Pegasus模型是由Jingqing Zhang、Yao Zhao、Mohammad Saleh和Peter J. Liu于2019年12月18日在[文章](https://arxiv.org/pdf/1912.08777.pdf)中提出的。

根据摘要，

- Pegasus的预训练任务与摘要相似：从输入文档中删除/屏蔽重要句子，并将其合并为一条输出序列，类似于摘要提取。
- Pegasus在所有12个下游任务中都实现了最先进的摘要性能，ROUGE和人工评估结果。

该模型由[sshleifer](https://huggingface.co/sshleifer)贡献。作者的代码可以在[这里](https://github.com/google-research/pegasus)找到。

提示：

- 与BART相同的编码器-解码器模型架构的序列到序列模型。Pegasus使用两个自监督目标函数一起进行预训练：遮蔽语言建模（MLM）和一种新颖的摘要特定预训练目标，称为Gap Sentence Generation (GSG)。

  * MLM：编码器输入标记被随机替换为遮蔽标记，并由编码器预测（类似于BERT）
  * GSG：将整个编码器输入句子替换为第二个遮蔽标记，并馈送到解码器，但带有一个因果性遮蔽以隐藏未来单词，就像一个常规自回归变压器解码器一样。

## 检查点

除了*pegasus-large*，所有的[检查点](https://huggingface.co/models?search=pegasus)都是为摘要化微调的。

- 每个检查点在磁盘上占用2.2 GB，具有568M个参数。
- 不支持FP16（对此的帮助/想法将不胜感激！）。
- 在v100 GPU上，使用默认参数汇总xsum大约需要400ms/样本。
- 完整的复制结果和正确预处理的数据可以在这个[问题](https://github.com/huggingface/transformers/issues/6844#issue-689259666)中找到。
- [精简检查点](https://huggingface.co/models?search=distill-pegasus)在这篇[文章](https://arxiv.org/abs/2010.13002)中进行了描述。

### 示例

- 使用脚本在XSUM数据集上进行微调pegasus。数据下载说明在[这里](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md)。
- 不支持FP16（对此的帮助/想法将不胜感激！）。
- 推荐使用adafactor优化器进行pegasus微调。

## 实现说明

- 所有的模型都是具有16层的transformer编码器-解码器组件。
- 实现完全继承自[`BartForConditionalGeneration`]。
- 一些关键的配置差异：

  - 静态，正弦的位置嵌入
  - 模型从pad_token_id（其具有0个token_embedding）作为前缀开始生成。
  - 使用更多的beam（`num_beams=8`）。
- 所有预训练的pegasus检查点除了三个属性之外是相同的：`tokenizer.model_max_length`（最大输入大小），`max_length`（要生成的最大标记数量）和`length_penalty`。
- 可以在作者的[repo](https://github.com/google-research/pegasus)中找到将训练的检查点转换为pytorch的代码`convert_pegasus_tf_to_pytorch.py`。

## 使用示例

```python
>>> from transformers import PegasusForConditionalGeneration, PegasusTokenizer
>>> import torch

>>> src_text = [
...     """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
... ]

... model_name = "google/pegasus-xsum"
... device = "cuda" if torch.cuda.is_available() else "cpu"
... tokenizer = PegasusTokenizer.from_pretrained(model_name)
... model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
... batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
... translated = model.generate(**batch)
... tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
... assert (
...     tgt_text[0]
...     == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
... )
```

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## PegasusConfig

[[autodoc]] PegasusConfig

## PegasusTokenizer

警告：`add_tokens`目前不可用。

[[autodoc]] PegasusTokenizer

## PegasusTokenizerFast

[[autodoc]] PegasusTokenizerFast

## PegasusModel

[[autodoc]] PegasusModel
    - 正向传播

## PegasusForConditionalGeneration

[[autodoc]] PegasusForConditionalGeneration
    - 正向传播

## PegasusForCausalLM

[[autodoc]] PegasusForCausalLM
    - 正向传播

## TFPegasusModel

[[autodoc]] TFPegasusModel
    - 调用

## TFPegasusForConditionalGeneration

[[autodoc]] TFPegasusForConditionalGeneration
    - 调用

## FlaxPegasusModel

[[autodoc]] FlaxPegasusModel
    - __call__
    - 编码
    - 解码

## FlaxPegasusForConditionalGeneration

[[autodoc]] FlaxPegasusForConditionalGeneration
    - __call__
    - 编码
    - 解码